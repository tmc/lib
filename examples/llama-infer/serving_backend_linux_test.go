//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

type fakeServingBackend struct {
	mu      sync.Mutex
	closed  bool
	prefill chan prefillObservation
}

func TestRealCUDAContinuousServing(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.MaxSeqLen = 8
	engine, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	prompts := [][]int32{{41, 42}, {43, 44}}
	want := make([][]int32, len(prompts))
	for i, prompt := range prompts {
		cache, err := engine.NewSequenceKV(cfg)
		if err != nil {
			t.Fatal(err)
		}
		first, err := GPUPrefillPassKV(engine, cfg, cache, prompt)
		if err == nil {
			want[i] = append(want[i], first)
			for position := len(prompt); position < len(prompt)+2; position++ {
				next, decodeErr := GPUForwardPassKV(engine, cfg, cache, want[i][len(want[i])-1], position)
				if decodeErr != nil {
					err = decodeErr
					break
				}
				want[i] = append(want[i], next)
			}
		}
		cache.Free()
		if err != nil {
			t.Fatal(err)
		}
	}

	server := NewTelemetryServer(engine, cfg)
	if server.continuous == nil {
		t.Fatal("CUDA continuous scheduler is unavailable")
	}
	backend, ok := server.continuousEngine.(*cudaServingBackend)
	if !ok {
		t.Fatalf("serving backend = %T, want *cudaServingBackend", server.continuousEngine)
	}
	defer backend.Close()
	defer server.continuous.Close()
	responses := make([]*continuousbatch.Response, len(prompts))
	for i, prompt := range prompts {
		responses[i], err = server.continuous.Submit(t.Context(), continuousbatch.Request{
			ID: fmt.Sprintf("cuda-real-%d", i), PromptTokenIDs: prompt, MaxTokens: 3,
			Sampler: SamplerConfig{Method: "argmax"},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for i, response := range responses {
		var got []int32
		for token := range response.Tokens {
			got = append(got, token.ID)
		}
		if !slices.Equal(got, want[i]) {
			t.Errorf("request %d tokens = %v, want %v", i, got, want[i])
		}
		if err := <-response.Done; err != nil {
			t.Errorf("request %d: %v", i, err)
		}
	}
}

type prefillObservation struct {
	requestIDs []string
	errors     []error
}

func (b *fakeServingBackend) Prefill(ctx context.Context, state *continuousbatch.State) (continuousbatch.Token, error) {
	result := b.PrefillBatch([]continuousbatch.PrefillRequest{{Context: ctx, State: state}})
	return result[0].Token, result[0].Err
}

func (b *fakeServingBackend) PrefillBatch(requests []continuousbatch.PrefillRequest) []continuousbatch.Result {
	observation := prefillObservation{requestIDs: make([]string, len(requests)), errors: make([]error, len(requests))}
	for i, request := range requests {
		observation.requestIDs[i] = request.State.ID
		observation.errors[i] = request.Context.Err()
	}
	b.prefill <- observation
	results := make([]continuousbatch.Result, len(requests))
	for i, request := range requests {
		if err := request.Context.Err(); err != nil {
			results[i].Err = err
			continue
		}
		results[i].Token.ID = 1
	}
	return results
}

func (*fakeServingBackend) DecodeBatch(_ context.Context, states []*continuousbatch.State) []continuousbatch.Result {
	return make([]continuousbatch.Result, len(states))
}

func (*fakeServingBackend) Release(*continuousbatch.State) {}

func (b *fakeServingBackend) Close() {
	b.mu.Lock()
	b.closed = true
	b.mu.Unlock()
}

func TestLinuxServingBackendRejectsIncompleteEngine(t *testing.T) {
	backend, err := newPlatformServingBackend(nil, Config{}, 1)
	if !errors.Is(err, errServingBackendUnavailable) || backend != nil {
		t.Fatalf("backend = %T, %v; want unavailable", backend, err)
	}
}

func TestValidateCUDASampler(t *testing.T) {
	tests := []struct {
		name    string
		sampler any
		wantErr bool
	}{
		{name: "default"},
		{name: "argmax", sampler: SamplerConfig{Method: "argmax"}},
		{name: "temperature", sampler: SamplerConfig{Method: "argmax", Temperature: 1}, wantErr: true},
		{name: "unsupported method", sampler: SamplerConfig{Method: "top-p"}, wantErr: true},
		{name: "unsupported type", sampler: struct{}{}, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := validateCUDASampler(test.sampler); (err != nil) != test.wantErr {
				t.Fatalf("validateCUDASampler(%#v) = %v, wantErr %v", test.sampler, err, test.wantErr)
			}
		})
	}
}

func TestCUDAServingConfigUsesCompiledModel(t *testing.T) {
	engine := &ModelEngine{mCfg: ModelConfig{
		HiddenSize: 16, IntermediateSize: 32, NumHiddenLayers: 3,
		NumAttentionHeads: 4, NumKeyValueHeads: 2, ExplicitHeadDim: 4,
		VocabSize: 64, MaxSeqLen: 128, RmsNormEps: 1e-6, RopeTheta: 500000,
		EOSTokenID: 9, EOSTokenIDs: []int32{7, 9},
	}}
	cfg := cudaServingConfig(engine, Config{BOSTokenID: 5, HTTPAddr: ":8080", VocabSize: 1, MaxSeqLen: 2, EOSTokenID: 3})
	if cfg.Dim != 16 || cfg.HiddenDim != 32 || cfg.NumLayers != 3 || cfg.NumHeads != 4 || cfg.NumKVHeads != 2 || cfg.HeadDim != 4 {
		t.Fatalf("compiled dimensions were not retained: %+v", cfg)
	}
	if cfg.VocabSize != 64 || cfg.MaxSeqLen != 128 || cfg.EOSTokenID != 9 || !slices.Equal(cfg.EOSTokenIDs, []int32{7, 9}) {
		t.Fatalf("compiled token semantics were not retained: %+v", cfg)
	}
	if cfg.BOSTokenID != 5 || cfg.HTTPAddr != ":8080" {
		t.Fatalf("runtime options were not retained: %+v", cfg)
	}
	engine.mCfg.EOSTokenIDs[0] = 11
	if cfg.EOSTokenIDs[0] != 7 {
		t.Fatal("serving EOS token IDs alias engine configuration")
	}
}

func TestCUDAServingBackendRejectsNilContexts(t *testing.T) {
	backend := new(cudaServingBackend)
	if _, err := backend.Prefill(nil, new(continuousbatch.State)); err == nil {
		t.Fatal("Prefill(nil) succeeded")
	}
	results := backend.PrefillBatch([]continuousbatch.PrefillRequest{{State: new(continuousbatch.State)}})
	if len(results) != 1 || results[0].Err == nil {
		t.Fatalf("PrefillBatch(nil context) = %+v", results)
	}
	results = backend.DecodeBatch(nil, []*continuousbatch.State{{}})
	if len(results) != 1 || results[0].Err == nil {
		t.Fatalf("DecodeBatch(nil) = %+v", results)
	}
}

func TestServingBackendBoundaryPreservesRequestContexts(t *testing.T) {
	backend := &fakeServingBackend{prefill: make(chan prefillObservation, 1)}
	scheduler, err := newServingScheduler(backend, continuousbatch.Config{
		MaxActive: 2, MaxQueue: 2, MaxBatch: 2, BatchWait: time.Millisecond,
		TokenBuffer: 1, PrefillWorkers: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer scheduler.Close()

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := scheduler.Submit(canceled, continuousbatch.Request{ID: "canceled", MaxTokens: 1}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled submit = %v, want context canceled", err)
	}
	response, err := scheduler.Submit(t.Context(), continuousbatch.Request{ID: "live", MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	observation := <-backend.prefill
	if len(observation.requestIDs) != 1 || observation.requestIDs[0] != "live" || observation.errors[0] != nil {
		t.Fatalf("prefill observation = %+v", observation)
	}
	if token := <-response.Tokens; token.ID != 1 {
		t.Fatalf("token = %d, want 1", token.ID)
	}
	if err := <-response.Done; err != nil {
		t.Fatal(err)
	}

	done := make(chan struct{})
	go func() {
		backend.Close()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("backend close blocked on scheduler or device work")
	}
}
