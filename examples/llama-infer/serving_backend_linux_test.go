//go:build linux

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
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

func TestBatchElements(t *testing.T) {
	for _, test := range []struct {
		name    string
		factors []int
		want    int
		wantErr bool
	}{
		{name: "product", factors: []int{4, 8, 16}, want: 512},
		{name: "zero", factors: []int{4, 0}, wantErr: true},
		{name: "negative", factors: []int{-1, 4}, wantErr: true},
		{name: "overflow", factors: []int{math.MaxInt, 2}, wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := batchElements(test.name, test.factors...)
			if (err != nil) != test.wantErr {
				t.Fatalf("batchElements() error = %v, want error %v", err, test.wantErr)
			}
			if got != test.want {
				t.Fatalf("batchElements() = %d, want %d", got, test.want)
			}
		})
	}
}

func TestRealCUDAProductionDecodeUsesSharedWeightBatch(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	for _, fused := range []bool{false, true} {
		for _, width := range []int{2, 4, 8} {
			t.Run(fmt.Sprintf("fused=%t/B%d", fused, width), func(t *testing.T) {
				cfg := Llama3_2_1BConfig(path)
				cfg.MaxSeqLen = 8
				cfg.FuseEGraph = fused
				engine, err := GetOrCompileEngine(cfg)
				if err != nil {
					t.Fatal(err)
				}
				backendValue, err := newPlatformServingBackend(engine, cfg, width)
				if err != nil {
					t.Fatal(err)
				}
				backend := backendValue.(*cudaServingBackend)
				defer backend.Close()
				states := make([]*continuousbatch.State, width)
				refCaches := make([]*SequenceKVCache, width)
				refLogits := make([][]float32, width)
				refTokens := make([]int32, width)
				for row := range width {
					prompt := []int32{int32(41 + row*2), int32(42 + row*2)}
					refCaches[row], err = engine.NewSequenceKV(cfg)
					if err != nil {
						t.Fatal(err)
					}
					defer refCaches[row].Free()
					current, err := GPUPrefillPassKV(engine, cfg, refCaches[row], prompt)
					if err != nil {
						t.Fatal(err)
					}
					refTokens[row], err = GPUForwardPassKV(engine, cfg, refCaches[row], current, len(prompt))
					if err != nil {
						t.Fatal(err)
					}
					refLogits[row], err = engine.ReadLogits()
					if err != nil {
						t.Fatal(err)
					}
					state := &continuousbatch.State{ID: fmt.Sprintf("row-%d", row), Current: current, Position: len(prompt), Sampler: SamplerConfig{Method: "argmax"}}
					state.KV = prefillStateKV(t, backend, prompt, state.Sampler)
					state.Current = current
					states[row] = state
				}
				defer func() {
					for _, state := range states {
						backend.Release(state)
					}
				}()
				results := backend.DecodeBatch(t.Context(), states)
				gotLogits := make([]float32, width*cfg.VocabSize)
				if err := backend.batch.Logits.Slice(0, len(gotLogits)).CopyToHost(gotLogits); err != nil {
					t.Fatal(err)
				}
				for row, result := range results {
					if result.Err != nil {
						t.Fatalf("row %d: %v", row, result.Err)
					}
					if result.Token.ID != refTokens[row] {
						t.Fatalf("row %d token = %d, want %d", row, result.Token.ID, refTokens[row])
					}
					checkCUDAFloatSlice(t, fmt.Sprintf("row %d logits", row), gotLogits[row*cfg.VocabSize:(row+1)*cfg.VocabSize], refLogits[row], 2e-3)
					sequence := states[row].KV.(cudaSequenceState)
					for layer := 0; layer < engine.mCfg.NumHiddenLayers; layer++ {
						checkCUDAKVCoordinate(t, sequence.cache, refCaches[row], layer, 2, engine.mCfg.NumKeyValueHeads*engine.mCfg.HeadDim(), 2e-3)
					}
				}
				launches := backend.lastBatchLaunches()
				wantMatrixLaunches := 2*engine.mCfg.NumHiddenLayers + 1
				if launches.Rows != width || launches.WeightLaunches <= 0 || launches.WeightLaunches >= launches.IndependentWeightLaunches || launches.MatrixLaunches != wantMatrixLaunches {
					t.Fatalf("launch receipt = %+v", launches)
				}
				t.Logf("receipt batch=%d weight_launches=%d independent_weight_launches=%d matrix_launches=%d", width, launches.WeightLaunches, launches.IndependentWeightLaunches, launches.MatrixLaunches)
			})
		}
	}
}

func prefillStateKV(t *testing.T, backend *cudaServingBackend, prompt []int32, sampler any) any {
	t.Helper()
	state := &continuousbatch.State{ID: "batch", PromptTokenIDs: prompt, Sampler: sampler}
	result := backend.PrefillBatch([]continuousbatch.PrefillRequest{{Context: t.Context(), State: state}})[0]
	if result.Err != nil {
		t.Fatal(result.Err)
	}
	return state.KV
}

func checkCUDAFloatSlice(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tolerance {
			t.Fatalf("%s[%d] = %g, want %g (diff %g)", name, i, got[i], want[i], diff)
		}
	}
}

func checkCUDAKVCoordinate(t *testing.T, got, want *SequenceKVCache, layer, position, kvDim int, tolerance float64) {
	t.Helper()
	gotLayer, err := got.Cache.Layer(layer)
	if err != nil {
		t.Fatal(err)
	}
	wantLayer, err := want.Cache.Layer(layer)
	if err != nil {
		t.Fatal(err)
	}
	for _, pair := range []struct {
		name      string
		got, want []float32
	}{
		{name: "K", got: make([]float32, kvDim), want: make([]float32, kvDim)},
		{name: "V", got: make([]float32, kvDim), want: make([]float32, kvDim)},
	} {
		var gotDevice, wantDevice cudamem.Slice[float32]
		if pair.name == "K" {
			gotDevice, wantDevice = gotLayer.K, wantLayer.K
		} else {
			gotDevice, wantDevice = gotLayer.V, wantLayer.V
		}
		if err := gotDevice.Slice(position*kvDim, (position+1)*kvDim).CopyToHost(pair.got); err != nil {
			t.Fatal(err)
		}
		if err := wantDevice.Slice(position*kvDim, (position+1)*kvDim).CopyToHost(pair.want); err != nil {
			t.Fatal(err)
		}
		checkCUDAFloatSlice(t, pair.name, pair.got, pair.want, tolerance)
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

func TestCUDAServingBackendRejectsStaleGeneration(t *testing.T) {
	cache := &SequenceKVCache{Cache: new(cudakv.Cache)}
	backend := &cudaServingBackend{live: map[*SequenceKVCache]uint64{cache: 2}}
	state := &continuousbatch.State{KV: cudaSequenceState{cache: cache, generation: 1}}
	results := backend.DecodeBatch(t.Context(), []*continuousbatch.State{state})
	if len(results) != 1 || results[0].Err == nil || !strings.Contains(results[0].Err.Error(), "stale cache state") {
		t.Fatalf("DecodeBatch(stale) = %+v", results)
	}
	backend.Release(state)
	if state.KV != nil {
		t.Fatal("Release left stale state attached")
	}
	if backend.live[cache] != 2 {
		t.Fatal("Release retired the live generation through a stale state")
	}
}

func TestCUDAServingBackendReleaseWaitsForDeviceOwner(t *testing.T) {
	engine := new(ModelEngine)
	backend := &cudaServingBackend{engine: engine, live: make(map[*SequenceKVCache]uint64)}
	state := &continuousbatch.State{KV: cudaSequenceState{cache: new(SequenceKVCache), generation: 1}}

	engine.generateMu.Lock()
	done := make(chan struct{})
	go func() {
		backend.Release(state)
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("Release completed while device owner was active")
	case <-time.After(20 * time.Millisecond):
	}
	engine.generateMu.Unlock()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Release did not complete after device owner released")
	}
}

func TestCUDAServingBackendCloseIsSerializedAndIdempotent(t *testing.T) {
	engine := new(ModelEngine)
	backend := &cudaServingBackend{engine: engine, live: make(map[*SequenceKVCache]uint64)}
	engine.generateMu.Lock()
	done := make(chan struct{})
	go func() {
		backend.Close()
		backend.Close()
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("Close completed while device owner was active")
	case <-time.After(20 * time.Millisecond):
	}
	engine.generateMu.Unlock()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Close did not complete after device owner released")
	}
	results := backend.DecodeBatch(t.Context(), []*continuousbatch.State{{}})
	if len(results) != 1 || results[0].Err == nil || !strings.Contains(results[0].Err.Error(), "backend closed") {
		t.Fatalf("DecodeBatch after Close = %+v", results)
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
