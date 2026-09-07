//go:build linux && !tpu

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

func TestServingPrefixTelemetryUnavailable(t *testing.T) {
	backend := &scheduledEngineAdapter{prefixes: &productionPrefixRuntime{}}
	metrics, available := backend.prefixCacheMetrics()
	if available || metrics != (PrefixCacheMetrics{}) {
		t.Fatalf("prefix cache = (%+v, %t), want unavailable", metrics, available)
	}
	srv := &TelemetryServer{engine: &ModelEngine{}, continuousEngine: backend}
	telemetry := srv.memoryTelemetry().PrefixCache
	if telemetry.State != "unavailable" {
		t.Fatalf("prefix telemetry = %+v, want unavailable", telemetry)
	}
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

func TestRealCUDAProductionServingStress(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.MaxSeqLen = 16
	engine, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	baseline, _ := engine.memory.Snapshot()
	requests := []continuousbatch.Request{
		{ID: "cancel-decode", PromptTokenIDs: []int32{41}, MaxTokens: 8, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "short", PromptTokenIDs: []int32{42}, MaxTokens: 3, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "medium-a", PromptTokenIDs: []int32{43, 44}, MaxTokens: 5, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "medium-b", PromptTokenIDs: []int32{45, 46}, MaxTokens: 6, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "long", PromptTokenIDs: []int32{47, 48, 49, 50}, MaxTokens: 7, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "fail", PromptTokenIDs: []int32{51, 52, 53}, MaxTokens: 4, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "late-a", PromptTokenIDs: []int32{54}, MaxTokens: 3, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "late-b", PromptTokenIDs: []int32{55, 56}, MaxTokens: 4, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "late-c", PromptTokenIDs: []int32{57, 58, 59}, MaxTokens: 5, Sampler: SamplerConfig{Method: "argmax"}},
		{ID: "late-d", PromptTokenIDs: []int32{60, 61, 62, 63}, MaxTokens: 6, Sampler: SamplerConfig{Method: "argmax"}},
	}
	wants := make(map[string][]int32, len(requests))
	for _, request := range requests {
		if request.ID != "cancel-decode" && request.ID != "fail" {
			wants[request.ID] = referenceCUDARequest(t, engine, cfg, request.PromptTokenIDs, request.MaxTokens)
		}
	}

	backendValue, err := newPlatformServingBackend(engine, cfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	raw := backendValue.(*cudaServingBackend)
	raw.enableBenchmarkLedger()
	ledgerStart := raw.benchmarkLedgerSnapshot()
	backend := &servingStressBackend{
		cudaServingBackend: raw,
		failID:             "fail",
		decodeEntered:      make(chan []string, 1),
		decodeGate:         make(chan struct{}),
	}
	scheduler, err := newServingScheduler(backend, continuousbatch.Config{
		MaxActive: 8, MaxQueue: 2, MaxBatch: 8, BatchWait: 10 * time.Millisecond,
		TokenBuffer: 16, PrefillWorkers: 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := scheduler.Submit(canceled, continuousbatch.Request{ID: "before-admission", MaxTokens: 2}); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-admission cancellation = %v", err)
	}

	responses := make(map[string]*continuousbatch.Response, len(requests))
	for i, request := range requests {
		response, err := scheduler.Submit(t.Context(), request)
		if err != nil {
			t.Fatalf("submit %s: %v", request.ID, err)
		}
		responses[request.ID] = response
		if i == 0 {
			time.Sleep(time.Millisecond)
		}
	}
	if _, err := scheduler.Submit(t.Context(), continuousbatch.Request{ID: "overflow", PromptTokenIDs: []int32{64}, MaxTokens: 2}); !errors.Is(err, continuousbatch.ErrQueueFull) {
		t.Fatalf("queue pressure = %v, want queue full", err)
	}

	var firstDecode []string
	select {
	case firstDecode = <-backend.decodeEntered:
	case <-time.After(30 * time.Second):
		t.Fatal("CUDA decode did not start")
	}
	if !slices.Contains(firstDecode, "cancel-decode") {
		t.Fatalf("first decode %v omits cancellation target", firstDecode)
	}
	responses["cancel-decode"].Cancel()
	close(backend.decodeGate)

	for _, request := range requests {
		response := responses[request.ID]
		var got []int32
		for token := range response.Tokens {
			got = append(got, token.ID)
		}
		err := <-response.Done
		switch request.ID {
		case "cancel-decode":
			if !errors.Is(err, context.Canceled) || len(got) != 1 {
				t.Errorf("decode cancellation tokens=%v err=%v, want one prefill token and cancellation", got, err)
			}
		case "fail":
			if err == nil || !strings.Contains(err.Error(), "injected request failure") || len(got) != 0 {
				t.Errorf("injected failure tokens=%v err=%v", got, err)
			}
		default:
			if err != nil {
				t.Errorf("request %s: %v", request.ID, err)
			}
			want := wants[request.ID]
			if !slices.Equal(got, want) {
				t.Errorf("request %s tokens=%v want=%v", request.ID, got, want)
			}
		}
	}
	cohort := make([]*continuousbatch.Response, 8)
	for i := range cohort {
		cohort[i], err = scheduler.Submit(t.Context(), continuousbatch.Request{
			ID: fmt.Sprintf("width-8-%d", i), PromptTokenIDs: []int32{42}, MaxTokens: 3,
			Sampler: SamplerConfig{Method: "argmax"},
		})
		if err != nil {
			t.Fatalf("submit width-8 cohort: %v", err)
		}
	}
	for i, response := range cohort {
		var got []int32
		for token := range response.Tokens {
			got = append(got, token.ID)
		}
		if err := <-response.Done; err != nil || !slices.Equal(got, wants["short"]) {
			t.Errorf("width-8 request %d tokens=%v err=%v want=%v", i, got, err, wants["short"])
		}
	}
	metrics := scheduler.Metrics()
	if metrics.Rejected != 1 || metrics.Canceled != 1 || metrics.Failed != 1 || metrics.Completed != 16 {
		t.Fatalf("terminal metrics = %+v", metrics)
	}
	waves, err := raw.benchmarkLedgerSince(ledgerStart)
	if err != nil {
		t.Fatal(err)
	}
	shared := false
	decodeWidths := make(map[int]int)
	for _, wave := range waves {
		if wave.Kind == "cuda-decode" {
			decodeWidths[len(wave.Requests)]++
			if len(wave.Requests) > 1 {
				shared = true
			}
		}
	}
	if !shared {
		t.Fatalf("physical waves contain no shared-weight decode: %+v", waves)
	}
	for _, width := range []int{1, 4, 8} {
		if decodeWidths[width] == 0 {
			t.Fatalf("decode widths = %v, want physical B1/B4/B8", decodeWidths)
		}
	}
	if err := scheduler.Close(); err != nil {
		t.Fatal(err)
	}
	backend.Close()
	current, _ := engine.memory.Snapshot()
	if current != baseline {
		t.Fatalf("allocation current=%d want baseline=%d", current, baseline)
	}
	t.Logf("receipt arrivals=staggered decode_widths=%v canceled_before_admission=true canceled_during_decode=true queue_rejected=1 failure_isolated=true survivors=16 shared_decode=true allocation_baseline=%d waves=%d", decodeWidths, baseline, len(waves))
}

func TestRealCUDAProductionServingShutdown(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.MaxSeqLen = 16
	engine, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	baseline, _ := engine.memory.Snapshot()
	backendValue, err := newPlatformServingBackend(engine, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	backend := &servingStressBackend{
		cudaServingBackend: backendValue.(*cudaServingBackend),
		decodeEntered:      make(chan []string, 1),
		decodeGate:         make(chan struct{}),
	}
	scheduler, err := newServingScheduler(backend, continuousbatch.Config{
		MaxActive: 1, MaxQueue: 1, MaxBatch: 1, BatchWait: time.Millisecond,
		TokenBuffer: 4, PrefillWorkers: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := scheduler.Submit(t.Context(), continuousbatch.Request{
		ID: "shutdown", PromptTokenIDs: []int32{41, 42}, MaxTokens: 8,
		Sampler: SamplerConfig{Method: "argmax"},
	})
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-backend.decodeEntered:
	case <-time.After(30 * time.Second):
		t.Fatal("CUDA decode did not start")
	}
	closed := make(chan error, 1)
	go func() { closed <- scheduler.Close() }()
	close(backend.decodeGate)
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
	var tokens []int32
	for token := range response.Tokens {
		tokens = append(tokens, token.ID)
	}
	if err := <-response.Done; !errors.Is(err, continuousbatch.ErrClosed) || len(tokens) != 1 {
		t.Fatalf("shutdown tokens=%v err=%v, want one committed prefill token and closed", tokens, err)
	}
	metrics := scheduler.Metrics()
	if metrics.Active != 0 || metrics.Queued != 0 || metrics.Failed != 1 {
		t.Fatalf("shutdown metrics = %+v", metrics)
	}
	backend.Close()
	current, _ := engine.memory.Snapshot()
	if current != baseline {
		t.Fatalf("allocation current=%d want baseline=%d", current, baseline)
	}
	t.Logf("receipt shutdown_in_flight=true post_shutdown_tokens=0 allocation_baseline=%d", baseline)
}

func referenceCUDARequest(t *testing.T, engine *ModelEngine, cfg Config, prompt []int32, limit int) []int32 {
	t.Helper()
	cache, err := engine.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Free()
	next, err := GPUPrefillPassKV(engine, cfg, cache, prompt)
	if err != nil {
		t.Fatal(err)
	}
	got := []int32{next}
	for position := len(prompt); len(got) < limit && !cfg.IsEOS(next); position++ {
		next, err = GPUForwardPassKV(engine, cfg, cache, next, position)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, next)
	}
	return got
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
		for _, width := range []int{1, 2, 4, 8} {
			t.Run(fmt.Sprintf("fused=%t/B%d", fused, width), func(t *testing.T) {
				cfg := Llama3_2_1BConfig(path)
				cfg.MaxSeqLen = 8
				cfg.FuseEGraph = fused
				engine, err := GetOrCompileEngine(cfg)
				if err != nil {
					t.Fatal(err)
				}
				cfg = cudaServingConfig(engine, cfg)
				backendValue, err := newPlatformServingBackend(engine, cfg, width)
				if err != nil {
					t.Fatal(err)
				}
				backend := backendValue.(*cudaServingBackend)
				defer backend.Close()
				backend.observedOutputs = newObservedOutputCollector()
				states := make([]*continuousbatch.State, width)
				refCaches := make([]*SequenceKVCache, width)
				refLogits := make([][]float32, width)
				refTokens := make([]int32, width)
				allPrompts := [][]int32{{41, 42}, {41, 42}, {43}, {44, 45, 46}}
				positions := make([]int, width)
				for row := range width {
					prompt := allPrompts[row%len(allPrompts)]
					positions[row] = len(prompt)
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
						checkCUDAKVCoordinate(t, sequence.cache, refCaches[row], layer, positions[row], engine.mCfg.NumKeyValueHeads*engine.mCfg.HeadDim(), 2e-3)
					}
				}
				for _, state := range states {
					state.Position++
					backend.Release(state)
				}
				ids := make([]string, width)
				for row := range states {
					ids[row] = states[row].ID
				}
				observed, err := backend.observedOutputs.outputs(ids)
				if err != nil {
					t.Fatal(err)
				}
				for row, output := range observed {
					wantTokens := positions[row] + 1
					wantElements := 2 * cfg.NumLayers * wantTokens * cfg.NumKVHeads * cfg.HeadDim
					if output.Request != row || output.FullLogitsSHA256 != hashFloat32(gotLogits[row*cfg.VocabSize:(row+1)*cfg.VocabSize]) || output.FullLogitsElements != cfg.VocabSize || output.PopulatedTokens != wantTokens || output.PopulatedKVElements != wantElements || len(output.Layers) != cfg.NumLayers {
						t.Fatalf("row %d observed output = %+v", row, output)
					}
				}
				launches := backend.lastBatchLaunches()
				wantMatrixLaunches := 2*engine.mCfg.NumHiddenLayers + 1
				if launches.Rows != width || launches.WeightLaunches <= 0 || launches.IndependentWeightLaunches != launches.WeightLaunches*width || width > 1 && launches.WeightLaunches >= launches.IndependentWeightLaunches || launches.MatrixLaunches != wantMatrixLaunches {
					t.Fatalf("launch receipt = %+v", launches)
				}
				t.Logf("receipt batch=%d weight_launches=%d independent_weight_launches=%d matrix_launches=%d observed_requests=%d observed_sha256=%s", width, launches.WeightLaunches, launches.IndependentWeightLaunches, launches.MatrixLaunches, len(observed), observed[0].FullLogitsSHA256)
			})
		}
	}
}

func TestRealCUDABenchmarkControlLedger(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	for _, test := range []struct {
		arm       string
		wantWaves int
		wantWidth int
	}{
		{"llama-infer", 1, 4},
		{"llama-infer-serial", 4, 1},
	} {
		t.Run(test.arm, func(t *testing.T) {
			cfg := Llama3_2_1BConfig(path)
			cfg.MaxSeqLen = 8
			engine, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			value, err := newPlatformServingBackend(engine, cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			backend := value.(*cudaServingBackend)
			defer backend.Close()
			backend.enableBenchmarkLedger()
			if err := configureBenchmarkArm(engine, test.arm); err != nil {
				t.Fatal(err)
			}
			states := make([]*continuousbatch.State, 4)
			for i := range states {
				state := &continuousbatch.State{ID: fmt.Sprintf("sample-0-%d", i), PromptTokenIDs: []int32{41, 42}, Sampler: SamplerConfig{Method: "argmax"}}
				result := backend.PrefillBatch([]continuousbatch.PrefillRequest{{Context: t.Context(), State: state}})[0]
				if result.Err != nil {
					t.Fatal(result.Err)
				}
				state.Current = result.Token.ID
				states[i] = state
				defer backend.Release(state)
			}
			before := backend.benchmarkLedgerSnapshot()
			results := backend.DecodeBatch(t.Context(), states)
			for i, result := range results {
				if result.Err != nil {
					t.Fatalf("row %d: %v", i, result.Err)
				}
			}
			waves, err := backend.benchmarkLedgerSince(before)
			if err != nil {
				t.Fatal(err)
			}
			if len(waves) != test.wantWaves {
				t.Fatalf("waves = %d, want %d: %+v", len(waves), test.wantWaves, waves)
			}
			for _, wave := range waves {
				if len(wave.Requests) != test.wantWidth || wave.Dispatches == 0 {
					t.Fatalf("wave = %+v, want width %d and measured dispatches", wave, test.wantWidth)
				}
			}
		})
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

type servingStressBackend struct {
	*cudaServingBackend
	failID        string
	decodeEntered chan []string
	decodeGate    chan struct{}
	decodeOnce    sync.Once
}

func (b *servingStressBackend) PrefillBatch(requests []continuousbatch.PrefillRequest) []continuousbatch.Result {
	results := b.cudaServingBackend.PrefillBatch(requests)
	for i, request := range requests {
		if request.State.ID == b.failID && results[i].Err == nil {
			results[i].Err = errors.New("injected request failure")
		}
	}
	return results
}

func (b *servingStressBackend) DecodeBatch(ctx context.Context, states []*continuousbatch.State) []continuousbatch.Result {
	b.decodeOnce.Do(func() {
		ids := make([]string, len(states))
		for i, state := range states {
			ids[i] = state.ID
		}
		b.decodeEntered <- ids
		<-b.decodeGate
	})
	end := benchmarkPhaseRange("llama-infer/serving-stress/decode")
	defer end()
	return b.cudaServingBackend.DecodeBatch(ctx, states)
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
	backend := &cudaServingBackend{live: map[*SequenceKVCache]uint64{cache: 2}, observedOutputs: newObservedOutputCollector()}
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
	if len(backend.observedOutputs.records) != 0 {
		t.Fatal("stale release produced observed output")
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

func TestCUDAServingInitializationTelemetry(t *testing.T) {
	s := NewTelemetryServer(&ModelEngine{}, Config{})
	got := s.runtimeSchedulingTelemetry()
	if got.ServingMode != "serial" || !strings.Contains(got.ServingError, "initialize continuous serving backend") {
		t.Fatalf("failed backend hidden: %+v", got)
	}
	if s.continuousEngine != nil || s.continuous != nil {
		t.Fatal("failed backend published as initialized")
	}
	backend := &fakeServingBackend{}
	scheduler, err := newServingScheduler(backend, continuousbatch.Config{})
	if err != nil {
		t.Fatal(err)
	}
	defer scheduler.Close()
	s = &TelemetryServer{engine: &ModelEngine{}, continuous: scheduler, continuousEngine: backend}
	got = s.runtimeSchedulingTelemetry()
	if got.ServingMode != "continuous-batch" || got.ServingError != "" {
		t.Fatalf("live scheduler not reported: %+v", got)
	}
	s = NewTelemetryServer(nil, Config{})
	if got := s.runtimeSchedulingTelemetry(); got.ServingMode != "unavailable" {
		t.Fatalf("missing engine reported as live: %+v", got)
	}
}
