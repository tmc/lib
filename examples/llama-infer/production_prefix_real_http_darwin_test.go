//go:build darwin

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

type productionHTTPReceipt struct {
	logits     []float32
	k, v       [][]float32
	tokenCount int
}

type productionHTTPRecorder struct {
	mu       sync.Mutex
	receipts map[string]productionHTTPReceipt
	err      error
}

func newProductionHTTPRecorder(adapter *scheduledEngineAdapter, cfg Config) *productionHTTPRecorder {
	r := &productionHTTPRecorder{receipts: make(map[string]productionHTTPReceipt)}
	adapter.prefillObserver = func(state *continuousbatch.State, lease *productionPrefixLease, logits []float32) {
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.err != nil {
			return
		}
		n := lease.Tokens()
		kd := cfg.NumKVHeads * cfg.HeadDim
		k := make([][]float32, cfg.NumLayers)
		v := make([][]float32, cfg.NumLayers)
		for layer := range cfg.NumLayers {
			k[layer] = make([]float32, n*kd)
			v[layer] = make([]float32, n*kd)
		}
		if err := lease.lease.Restore(k, v, n); err != nil {
			r.err = err
			return
		}
		r.receipts[state.Prompt] = productionHTTPReceipt{logits: append([]float32(nil), logits...), k: k, v: v, tokenCount: state.Position}
	}
	return r
}

func (r *productionHTTPRecorder) snapshot() (map[string]productionHTTPReceipt, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	got := make(map[string]productionHTTPReceipt, len(r.receipts))
	for prompt, receipt := range r.receipts {
		got[prompt] = receipt
	}
	return got, r.err
}

func TestProductionPrefixRealHTTPColdParity(t *testing.T) {
	model := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if model == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(model)
	cfg.Name += "-production-prefix-http"
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 128
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !eng.isF16 || len(eng.layersF16) != cfg.NumLayers {
		t.Fatalf("model f16=%v layers=%d, want %d", eng.isF16, len(eng.layersF16), cfg.NumLayers)
	}
	const common = "The capital of France is"
	for _, width := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("B%d", width), func(t *testing.T) {
			prompts := make([]string, width)
			for i := range width {
				prompts[i] = fmt.Sprintf("%s option %d", common, i)
				commonTokens, err := prefillTokens(cfg, eng.tokenizer, &continuousbatch.State{Prompt: common})
				if err != nil {
					t.Fatal(err)
				}
				promptTokens, err := prefillTokens(cfg, eng.tokenizer, &continuousbatch.State{Prompt: prompts[i]})
				if err != nil {
					t.Fatal(err)
				}
				if len(promptTokens) <= len(commonTokens) || !reflect.DeepEqual(promptTokens[:len(commonTokens)], commonTokens) {
					t.Fatalf("prompt %q does not extend common token prefix", prompts[i])
				}
			}

			cold, coldRecorder := newProductionHTTPTestServer(t, eng, cfg, width)
			coldTokens := runProductionHTTPBatch(t, cold, prompts)
			coldBackend := scheduledBackend(t, cold.continuousEngine)
			coldStats := coldBackend.prefixes.Stats()
			if coldStats.Hits != 0 || coldStats.Misses != int64(width) || coldStats.SavedTokens != 0 {
				t.Fatalf("cold prefix stats=%+v, want %d misses and no reuse", coldStats, width)
			}
			coldReceipts, err := coldRecorder.snapshot()
			if err != nil {
				t.Fatal(err)
			}
			cold.continuous.Close()
			scheduledBackend(t, cold.continuousEngine).Close()

			warm, warmRecorder := newProductionHTTPTestServer(t, eng, cfg, width)
			primeHTTPPrompt(t, warm, common)
			warmBackend := scheduledBackend(t, warm.continuousEngine)
			beforeWarm := warmBackend.prefixes.Stats()
			warmRecorder.mu.Lock()
			warmRecorder.receipts = make(map[string]productionHTTPReceipt)
			warmRecorder.mu.Unlock()
			warmTokens := runProductionHTTPBatch(t, warm, prompts)
			servingStats := warmBackend.prefixes.Stats()
			commonTokens, err := prefillTokens(cfg, eng.tokenizer, &continuousbatch.State{Prompt: common})
			if err != nil {
				t.Fatal(err)
			}
			if got := servingStats.Hits - beforeWarm.Hits; got != int64(width) {
				t.Fatalf("warm prefix hits=%d, want %d; before=%+v after=%+v", got, width, beforeWarm, servingStats)
			}
			if got, want := servingStats.SavedTokens-beforeWarm.SavedTokens, int64(width*len(commonTokens)); got < want {
				t.Fatalf("warm saved tokens=%d, want at least %d; before=%+v after=%+v", got, want, beforeWarm, servingStats)
			}
			telemetry := warm.memoryTelemetry().PrefixCache
			if telemetry.State != "observed" || telemetry.Entries != servingStats.Entries || telemetry.Hits != servingStats.Hits || telemetry.Misses != servingStats.Misses || telemetry.SavedTokens != servingStats.SavedTokens {
				t.Fatalf("prefix telemetry=%+v, want active serving stats=%+v", telemetry, servingStats)
			}
			warmReceipts, err := warmRecorder.snapshot()
			if err != nil {
				t.Fatal(err)
			}
			observations := scheduledBackend(t, warm.continuousEngine).PrefillObservations()
			warm.continuous.Close()
			scheduledBackend(t, warm.continuousEngine).Close()
			if !reflect.DeepEqual(warmTokens, coldTokens) {
				t.Fatalf("warm tokens=%v cold=%v", warmTokens, coldTokens)
			}
			foundWidth := width == 1
			for _, observation := range observations {
				if observation.Batch == width {
					foundWidth = true
				}
			}
			if !foundWidth {
				t.Fatalf("no B=%d production prefill wave: %+v", width, observations)
			}

			var maxLogit, maxKV float32
			coordinates := 0
			for _, prompt := range prompts {
				coldReceipt, coldOK := coldReceipts[prompt]
				warmReceipt, warmOK := warmReceipts[prompt]
				if !coldOK || !warmOK {
					t.Fatalf("prompt %q receipt cold=%v warm=%v", prompt, coldOK, warmOK)
				}
				logitDiff, kvDiff, n, err := compareProductionHTTPReceipt(cfg, coldReceipt, warmReceipt)
				if err != nil {
					t.Fatalf("prompt %q: %v", prompt, err)
				}
				maxLogit = max(maxLogit, logitDiff)
				maxKV = max(maxKV, kvDiff)
				coordinates += n
			}
			if maxLogit > 5e-4 || maxKV > 5e-4 {
				t.Fatalf("max logit=%g KV=%g", maxLogit, maxKV)
			}
			t.Logf("width=%d model=%s logits=%d kv_coordinates=%d max_logit=%g max_kv=%g cold_prefix=%+v warm_before=%+v warm_after=%+v observations=%+v", width, model, width*cfg.VocabSize, coordinates, maxLogit, maxKV, coldStats, beforeWarm, servingStats, observations)
		})
	}
}

func compareProductionHTTPReceipt(cfg Config, cold, warm productionHTTPReceipt) (maxLogit, maxKV float32, coordinates int, err error) {
	if len(cold.logits) != cfg.VocabSize || len(warm.logits) != cfg.VocabSize {
		return 0, 0, 0, fmt.Errorf("logit lengths=%d/%d, want %d", len(cold.logits), len(warm.logits), cfg.VocabSize)
	}
	maxLogit, err = finiteMaxAbsDiff32(cold.logits, warm.logits)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("logits: %w", err)
	}
	if cold.tokenCount <= 0 || cold.tokenCount != warm.tokenCount {
		return 0, 0, 0, fmt.Errorf("token counts=%d/%d", cold.tokenCount, warm.tokenCount)
	}
	if len(cold.k) != cfg.NumLayers || len(cold.v) != cfg.NumLayers || len(warm.k) != cfg.NumLayers || len(warm.v) != cfg.NumLayers {
		return 0, 0, 0, fmt.Errorf("KV layer counts cold=%d/%d warm=%d/%d, want %d", len(cold.k), len(cold.v), len(warm.k), len(warm.v), cfg.NumLayers)
	}
	want := cold.tokenCount * cfg.NumKVHeads * cfg.HeadDim
	if want <= 0 {
		return 0, 0, 0, fmt.Errorf("invalid KV geometry: tokens=%d heads=%d dim=%d", cold.tokenCount, cfg.NumKVHeads, cfg.HeadDim)
	}
	for layer := range cfg.NumLayers {
		vectors := []struct {
			name       string
			cold, warm []float32
		}{
			{"K", cold.k[layer], warm.k[layer]},
			{"V", cold.v[layer], warm.v[layer]},
		}
		for _, vector := range vectors {
			if len(vector.cold) != want || len(vector.warm) != want {
				return 0, 0, 0, fmt.Errorf("layer %d %s lengths=%d/%d, want %d", layer, vector.name, len(vector.cold), len(vector.warm), want)
			}
			diff, diffErr := finiteMaxAbsDiff32(vector.cold, vector.warm)
			if diffErr != nil {
				return 0, 0, 0, fmt.Errorf("layer %d %s: %w", layer, vector.name, diffErr)
			}
			maxKV = max(maxKV, diff)
			coordinates += len(vector.cold)
		}
	}
	return maxLogit, maxKV, coordinates, nil
}

func finiteMaxAbsDiff32(a, b []float32) (float32, error) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("lengths=%d/%d", len(a), len(b))
	}
	var maxDiff float32
	for i := range a {
		if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.IsNaN(float64(b[i])) || math.IsInf(float64(b[i]), 0) {
			return 0, fmt.Errorf("non-finite value at %d: %g/%g", i, a[i], b[i])
		}
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if math.IsNaN(float64(diff)) || math.IsInf(float64(diff), 0) {
			return 0, fmt.Errorf("non-finite difference at %d: %g", i, diff)
		}
		maxDiff = max(maxDiff, diff)
	}
	return maxDiff, nil
}

func TestCompareProductionHTTPReceiptRejectsInvalidData(t *testing.T) {
	cfg := Config{VocabSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}
	valid := productionHTTPReceipt{logits: []float32{1, 2}, k: [][]float32{{3, 4}}, v: [][]float32{{5, 6}}, tokenCount: 1}
	clone := func() productionHTTPReceipt {
		return productionHTTPReceipt{
			logits:     append([]float32(nil), valid.logits...),
			k:          [][]float32{append([]float32(nil), valid.k[0]...)},
			v:          [][]float32{append([]float32(nil), valid.v[0]...)},
			tokenCount: valid.tokenCount,
		}
	}
	if _, _, _, err := compareProductionHTTPReceipt(cfg, valid, clone()); err != nil {
		t.Fatalf("valid receipt: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*productionHTTPReceipt)
	}{
		{"cold NaN", func(r *productionHTTPReceipt) { r.logits[0] = float32(math.NaN()) }},
		{"warm infinity", func(r *productionHTTPReceipt) { r.v[0][0] = float32(math.Inf(1)) }},
		{"empty logits", func(r *productionHTTPReceipt) { r.logits = nil }},
		{"short logits", func(r *productionHTTPReceipt) { r.logits = r.logits[:1] }},
		{"empty K", func(r *productionHTTPReceipt) { r.k[0] = nil }},
		{"empty V", func(r *productionHTTPReceipt) { r.v[0] = nil }},
		{"short V", func(r *productionHTTPReceipt) { r.v[0] = r.v[0][:1] }},
		{"missing V layer", func(r *productionHTTPReceipt) { r.v = nil }},
		{"zero tokens", func(r *productionHTTPReceipt) { r.tokenCount = 0 }},
	}
	for _, test := range tests {
		for _, arm := range []string{"cold", "warm"} {
			t.Run(test.name+"/"+arm, func(t *testing.T) {
				cold, warm := clone(), clone()
				if arm == "cold" {
					test.mutate(&cold)
				} else {
					test.mutate(&warm)
				}
				if _, _, _, err := compareProductionHTTPReceipt(cfg, cold, warm); err == nil {
					t.Fatal("invalid receipt passed")
				}
			})
		}
	}
}

func newProductionHTTPTestServer(t *testing.T, eng *ModelEngine, cfg Config, width int) (*TelemetryServer, *productionHTTPRecorder) {
	t.Helper()
	srv := NewTelemetryServer(eng, cfg)
	srv.continuous.Close()
	recorder := newProductionHTTPRecorder(scheduledBackend(t, srv.continuousEngine), cfg)
	var err error
	srv.continuous, err = continuousbatch.New(scheduledBackend(t, srv.continuousEngine), continuousbatch.Config{MaxActive: 8, MaxQueue: 8, MaxBatch: 8, BatchWait: 100 * time.Millisecond, TokenBuffer: 2, PrefillWorkers: width})
	if err != nil {
		scheduledBackend(t, srv.continuousEngine).Close()
		t.Fatal(err)
	}
	return srv, recorder
}

func runProductionHTTPBatch(t *testing.T, srv *TelemetryServer, prompts []string) [][]int32 {
	t.Helper()
	got := make([][]int32, len(prompts))
	errs := make(chan error, len(prompts))
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range prompts {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			body, _ := json.Marshal(ChatCompletionRequest{Prompt: prompts[i], MaxTokens: 1})
			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
			if rec.Code != http.StatusOK {
				errs <- &httpError{code: rec.Code, body: rec.Body.String()}
				return
			}
			var response ChatCompletionResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				errs <- err
				return
			}
			got[i] = response.TokenIDs
		}(i)
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	return got
}
