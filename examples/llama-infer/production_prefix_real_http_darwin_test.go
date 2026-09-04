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
	logits []float32
	k, v   [][]float32
	tokens []int32
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
		r.receipts[state.Prompt] = productionHTTPReceipt{logits: append([]float32(nil), logits...), k: k, v: v}
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
			coldReceipts, err := coldRecorder.snapshot()
			if err != nil {
				t.Fatal(err)
			}
			cold.continuous.Close()
			scheduledBackend(t, cold.continuousEngine).Close()

			warm, warmRecorder := newProductionHTTPTestServer(t, eng, cfg, width)
			primeHTTPPrompt(t, warm, common)
			warmRecorder.mu.Lock()
			warmRecorder.receipts = make(map[string]productionHTTPReceipt)
			warmRecorder.mu.Unlock()
			warmTokens := runProductionHTTPBatch(t, warm, prompts)
			servingStats := scheduledBackend(t, warm.continuousEngine).prefixes.Stats()
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
				if !coldOK || !warmOK || len(coldReceipt.logits) != cfg.VocabSize || len(warmReceipt.logits) != cfg.VocabSize {
					t.Fatalf("prompt %q receipt cold=%v warm=%v logits=%d/%d", prompt, coldOK, warmOK, len(coldReceipt.logits), len(warmReceipt.logits))
				}
				for i := range coldReceipt.logits {
					maxLogit = max(maxLogit, float32(math.Abs(float64(coldReceipt.logits[i]-warmReceipt.logits[i]))))
				}
				if len(coldReceipt.k) != cfg.NumLayers || len(warmReceipt.k) != cfg.NumLayers {
					t.Fatalf("prompt %q layer count differs", prompt)
				}
				for layer := range cfg.NumLayers {
					if len(coldReceipt.k[layer]) != len(warmReceipt.k[layer]) || len(coldReceipt.v[layer]) != len(warmReceipt.v[layer]) {
						t.Fatalf("prompt %q layer %d KV shape differs", prompt, layer)
					}
					for i := range coldReceipt.k[layer] {
						maxKV = max(maxKV, float32(math.Abs(float64(coldReceipt.k[layer][i]-warmReceipt.k[layer][i]))))
						maxKV = max(maxKV, float32(math.Abs(float64(coldReceipt.v[layer][i]-warmReceipt.v[layer][i]))))
					}
					coordinates += len(coldReceipt.k[layer]) + len(coldReceipt.v[layer])
				}
			}
			if maxLogit > 5e-4 || maxKV > 5e-4 {
				t.Fatalf("max logit=%g KV=%g", maxLogit, maxKV)
			}
			t.Logf("width=%d model=%s logits=%d kv_coordinates=%d max_logit=%g max_kv=%g observations=%+v", width, model, width*cfg.VocabSize, coordinates, maxLogit, maxKV, observations)
		})
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
