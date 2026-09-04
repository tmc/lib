//go:build darwin

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/examples/internal/continuousbatch"
)

func TestFP16RealModelBatchedHTTPMatchesSequentialGPU(t *testing.T) {
	modelPath := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if modelPath == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(modelPath)
	cfg.Precision = "f16"
	cfg.MaxSeqLen = 128
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	previousPrefixes := eng.prefixes
	eng.prefixes = newPrefixCache(64<<20, 8)
	t.Cleanup(func() { eng.prefixes = previousPrefixes })
	if !eng.isF16 || len(eng.layersF16) != cfg.NumLayers {
		t.Fatalf("loaded precision f16=%v layers=%d, want %d", eng.isF16, len(eng.layersF16), cfg.NumLayers)
	}
	compareTailCeiling(t, eng, cfg)
	prompt := "The capital of France is"
	want := sequentialGPUReceipt(t, eng, cfg, prompt, 3)
	testFP16RealModelLogits(t, eng, cfg, prompt)
	for _, width := range []int{1, 2, 4} {
		srv := NewTelemetryServer(eng, cfg)
		if scheduledBackend(t, srv.continuousEngine) == nil {
			t.Fatal("FP16 batch backend is disabled")
		}
		srv.continuous.Close()
		srv.continuous, err = continuousbatch.New(scheduledBackend(t, srv.continuousEngine), continuousbatch.Config{MaxActive: 8, MaxQueue: 8, MaxBatch: 8, BatchWait: 100 * time.Millisecond, TokenBuffer: 8, PrefillWorkers: 4})
		if err != nil {
			t.Fatal(err)
		}
		start := make(chan struct{})
		gate := newPrefillGate(scheduledBackend(t, srv.continuousEngine), width)
		srv.continuous.Close()
		srv.continuous, err = continuousbatch.New(gate, continuousbatch.Config{MaxActive: 8, MaxQueue: 8, MaxBatch: 8, BatchWait: 100 * time.Millisecond, TokenBuffer: 8, PrefillWorkers: 4})
		if err != nil {
			t.Fatal(err)
		}
		errs := make(chan error, width)
		var wg sync.WaitGroup
		for range width {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				body, _ := json.Marshal(ChatCompletionRequest{Prompt: prompt, MaxTokens: 3})
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
				if !reflect.DeepEqual(response.TokenIDs, want) {
					errs <- &tokenMismatchError{got: response.TokenIDs, want: want}
				}
			}()
		}
		close(start)
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Fatal(err)
		}
		found := false
		for _, observation := range scheduledBackend(t, srv.continuousEngine).Observations() {
			if observation.Dispatches != 1+9*cfg.NumLayers+3 {
				t.Fatalf("width=%d observation=%+v", width, observation)
			}
			if observation.Batch == width {
				found = true
			}
		}
		if !found {
			t.Fatalf("width %d did not reach physical batch: %+v", width, scheduledBackend(t, srv.continuousEngine).Observations())
		}
		if scheduledBackend(t, srv.continuousEngine).tail.logits.Len() != 0 {
			t.Fatalf("width=%d serving tail allocated %d diagnostic logits", width, scheduledBackend(t, srv.continuousEngine).tail.logits.Len())
		}
		tiles := (cfg.VocabSize + 255) / 256
		t.Logf("width=%d serving_candidates=%d candidate_bytes=%d diagnostic_logits=0", width, width*tiles, width*tiles*2*4)
		t.Logf("width=%d tokens=%v batches=%+v", width, want, scheduledBackend(t, srv.continuousEngine).Observations())
		srv.continuous.Close()
		scheduledBackend(t, srv.continuousEngine).Close()
	}
	kvBytes := int64(2 * cfg.NumLayers * 8 * cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim * 4)
	t.Logf("weights=shared-f16 cache_rows=8 kv_bytes=%d vocab=%d tiles_per_row=%d", kvBytes, cfg.VocabSize, (cfg.VocabSize+255)/256)
	if metrics := eng.prefixes.metrics(); metrics.Entries != 1 || metrics.Hits < 1 || metrics.Bytes <= 0 {
		t.Fatalf("explicit prefix cache metrics=%+v", metrics)
	} else {
		t.Logf("explicit_prefix_cache=%+v", metrics)
	}
}

type prefillGate struct {
	*scheduledEngineAdapter
	want  int
	mu    sync.Mutex
	n     int
	ready chan struct{}
}

func newPrefillGate(backend *scheduledEngineAdapter, want int) *prefillGate {
	return &prefillGate{scheduledEngineAdapter: backend, want: want, ready: make(chan struct{})}
}

func (g *prefillGate) Prefill(ctx context.Context, state *continuousbatch.State) (continuousbatch.Token, error) {
	token, err := g.scheduledEngineAdapter.Prefill(ctx, state)
	if err != nil {
		return token, err
	}
	g.mu.Lock()
	g.n++
	if g.n == g.want {
		close(g.ready)
	}
	g.mu.Unlock()
	select {
	case <-g.ready:
		return token, nil
	case <-ctx.Done():
		return continuousbatch.Token{}, ctx.Err()
	}
}

func compareTailCeiling(t *testing.T, eng *ModelEngine, cfg Config) {
	t.Helper()
	hidden := tailTestHidden(1, cfg.Dim)
	tail, err := eng.NewBatchedModelTail(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer tail.Free()
	diagnosticStart := time.Now()
	diagnostic, err := tail.Run(eng, cfg, hidden, nil)
	if err != nil {
		t.Fatal(err)
	}
	diagnosticElapsed := time.Since(diagnosticStart)
	input, err := mempool.AllocFromSlice(eng.dev, hidden)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Free()
	tiledStart := time.Now()
	tiled, err := tail.runBuffer(eng, cfg, input, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	tiledElapsed := time.Since(tiledStart)
	if !reflect.DeepEqual(tiled.Tokens, diagnostic.Tokens) {
		t.Fatalf("tiled tokens=%v diagnostic=%v", tiled.Tokens, diagnostic.Tokens)
	}
	t.Logf("tail_ceiling_b=1 diagnostic=%s tiled=%s diagnostic_logits_bytes=%d tiled_candidate_bytes=%d", diagnosticElapsed, tiledElapsed, cfg.VocabSize*4, ((cfg.VocabSize+255)/256)*2*4)
}

func testFP16RealModelLogits(t *testing.T, eng *ModelEngine, cfg Config, prompt string) {
	t.Helper()
	tokens, err := eng.tokenizer.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	tokens = append([]int32{cfg.BOSTokenID}, tokens...)
	sequentialKV, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	current, err := GPUPrefillPassKV(eng, cfg, sequentialKV, tokens)
	if err != nil {
		t.Fatal(err)
	}
	wantToken, err := GPUForwardPassKV(eng, cfg, sequentialKV, current, len(tokens))
	if err != nil {
		t.Fatal(err)
	}
	wantLogits, err := eng.ReadLogits()
	if err != nil {
		t.Fatal(err)
	}
	sequentialKV.Free()

	batchKV, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	batchCurrent, err := GPUPrefillPassKV(eng, cfg, batchKV, tokens)
	if err != nil {
		t.Fatal(err)
	}
	cache, err := eng.NewBatchedAttentionCache(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Free()
	if err := cache.ImportSequence(0, len(tokens), batchKV); err != nil {
		t.Fatal(err)
	}
	batchKV.Free()
	chain, err := RunBatchedLayerChain(eng, cfg, cache, []int32{batchCurrent}, []int{len(tokens)}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	tail, err := eng.NewBatchedModelTail(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer tail.Free()
	got, err := RunBatchedModelTailFromChain(eng, cfg, tail, chain, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Tokens[0] != wantToken {
		t.Fatalf("batched token=%d, sequential=%d", got.Tokens[0], wantToken)
	}
	var maxAbs float32
	for i, value := range got.Logits {
		d := float32(math.Abs(float64(value - wantLogits[i])))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 0.02 {
		t.Fatalf("max logit difference %g exceeds 0.02", maxAbs)
	}
	t.Logf("sequential_gpu_logit_max_abs=%g token=%d", maxAbs, wantToken)
}

type tokenMismatchError struct{ got, want []int32 }

func (e *tokenMismatchError) Error() string { return "batched tokens differ from sequential GPU" }
