//go:build darwin

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/tmc/lib/examples/internal/continuousbatch"
)

func TestConcurrentHTTPUsesMetalBatchDecode(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = "Llama-3.2-Tiny-Test"
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	srv := NewTelemetryServer(eng, cfg)
	if srv.continuous == nil {
		t.Fatal("continuous scheduler is disabled")
	}
	if srv.continuousEngine == nil {
		t.Fatal("full-model batch backend is disabled")
	}

	start := make(chan struct{})
	outputs := make([][]int32, 4)
	errs := make(chan error, len(outputs))
	var wg sync.WaitGroup
	for i := range outputs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			body, _ := json.Marshal(ChatCompletionRequest{Prompt: "same prompt", MaxTokens: 4})
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				errs <- &httpError{code: rec.Code, body: rec.Body.String()}
				return
			}
			var response ChatCompletionResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				errs <- err
				return
			}
			outputs[i] = response.TokenIDs
		}(i)
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	for i := 1; i < len(outputs); i++ {
		if !reflect.DeepEqual(outputs[0], outputs[i]) {
			t.Fatalf("request 0 tokens %v differ from request %d tokens %v", outputs[0], i, outputs[i])
		}
	}
	cpu := cpuReceipt(t, eng, cfg, "same prompt", 4)
	sequential := sequentialGPUReceipt(t, eng, cfg, "same prompt", 4)
	if !reflect.DeepEqual(outputs[0], cpu) || !reflect.DeepEqual(outputs[0], sequential) {
		t.Fatalf("HTTP=%v CPU=%v sequential_GPU=%v", outputs[0], cpu, sequential)
	}
	shared := false
	for size := range srv.continuous.Summary().BatchSizes {
		if size > 1 {
			shared = true
		}
	}
	if !shared {
		t.Fatalf("no shared decode batch: %+v", srv.continuous.Summary().BatchSizes)
	}
	wantDispatches := 1 + 9*cfg.NumLayers + 3
	for _, observation := range srv.continuousEngine.Observations() {
		if observation.Dispatches != wantDispatches {
			t.Fatalf("batch %d dispatches=%d, want %d", observation.Batch, observation.Dispatches, wantDispatches)
		}
	}
}

func TestHTTPFullModelBatchWidths(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := cpuReceipt(t, eng, cfg, "same prompt", 4)
	for _, width := range []int{1, 2, 4, 8} {
		srv := NewTelemetryServer(eng, cfg)
		start := make(chan struct{})
		errs := make(chan error, width)
		var wg sync.WaitGroup
		for range width {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				body, _ := json.Marshal(ChatCompletionRequest{Prompt: "same prompt", MaxTokens: 4})
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
					errs <- fmt.Errorf("tokens=%v, want %v", response.TokenIDs, want)
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
		for _, observation := range srv.continuousEngine.Observations() {
			if observation.Dispatches != 1+9*cfg.NumLayers+3 {
				t.Fatalf("width=%d observation=%+v", width, observation)
			}
			if observation.Batch == width {
				found = true
			}
		}
		if !found {
			t.Fatalf("width %d did not reach physical batch: %+v", width, srv.continuousEngine.Observations())
		}
		t.Logf("width=%d batches=%+v", width, srv.continuousEngine.Observations())
		srv.continuous.Close()
		srv.continuousEngine.Close()
	}
}

func TestScheduledPrefixSnapshotRowLease(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	previous := eng.prefixes
	eng.prefixes = newPrefixCache(64<<20, 2)
	t.Cleanup(func() { eng.prefixes = previous })
	adapter, err := newScheduledEngineAdapter(eng, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	first := &continuousbatch.State{Prompt: "same prompt"}
	firstToken, err := adapter.Prefill(t.Context(), first)
	if err != nil {
		t.Fatal(err)
	}
	firstLease := first.KV.(scheduledSequenceState)
	offset := firstLease.row * adapter.cache.maxSeqLen * adapter.cache.kvDim
	want := adapter.cache.K[0].HostSlice()[offset]
	adapter.cache.K[0].HostSlice()[offset] = want + 1000
	stale := *first
	adapter.Release(first)
	second := &continuousbatch.State{Prompt: "same prompt"}
	secondToken, err := adapter.Prefill(t.Context(), second)
	if err != nil {
		t.Fatal(err)
	}
	secondLease := second.KV.(scheduledSequenceState)
	if secondLease.row != firstLease.row || secondLease.lease == firstLease.lease {
		t.Fatalf("row leases first=%+v second=%+v", firstLease, secondLease)
	}
	if secondToken.ID != firstToken.ID {
		t.Fatalf("cached token=%d, want %d", secondToken.ID, firstToken.ID)
	}
	if got := adapter.cache.K[0].HostSlice()[offset]; got != want {
		t.Fatalf("cached row mutation leaked: got %g, want %g", got, want)
	}
	metrics := eng.prefixes.metrics()
	if metrics.Hits != 1 || metrics.Entries != 1 || metrics.Bytes <= 0 || metrics.SavedTokens <= 0 {
		t.Fatalf("prefix metrics=%+v", metrics)
	}
	adapter.Release(&stale)
	adapter.rowsMu.Lock()
	free, leased := len(adapter.freeRows), len(adapter.leases)
	adapter.rowsMu.Unlock()
	if free != 0 || leased != 1 {
		t.Fatalf("stale release: free=%d leased=%d, want 0,1", free, leased)
	}
	adapter.Release(second)
	adapter.rowsMu.Lock()
	free, leased = len(adapter.freeRows), len(adapter.leases)
	adapter.rowsMu.Unlock()
	if free != 1 || leased != 0 {
		t.Fatalf("final release: free=%d leased=%d, want 1,0", free, leased)
	}
}

func TestScheduledPrefixCacheEvictionIsBounded(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	previous := eng.prefixes
	eng.prefixes = newPrefixCache(64<<20, 1)
	t.Cleanup(func() { eng.prefixes = previous })
	adapter, err := newScheduledEngineAdapter(eng, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	for _, prompt := range []string{"first prompt", "second distinct prompt"} {
		state := &continuousbatch.State{Prompt: prompt}
		if _, err := adapter.Prefill(t.Context(), state); err != nil {
			t.Fatal(err)
		}
		adapter.Release(state)
	}
	metrics := eng.prefixes.metrics()
	if metrics.Entries != 1 || metrics.Bytes <= 0 || metrics.Bytes > 64<<20 || metrics.Misses != 2 {
		t.Fatalf("bounded prefix metrics=%+v", metrics)
	}
}

func cpuReceipt(t *testing.T, eng *ModelEngine, cfg Config, prompt string, limit int) []int32 {
	t.Helper()
	tokens, err := eng.tokenizer.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) == 0 {
		tokens = []int32{1}
	}
	if cfg.BOSTokenID != 0 {
		tokens = append([]int32{cfg.BOSTokenID}, tokens...)
	}
	mcfg := ModelConfig{HiddenSize: cfg.Dim, IntermediateSize: cfg.HiddenDim, NumHiddenLayers: cfg.NumLayers, NumAttentionHeads: cfg.NumHeads, NumKeyValueHeads: cfg.NumKVHeads, VocabSize: cfg.VocabSize, RmsNormEps: cfg.Eps, RopeTheta: cfg.RopeTheta, MaxSeqLen: cfg.MaxSeqLen}
	weights := GenerateDeterministicWeights(&mcfg)
	kv := NewKVCacheState(mcfg.NumHiddenLayers, mcfg.MaxSeqLen, mcfg.NumKeyValueHeads*mcfg.HeadDim())
	var logits []float32
	for pos, token := range tokens {
		logits = Forward(weights, &mcfg, kv, token, pos)
	}
	current := sampleArgmax(logits)
	out := []int32{current}
	for len(out) < limit {
		logits = Forward(weights, &mcfg, kv, current, len(tokens)+len(out)-1)
		current = sampleArgmax(logits)
		out = append(out, current)
	}
	return out
}

func sequentialGPUReceipt(t *testing.T, eng *ModelEngine, cfg Config, prompt string, limit int) []int32 {
	t.Helper()
	tokens, err := eng.tokenizer.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) == 0 {
		tokens = []int32{1}
	}
	if cfg.BOSTokenID != 0 {
		tokens = append([]int32{cfg.BOSTokenID}, tokens...)
	}
	kv, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer kv.Free()
	current, err := GPUPrefillPassKV(eng, cfg, kv, tokens)
	if err != nil {
		t.Fatal(err)
	}
	out := []int32{current}
	for len(out) < limit {
		current, err = GPUForwardPassKV(eng, cfg, kv, current, len(tokens)+len(out)-1)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, current)
	}
	return out
}

type httpError struct {
	code int
	body string
}

func (e *httpError) Error() string { return http.StatusText(e.code) + ": " + e.body }
