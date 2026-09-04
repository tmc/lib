//go:build darwin

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	apple "github.com/tmc/apple/metal"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/examples/internal/continuousbatch"
	"github.com/tmc/lib/examples/internal/metalprefix"
)

func TestConcurrentHTTPUsesMetalBatchDecode(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = "Llama-3.2-Tiny-Test"
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eng.prefixes = newPrefixCache(64<<20, 8)
	srv := NewTelemetryServer(eng, cfg)
	if srv.continuous == nil {
		t.Fatal("continuous scheduler is disabled")
	}
	if scheduledBackend(t, srv.continuousEngine) == nil {
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
	for _, observation := range scheduledBackend(t, srv.continuousEngine).Observations() {
		if observation.Dispatches != wantDispatches {
			t.Fatalf("batch %d dispatches=%d, want %d", observation.Batch, observation.Dispatches, wantDispatches)
		}
	}
	prefills := scheduledBackend(t, srv.continuousEngine).PrefillObservations()
	sharedPrefill := false
	for _, observation := range prefills {
		if observation.Batch > 1 {
			sharedPrefill = true
		}
	}
	if !sharedPrefill {
		t.Fatalf("no shared prefill wave: %+v", prefills)
	}
}

func TestScheduledSegmentedPrefillWidthsAndFailureIsolation(t *testing.T) {
	for _, width := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("B%d", width), func(t *testing.T) {
			cfg := DefaultTinyLlamaConfig()
			cfg.Name = fmt.Sprintf("segmented-scheduled-B%d", width)
			eng, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			eng.prefixes = newPrefixCache(64<<20, 8)
			adapter, err := newScheduledEngineAdapter(eng, cfg, width)
			if err != nil {
				t.Fatal(err)
			}
			defer adapter.Close()
			inputs := make([]continuousbatch.PrefillRequest, width)
			states := make([]continuousbatch.State, width)
			want := make([]int32, width)
			for i := range width {
				tokens := make([]int32, i+2)
				for j := range tokens {
					tokens[j] = int32(10 + i + j)
				}
				sequence, err := eng.NewSequenceKV(cfg)
				if err != nil {
					t.Fatal(err)
				}
				want[i], err = GPUBatchedPrefillPassKV(eng, cfg, sequence, tokens, 0)
				sequence.Free()
				if err != nil {
					t.Fatal(err)
				}
				states[i] = continuousbatch.State{ID: fmt.Sprintf("request-%d", i), PromptTokenIDs: tokens}
				inputs[i] = continuousbatch.PrefillRequest{Context: t.Context(), State: &states[i]}
			}
			got := adapter.PrefillBatch(inputs)
			for i := range got {
				if got[i].Err != nil || got[i].Token.ID != want[i] || states[i].Position != len(states[i].PromptTokenIDs) {
					t.Fatalf("request %d: got=%+v position=%d want_token=%d", i, got[i], states[i].Position, want[i])
				}
				adapter.Release(&states[i])
			}
			observations := adapter.PrefillObservations()
			if len(observations) != 1 || observations[0].Batch != width {
				t.Fatalf("prefill observations=%+v, want one B=%d wave", observations, width)
			}
		})
	}
}

func TestScheduledPrefixResetMakesNextPrefillCold(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := newScheduledEngineAdapter(eng, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()

	prefill := func(id string) {
		state := continuousbatch.State{ID: id, PromptTokenIDs: []int32{3, 4}}
		if _, err := adapter.Prefill(t.Context(), &state); err != nil {
			t.Fatal(err)
		}
		adapter.Release(&state)
	}
	prefill("before-reset")
	if got := adapter.prefixes.Stats(); got.Entries == 0 || got.DeviceWriteBytes == 0 {
		t.Fatalf("warm prefix stats = %+v", got)
	}
	if err := adapter.resetPrefixes(); err != nil {
		t.Fatal(err)
	}
	if got := adapter.prefixes.Stats(); got.Entries != 0 || got.DeviceReadBytes != 0 || got.DeviceWriteBytes != 0 {
		t.Fatalf("reset prefix stats = %+v", got)
	}
	prefill("after-reset")
	if got := adapter.prefixes.Stats(); got.Misses == 0 || got.DeviceWriteBytes == 0 {
		t.Fatalf("cold prefix stats = %+v", got)
	}
}

func TestScheduledSegmentedPrefillRejectsOneInvalidRequest(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = "segmented-scheduled-partial-failure"
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eng.prefixes = newPrefixCache(64<<20, 8)
	adapter, err := newScheduledEngineAdapter(eng, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	good := continuousbatch.State{ID: "good", PromptTokenIDs: []int32{3, 4}}
	bad := continuousbatch.State{ID: "bad", PromptTokenIDs: []int32{int32(cfg.VocabSize)}}
	results := adapter.PrefillBatch([]continuousbatch.PrefillRequest{{Context: t.Context(), State: &good}, {Context: t.Context(), State: &bad}})
	if results[0].Err != nil || good.KV == nil {
		t.Fatalf("good result=%+v KV=%v", results[0], good.KV)
	}
	if results[1].Err == nil || bad.KV != nil {
		t.Fatalf("bad result=%+v KV=%v", results[1], bad.KV)
	}
	adapter.Release(&good)
	adapter.rowsMu.Lock()
	defer adapter.rowsMu.Unlock()
	if len(adapter.freeRows) != 2 || len(adapter.leases) != 0 {
		t.Fatalf("rows after partial failure: free=%d leases=%d", len(adapter.freeRows), len(adapter.leases))
	}
}

func TestScheduledSegmentedPrefillUsesRowsAboveFreeCount(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = "segmented-scheduled-occupied-row"
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eng.prefixes = newPrefixCache(64<<20, 8)
	adapter, err := newScheduledEngineAdapter(eng, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	first := continuousbatch.State{ID: "first", PromptTokenIDs: []int32{3, 4}}
	if _, err := adapter.Prefill(t.Context(), &first); err != nil {
		t.Fatal(err)
	}
	defer adapter.Release(&first)
	second := continuousbatch.State{ID: "second", PromptTokenIDs: []int32{5, 6}}
	if _, err := adapter.Prefill(t.Context(), &second); err != nil {
		t.Fatalf("prefill second stable row: %v", err)
	}
	adapter.Release(&second)
}

func TestDecodeFailedCompletionFailsWaveAndSuppressesOutput(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := newScheduledEngineAdapter(eng, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	adapter.enablePhysicalLedger()
	states := []*continuousbatch.State{
		{ID: "first", PromptTokenIDs: []int32{3, 4}},
		{ID: "second", PromptTokenIDs: []int32{5, 6}},
	}
	for _, state := range states {
		if _, err := adapter.Prefill(t.Context(), state); err != nil {
			t.Fatal(err)
		}
	}
	wantErr := &libmetal.CommandBufferError{Failure: libmetal.CommandBufferFailed, Status: apple.MTLCommandBufferStatusError, DeviceTerminal: true}
	adapter.prefixes.decode = func(context.Context, *ModelEngine, Config, *BatchedAttentionCache, *metalBlockWorkspace, []*metalprefix.Lease, *BatchedModelTail, []SamplerConfig, []int32, []int, []int) (BatchedLayerChainResult, BatchedModelTailResult, error) {
		return BatchedLayerChainResult{}, BatchedModelTailResult{}, wantErr
	}
	beforeFailure := adapter.physical.Snapshot()
	results := adapter.DecodeBatch(t.Context(), states)
	for i, result := range results {
		if !errors.Is(result.Err, wantErr) {
			t.Fatalf("result %d error = %v, want shared completion failure", i, result.Err)
		}
		if result.Token != (continuousbatch.Token{}) || result.Done {
			t.Fatalf("result %d exposed output after failure: %+v", i, result)
		}
	}
	if waves, err := adapter.physical.Since(beforeFailure); err != nil || len(waves) != 0 {
		t.Fatalf("failed wave was published: %+v, %v", waves, err)
	}
	for _, state := range states {
		adapter.Release(state)
	}
	adapter.rowsMu.Lock()
	defer adapter.rowsMu.Unlock()
	if len(adapter.freeRows) != 2 || len(adapter.leases) != 0 {
		t.Fatalf("rows after failed wave release: free=%d leases=%d", len(adapter.freeRows), len(adapter.leases))
	}
}

func TestScheduledSegmentedPrefillExcludesCanceledBeforeCommit(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = "segmented-scheduled-cancel"
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eng.prefixes = newPrefixCache(64<<20, 8)
	adapter, err := newScheduledEngineAdapter(eng, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	canceledContext, cancel := context.WithCancel(t.Context())
	cancel()
	good := continuousbatch.State{ID: "good", PromptTokenIDs: []int32{3, 4}}
	canceled := continuousbatch.State{ID: "canceled", PromptTokenIDs: []int32{5, 6}}
	results := adapter.PrefillBatch([]continuousbatch.PrefillRequest{{Context: t.Context(), State: &good}, {Context: canceledContext, State: &canceled}})
	if results[0].Err != nil || good.KV == nil {
		t.Fatalf("good result=%+v KV=%v", results[0], good.KV)
	}
	if !errors.Is(results[1].Err, context.Canceled) || canceled.KV != nil {
		t.Fatalf("canceled result=%+v KV=%v", results[1], canceled.KV)
	}
	observations := adapter.PrefillObservations()
	if len(observations) != 1 || observations[0].Batch != 1 {
		t.Fatalf("prefill observations=%+v, want canceled request excluded", observations)
	}
	adapter.Release(&good)
}

func TestScheduledSegmentedPrefillRestores127And128Tokens(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.Name = "segmented-scheduled-prefix-boundary"
	cfg.MaxSeqLen = 256
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	eng.prefixes = newPrefixCache(64<<20, 8)
	adapter, err := newScheduledEngineAdapter(eng, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()
	prefix := make([]int32, 127)
	for i := range prefix {
		prefix[i] = int32(i%200 + 3)
	}
	prime := continuousbatch.State{ID: "prime-127", PromptTokenIDs: prefix}
	if _, err := adapter.Prefill(t.Context(), &prime); err != nil {
		t.Fatal(err)
	}
	adapter.Release(&prime)
	for _, n := range []int{128, 129} {
		tokens := append([]int32(nil), prefix...)
		for len(tokens) < n {
			tokens = append(tokens, int32(len(tokens)%200+3))
		}
		state := continuousbatch.State{ID: fmt.Sprintf("prefix-%d", n), PromptTokenIDs: tokens}
		if _, err := adapter.Prefill(t.Context(), &state); err != nil {
			t.Fatal(err)
		}
		adapter.Release(&state)
	}
	metrics := adapter.prefixes.Stats()
	if metrics.Hits < 2 || metrics.SavedTokens < 254 {
		t.Fatalf("prefix metrics=%+v, want two 127-token block hits", metrics)
	}
}

func TestHTTPFullModelBatchWidths(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, width := range []int{1, 2, 4, 8} {
		eng.prefixes = newPrefixCache(64<<20, 8)
		srv := NewTelemetryServer(eng, cfg)
		scheduledBackend(t, srv.continuousEngine).enablePhysicalLedger()
		prompts := make([]string, width)
		want := make([][]int32, width)
		for i := range width {
			prompts[i] = fmt.Sprintf("prompt-%d-%s", i, strings.Repeat("x", i))
			want[i] = cpuReceipt(t, eng, cfg, prompts[i], 4)
		}
		start := make(chan struct{})
		ledgerBefore := scheduledBackend(t, srv.continuousEngine).physical.Snapshot()
		errs := make(chan error, width)
		var wg sync.WaitGroup
		for i := range width {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				<-start
				body, _ := json.Marshal(ChatCompletionRequest{Prompt: prompts[i], MaxTokens: 4})
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
				if !reflect.DeepEqual(response.TokenIDs, want[i]) {
					errs <- fmt.Errorf("request %d tokens=%v, want %v", i, response.TokenIDs, want[i])
				}
			}(i)
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
		prefillFound := false
		for _, observation := range scheduledBackend(t, srv.continuousEngine).PrefillObservations() {
			if observation.Batch == width {
				prefillFound = true
			}
		}
		if !prefillFound {
			t.Fatalf("width %d did not reach physical prefill batch: %+v", width, scheduledBackend(t, srv.continuousEngine).PrefillObservations())
		}
		waves, err := scheduledBackend(t, srv.continuousEngine).physical.Since(ledgerBefore)
		if err != nil {
			t.Fatal(err)
		}
		sharedDecode := false
		for _, wave := range waves {
			if wave.Kind == "decode-chain-tail" && len(wave.Requests) == width {
				sharedDecode = true
			}
		}
		if !sharedDecode {
			t.Fatalf("width %d missing shared decode ledger wave: %+v", width, waves)
		}
		t.Logf("width=%d prefills=%+v batches=%+v physical_waves=%+v", width, scheduledBackend(t, srv.continuousEngine).PrefillObservations(), scheduledBackend(t, srv.continuousEngine).Observations(), waves)
		srv.continuous.Close()
		scheduledBackend(t, srv.continuousEngine).Close()
	}
}

func TestHTTPProductionMetalPrefixMixedWidths(t *testing.T) {
	for _, width := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("B%d", width), func(t *testing.T) {
			cfg := DefaultTinyLlamaConfig()
			eng, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			srv := NewTelemetryServer(eng, cfg)
			defer srv.continuous.Close()
			defer scheduledBackend(t, srv.continuousEngine).Close()
			const common = "leased-prefix:"
			primeHTTPPrompt(t, srv, common)
			before := scheduledBackend(t, srv.continuousEngine).prefixes.Stats()
			prompts := make([]string, width)
			want := make([][]int32, width)
			got := make([][]int32, width)
			start := make(chan struct{})
			errs := make(chan error, width)
			var wg sync.WaitGroup
			for i := range width {
				prompts[i] = fmt.Sprintf("%s%c", common, 'a'+i)
				want[i] = cpuReceipt(t, eng, cfg, prompts[i], 3)
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					<-start
					body, _ := json.Marshal(ChatCompletionRequest{Prompt: prompts[i], MaxTokens: 3})
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
			for i := range width {
				if !reflect.DeepEqual(got[i], want[i]) {
					t.Fatalf("request %d tokens=%v, want %v", i, got[i], want[i])
				}
			}
			after := scheduledBackend(t, srv.continuousEngine).prefixes.Stats()
			if after.Hits-before.Hits < int64(width) || after.Entries != before.Entries+width || after.Bytes <= before.Bytes || after.Blocks <= before.Blocks || after.DeviceReadBytes <= before.DeviceReadBytes || after.DeviceWriteBytes <= before.DeviceWriteBytes ||
				after.WorkspaceBytes != before.WorkspaceBytes || after.DescriptorAddress != before.DescriptorAddress || after.RowAddress != before.RowAddress {
				t.Fatalf("prefix stats before=%+v after=%+v", before, after)
			}
			found := width == 1
			for _, observation := range scheduledBackend(t, srv.continuousEngine).PrefillObservations() {
				if observation.Batch == width {
					found = true
				}
			}
			if !found {
				t.Fatalf("B=%d did not reach a shared leased-prefix wave: %+v", width, scheduledBackend(t, srv.continuousEngine).PrefillObservations())
			}
			t.Logf("width=%d prefix=%+v prefills=%+v", width, after, scheduledBackend(t, srv.continuousEngine).PrefillObservations())
		})
	}
}

func TestScheduledMetalPrefixPartitionsHitAndMiss(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := newScheduledEngineAdapter(eng, cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer adapter.Close()

	prime := continuousbatch.State{ID: "prime", PromptTokenIDs: []int32{3, 4, 5}}
	if _, err := adapter.Prefill(t.Context(), &prime); err != nil {
		t.Fatal(err)
	}
	adapter.Release(&prime)
	beforeObservations := len(adapter.PrefillObservations())
	beforeStats := adapter.prefixes.Stats()

	states := []continuousbatch.State{
		{ID: "hit", PromptTokenIDs: []int32{3, 4, 5, 6}},
		{ID: "miss", PromptTokenIDs: []int32{100, 101, 102, 103, 104}},
	}
	want := make([]int32, len(states))
	inputs := make([]continuousbatch.PrefillRequest, len(states))
	for i := range states {
		sequence, err := eng.NewSequenceKV(cfg)
		if err != nil {
			t.Fatal(err)
		}
		want[i], err = GPUBatchedPrefillPassKV(eng, cfg, sequence, states[i].PromptTokenIDs, 0)
		sequence.Free()
		if err != nil {
			t.Fatal(err)
		}
		inputs[i] = continuousbatch.PrefillRequest{Context: t.Context(), State: &states[i]}
	}
	results := adapter.PrefillBatch(inputs)
	for i := range results {
		if results[i].Err != nil || results[i].Token.ID != want[i] {
			t.Fatalf("row %d result=%+v want token %d", i, results[i], want[i])
		}
		adapter.Release(&states[i])
	}

	observations := adapter.PrefillObservations()[beforeObservations:]
	wantLeased := 1 + 9*cfg.NumLayers + 3
	wantSegmented := 2 + 9*cfg.NumLayers + 3
	if len(observations) != 2 || observations[0] != (fullBatchObservation{Batch: 1, Dispatches: wantLeased}) || observations[1] != (fullBatchObservation{Batch: 1, Dispatches: wantSegmented}) {
		t.Fatalf("mixed observations=%+v, want one leased %d-dispatch wave and one segmented %d-dispatch wave", observations, wantLeased, wantSegmented)
	}
	afterStats := adapter.prefixes.Stats()
	if afterStats.Hits-beforeStats.Hits != 1 || afterStats.Misses-beforeStats.Misses != 1 {
		t.Fatalf("mixed prefix stats before=%+v after=%+v", beforeStats, afterStats)
	}
	t.Logf("observations=%+v prefix_before=%+v prefix_after=%+v", observations, beforeStats, afterStats)
}

func TestHTTPPrefixPressureFairness(t *testing.T) {
	for _, width := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("B%d", width), func(t *testing.T) {
			cfg := DefaultTinyLlamaConfig()
			eng, err := GetOrCompileEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			srv := NewTelemetryServer(eng, cfg)
			defer srv.continuous.Close()
			defer scheduledBackend(t, srv.continuousEngine).Close()

			bases := make([]string, 10)
			for i := range bases {
				bases[i] = fmt.Sprintf("pressure-prefix-%02d:", i)
				primeHTTPPrompt(t, srv, bases[i])
			}
			if stats := scheduledBackend(t, srv.continuousEngine).prefixes.Stats(); stats.Entries != 8 {
				t.Fatalf("eviction did not bound entries: %+v", stats)
			}

			requests := 2 * width
			prompts := make([]string, requests)
			limits := make([]int, requests)
			want := make([][]int32, requests)
			got := make([][]int32, requests)
			before := scheduledBackend(t, srv.continuousEngine).prefixes.Stats()
			for i := range requests {
				if i < width {
					prompts[i] = fmt.Sprintf("%s%c", bases[9-i], 'a'+i)
				} else {
					prompts[i] = fmt.Sprintf("cold-pressure-%d-%s", i, strings.Repeat("x", i+1))
				}
				limits[i] = 1 + i%4
				want[i] = cpuReceipt(t, eng, cfg, prompts[i], limits[i])
			}

			start := make(chan struct{})
			errs := make(chan error, requests)
			var wg sync.WaitGroup
			for i := range requests {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					<-start
					ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
					defer cancel()
					body, _ := json.Marshal(ChatCompletionRequest{Prompt: prompts[i], MaxTokens: limits[i]})
					rec := httptest.NewRecorder()
					srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)).WithContext(ctx))
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
			for i := range requests {
				if !reflect.DeepEqual(got[i], want[i]) {
					t.Fatalf("request %d tokens=%v, want %v", i, got[i], want[i])
				}
			}

			canceled, cancel := context.WithCancel(t.Context())
			cancel()
			body, _ := json.Marshal(ChatCompletionRequest{Prompt: "canceled-before-prefill", MaxTokens: 4})
			rec := httptest.NewRecorder()
			srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)).WithContext(canceled))

			after := scheduledBackend(t, srv.continuousEngine).prefixes.Stats()
			if after.Hits <= before.Hits {
				t.Fatalf("hot requests recorded no prefix hit: before=%+v after=%+v", before, after)
			}
			found := width == 1
			for _, observation := range scheduledBackend(t, srv.continuousEngine).PrefillObservations() {
				if observation.Batch == width {
					found = true
				}
			}
			if !found {
				t.Fatalf("no physical B=%d prefill: %+v", width, scheduledBackend(t, srv.continuousEngine).PrefillObservations())
			}
			scheduledBackend(t, srv.continuousEngine).rowsMu.Lock()
			freeRows, leases := len(scheduledBackend(t, srv.continuousEngine).freeRows), len(scheduledBackend(t, srv.continuousEngine).leases)
			scheduledBackend(t, srv.continuousEngine).rowsMu.Unlock()
			if freeRows != 8 || leases != 0 {
				t.Fatalf("row cleanup free=%d leases=%d", freeRows, leases)
			}
			t.Logf("width=%d limits=%v prefix_before=%+v prefix_after=%+v prefills=%+v decode=%+v", width, limits, before, after, scheduledBackend(t, srv.continuousEngine).PrefillObservations(), scheduledBackend(t, srv.continuousEngine).Observations())
		})
	}
}

func primeHTTPPrompt(t *testing.T, srv *TelemetryServer, prompt string) {
	t.Helper()
	body, _ := json.Marshal(ChatCompletionRequest{Prompt: prompt, MaxTokens: 1})
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("prime prefix: %s", rec.Body.String())
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
	if got := adapter.cache.K[0].HostSlice()[offset]; got != want+1000 {
		t.Fatalf("dense row was restored from prefix blocks: got %g, want retained mutation %g", got, want+1000)
	}
	metrics := adapter.prefixes.Stats()
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
	for i := 0; i < 9; i++ {
		prompt := fmt.Sprintf("distinct prompt %d", i)
		state := &continuousbatch.State{Prompt: prompt}
		if _, err := adapter.Prefill(t.Context(), state); err != nil {
			t.Fatal(err)
		}
		adapter.Release(state)
	}
	metrics := adapter.prefixes.Stats()
	if metrics.Entries != 8 || metrics.Bytes <= 0 || metrics.Bytes > 64<<20 || metrics.Misses != 9 {
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
