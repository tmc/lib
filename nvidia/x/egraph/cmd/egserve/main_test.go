package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

func postJSON(t *testing.T, client *http.Client, url string, payload any) *CompletionResponse {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	resp, err := client.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		errBody, _ := json.Marshal(payload)
		t.Fatalf("POST %s status = %d, want 200 (req=%s)", url, resp.StatusCode, string(errBody))
	}
	var out CompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode response from %s: %v", url, err)
	}
	return &out
}

func TestEGServerMultiTurnChatPrefixCachingAndC2CPromotion(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:            "sm_121a",
		HiddenDim:       256,
		IntermDim:       512,
		Layers:          4,
		PageSize:        4,
		TotalKVPages:    128,
		HBMPageCapacity: 12,
		C2CPageCapacity: 32,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	sysPrompt := ChatMessage{
		Role:    "system",
		Content: "You are an expert CUDA e-graph persistent megakernel compiler assistant.",
	}
	userTurn1 := ChatMessage{
		Role:    "user",
		Content: "How do warp-specialized sub-CTA stages avoid __syncthreads deadlock?",
	}

	// Turn 1: cold cache -> 0 prefix hit tokens.
	resp1 := postJSON(t, ts.Client(), ts.URL+"/v1/chat/completions", map[string]any{
		"model": "egraph-llama-3",
		"messages": []ChatMessage{
			sysPrompt,
			userTurn1,
		},
		"max_tokens":  16,
		"temperature": 0.0,
	})
	if !resp1.EGraphFlight.BarrierSafetyVerified {
		t.Fatalf("turn 1 BarrierSafetyVerified = false, want true")
	}
	if resp1.EGraphFlight.RadixPrefixHitTokens != 0 {
		t.Fatalf("turn 1 RadixPrefixHitTokens = %d, want 0 on cold start", resp1.EGraphFlight.RadixPrefixHitTokens)
	}

	// Turn 2: shares system prompt prefix -> positive radix_prefix_hit_tokens > turn 1.
	resp2 := postJSON(t, ts.Client(), ts.URL+"/v1/chat/completions", map[string]any{
		"model": "egraph-llama-3",
		"messages": []ChatMessage{
			sysPrompt,
			{Role: "user", Content: "What is the role of named barrier bar.sync 1 in sub-CTA stages?"},
		},
		"max_tokens":  16,
		"temperature": 0.0,
	})
	if resp2.EGraphFlight.RadixPrefixHitTokens <= resp1.EGraphFlight.RadixPrefixHitTokens {
		t.Fatalf("turn 2 RadixPrefixHitTokens = %d, want > turn 1 (%d)",
			resp2.EGraphFlight.RadixPrefixHitTokens, resp1.EGraphFlight.RadixPrefixHitTokens)
	}

	// Turn 3: shares system prompt + turn 1 user + turn 1 assistant -> even larger prefix hit.
	resp3 := postJSON(t, ts.Client(), ts.URL+"/v1/chat/completions", map[string]any{
		"model": "egraph-llama-3",
		"messages": []ChatMessage{
			sysPrompt,
			userTurn1,
			{Role: "assistant", Content: resp1.Choices[0].Message.Content},
			{Role: "user", Content: "Show how the ping-pong arena verifier checks WAR dominance."},
		},
		"max_tokens":  16,
		"temperature": 0.0,
	})
	if resp3.EGraphFlight.RadixPrefixHitTokens <= resp2.EGraphFlight.RadixPrefixHitTokens {
		t.Fatalf("turn 3 RadixPrefixHitTokens = %d, want > turn 2 (%d)",
			resp3.EGraphFlight.RadixPrefixHitTokens, resp2.EGraphFlight.RadixPrefixHitTokens)
	}

	// Send unrelated requests to fill Tier-1 HBM3e (capacity 12 pages) and demote
	// the earlier conversation prefix into Tier-2 Grace C2C LPDDR5X.
	for i := range 3 {
		_ = postJSON(t, ts.Client(), ts.URL+"/v1/completions", map[string]any{
			"model":  "egraph-llama-3",
			"prompt": "Unrelated eviction burst sequence alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " + string(rune('A'+i)),
		})
	}

	// Turn 4: revisits sysPrompt -> promotes pages from Tier-2 Grace C2C back to Tier-1 HBM3e.
	resp4 := postJSON(t, ts.Client(), ts.URL+"/v1/chat/completions", map[string]any{
		"model": "egraph-llama-3",
		"messages": []ChatMessage{
			sysPrompt,
			{Role: "user", Content: "Verify Tier-2 Grace C2C page promotion."},
		},
		"max_tokens": 16,
	})
	if resp4.EGraphFlight.C2CPromotedPages <= 0 {
		t.Fatalf("turn 4 C2CPromotedPages = %d, want > 0 after HBM eviction to C2C", resp4.EGraphFlight.C2CPromotedPages)
	}
	if resp4.EGraphFlight.RadixPrefixHitTokens <= 0 {
		t.Fatalf("turn 4 RadixPrefixHitTokens = %d, want > 0", resp4.EGraphFlight.RadixPrefixHitTokens)
	}
}

func TestEGServerConcurrentBatchingAndBucketTransitions(t *testing.T) {
	srv, err := NewEGServer(ServerConfig{
		Arch:        "sm_121a",
		HiddenDim:   256,
		IntermDim:   512,
		Layers:      4,
		BatchWindow: 5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewEGServer: %v", err)
	}
	ts := httptest.NewServer(srv)
	defer ts.Close()

	// Route across buckets M in {1, 2, 4, 8} via draft_tokens and concurrent batching.
	observedBuckets := make(map[int]bool)

	for _, tc := range []struct {
		prompt      string
		draftTokens int
		wantBucketM int
	}{
		{prompt: "Single token greedy decode step", draftTokens: 0, wantBucketM: 1},
		{prompt: "Two token speculative verify step", draftTokens: 1, wantBucketM: 2},
		{prompt: "Four token speculative tree verify step", draftTokens: 3, wantBucketM: 4},
		{prompt: "Eight token speculative tree verify step", draftTokens: 7, wantBucketM: 8},
	} {
		resp := postJSON(t, ts.Client(), ts.URL+"/v1/completions", map[string]any{
			"model":        "egraph-llama-3",
			"prompt":       tc.prompt,
			"max_tokens":   8,
			"draft_tokens": tc.draftTokens,
		})
		if resp.EGraphFlight.ActiveBucketM != tc.wantBucketM {
			t.Fatalf("draft_tokens=%d got ActiveBucketM=%d, want %d",
				tc.draftTokens, resp.EGraphFlight.ActiveBucketM, tc.wantBucketM)
		}
		if resp.EGraphFlight.MegakernelKernelName == "" {
			t.Fatalf("draft_tokens=%d returned empty MegakernelKernelName", tc.draftTokens)
		}
		observedBuckets[resp.EGraphFlight.ActiveBucketM] = true
	}

	// Also launch 4 concurrent requests simultaneously to verify concurrent batch packing.
	const numConcurrent = 4
	var wg sync.WaitGroup
	concResps := make([]*CompletionResponse, numConcurrent)
	for i := range numConcurrent {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			concResps[idx] = postJSON(t, ts.Client(), ts.URL+"/v1/chat/completions", map[string]any{
				"model": "egraph-llama-3",
				"messages": []ChatMessage{
					{Role: "system", Content: "Shared concurrent batch system prompt"},
					{Role: "user", Content: "Concurrent decode request"},
				},
				"max_tokens": 8,
			})
		}(i)
	}
	wg.Wait()
	for i, r := range concResps {
		if r == nil {
			t.Fatalf("concurrent response %d is nil", i)
		}
		observedBuckets[r.EGraphFlight.ActiveBucketM] = true
	}

	for _, wantM := range []int{1, 2, 4, 8} {
		if !observedBuckets[wantM] {
			t.Fatalf("bucket M=%d was never observed in %v", wantM, observedBuckets)
		}
	}

	// Verify GET /v1/flight/trace.json returns valid Perfetto JSON with traceEvents
	// including bucket transitions and stage spans.
	traceResp, err := ts.Client().Get(ts.URL + "/v1/flight/trace.json")
	if err != nil {
		t.Fatalf("GET /v1/flight/trace.json: %v", err)
	}
	defer traceResp.Body.Close()
	if traceResp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/flight/trace.json status = %d, want 200", traceResp.StatusCode)
	}
	var traceDoc autotune.ChromeTraceDocument
	if err := json.NewDecoder(traceResp.Body).Decode(&traceDoc); err != nil {
		t.Fatalf("decode trace.json: %v", err)
	}
	if len(traceDoc.TraceEvents) == 0 {
		t.Fatal("expected non-empty traceEvents in /v1/flight/trace.json")
	}
	var bucketTransitions, stageSpans int
	for _, ev := range traceDoc.TraceEvents {
		switch ev.Cat {
		case "specdecode.bucket":
			bucketTransitions++
		case "megakernel.stage":
			stageSpans++
		}
	}
	if bucketTransitions == 0 {
		t.Fatalf("expected recorded specdecode.bucket transition events in traceEvents, got 0")
	}
	if stageSpans == 0 {
		t.Fatalf("expected recorded megakernel.stage span events in traceEvents, got 0")
	}

	// Verify GET /v1/pgo/status reports barrier_safety_verified: true, bucket summaries,
	// Radix KV cache stats, and disaggregated P/D pool ratios.
	statusResp, err := ts.Client().Get(ts.URL + "/v1/pgo/status")
	if err != nil {
		t.Fatalf("GET /v1/pgo/status: %v", err)
	}
	defer statusResp.Body.Close()
	if statusResp.StatusCode != http.StatusOK {
		t.Fatalf("GET /v1/pgo/status status = %d, want 200", statusResp.StatusCode)
	}
	var status PGOStatusResponse
	if err := json.NewDecoder(statusResp.Body).Decode(&status); err != nil {
		t.Fatalf("decode /v1/pgo/status: %v", err)
	}
	if !status.BarrierSafetyVerified {
		t.Fatal("expected status.BarrierSafetyVerified == true")
	}
	if len(status.Buckets) != 4 {
		t.Fatalf("len(status.Buckets) = %d, want 4", len(status.Buckets))
	}
	for _, b := range status.Buckets {
		if !slices.Contains([]int{1, 2, 4, 8}, b.BucketM) {
			t.Fatalf("unexpected bucket M=%d in status", b.BucketM)
		}
		if !b.BarrierSafetyVerified {
			t.Fatalf("bucket M=%d BarrierSafetyVerified = false", b.BucketM)
		}
	}
	if status.RadixKVCache.HitCount <= 0 || status.RadixKVCache.MissCount <= 0 {
		t.Fatalf("expected positive RadixKVCache HitCount and MissCount, got %+v", status.RadixKVCache)
	}
	if status.PrefillGPUs <= 0 || status.DecodeGPUs <= 0 || status.PDPool.PDRatio <= 0 {
		t.Fatalf("expected positive PrefillGPUs/DecodeGPUs in status, got prefill=%d decode=%d pool=%+v",
			status.PrefillGPUs, status.DecodeGPUs, status.PDPool)
	}
}

func TestEGServeSelfTestCLIFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run([]string{
		"-arch", "sm_121a",
		"-hidden-dim", "256",
		"-interm-dim", "512",
		"-layers", "4",
		"-pgo-cache-dir", t.TempDir(),
		"-self-test",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run(-self-test) exited with %d; stderr:\n%s", code, stderr.String())
	}

	var summary SelfTestSummary
	if err := json.Unmarshal(stdout.Bytes(), &summary); err != nil {
		t.Fatalf("unmarshal -self-test JSON summary: %v\nraw:\n%s", err, stdout.String())
	}
	if !summary.BarrierSafetyVerified || !summary.PGOStatus.BarrierSafetyVerified {
		t.Fatalf("expected BarrierSafetyVerified=true in self-test summary: %+v", summary)
	}
	if summary.Turn2PrefixHitTokens <= summary.Turn1PrefixHitTokens {
		t.Fatalf("expected Turn2PrefixHitTokens (%d) > Turn1PrefixHitTokens (%d)",
			summary.Turn2PrefixHitTokens, summary.Turn1PrefixHitTokens)
	}
	wantBuckets := []int{1, 2, 4, 8}
	if !slices.Equal(summary.ObservedBuckets, wantBuckets) {
		t.Fatalf("ObservedBuckets = %v, want %v", summary.ObservedBuckets, wantBuckets)
	}
	if summary.TraceEventCount <= 0 {
		t.Fatalf("expected positive TraceEventCount, got %d", summary.TraceEventCount)
	}
}
