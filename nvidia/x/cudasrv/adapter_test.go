package cudasrv_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudasrv"
)

type simpleWordTokenizer struct{}

func (t *simpleWordTokenizer) Encode(text string) ([]int32, error) {
	words := strings.Fields(text)
	var tokens []int32
	for _, w := range words {
		var h int32
		for _, b := range []byte(w) {
			h = h*31 + int32(b)
		}
		if h < 0 {
			h = -h
		}
		tokens = append(tokens, (h%10000)+1)
	}
	return tokens, nil
}

func (t *simpleWordTokenizer) Decode(tokens []int32) (string, error) {
	var parts []string
	for _, tok := range tokens {
		parts = append(parts, fmt.Sprintf("word%d", tok))
	}
	return strings.Join(parts, " "), nil
}

func TestBatchEngineAdapterIntegration(t *testing.T) {
	alloc := cudakv.NewBlockAllocator(16, 20)
	tracker := cudabatch.NewMemoryBudgetTracker(alloc)
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 64,
		ChunkSize:       16,
	}, tracker)

	engine := cudabatch.NewHostDeterministicEngine(0, []int32{101, 102})
	tok := &simpleWordTokenizer{}

	adapter := cudasrv.NewBatchEngineAdapter(sched, engine, tok)

	server := cudasrv.New(adapter, "test-model")

	// Drive the adapter step loop in background
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				_ = adapter.StepIteration(ctx)
				time.Sleep(1 * time.Millisecond)
			}
		}
	}()

	payload := `{"model":"test-model","messages":[{"role":"user","content":"hello world"}],"max_tokens":2}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(payload))
	w := httptest.NewRecorder()

	server.Handler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	var resp cudasrv.ChatCompletionResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least 1 choice in response")
	}
	if resp.Choices[0].Message.Content == "" {
		t.Fatal("expected generated content in response")
	}
}

func TestProtocolContinuationTracking(t *testing.T) {
	alloc := cudakv.NewBlockAllocator(16, 30)
	tracker := cudabatch.NewMemoryBudgetTracker(alloc)
	sched := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{
		MaxBatchSize:    4,
		MaxBudgetTokens: 64,
		ChunkSize:       16,
	}, tracker)

	engine := cudabatch.NewHostDeterministicEngine(0, []int32{201, 202})
	tok := &simpleWordTokenizer{}

	adapter := cudasrv.NewBatchEngineAdapter(sched, engine, tok)
	server := cudasrv.New(adapter, "test-model")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				_ = adapter.StepIteration(ctx)
				time.Sleep(1 * time.Millisecond)
			}
		}
	}()

	// Turn 1 with thinking visible and session ID
	turn1Payload := `{"model":"test-model","session_id":"sess-123","thinking_visible":true,"messages":[{"role":"user","content":"calculate something"}],"max_tokens":2}`
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(turn1Payload))
	w1 := httptest.NewRecorder()
	server.Handler().ServeHTTP(w1, req1)

	if w1.Code != http.StatusOK {
		t.Fatalf("turn 1 failed: %d (%s)", w1.Code, w1.Body.String())
	}

	// Verify session continuation was saved in tracker
	sess, ok := adapter.ContinuationTracker().GetSession("sess-123")
	if !ok {
		t.Fatal("session was not saved in ContinuationTracker")
	}
	if sess.Visibility != cudabatch.VisibilityThinkingVisible {
		t.Fatalf("session visibility = %v, want ThinkingVisible", sess.Visibility)
	}
	if len(sess.Tokens) == 0 {
		t.Fatal("session tokens empty")
	}

	// Turn 2 continuation with the same session
	turn2Payload := `{"model":"test-model","session_id":"sess-123","thinking_visible":true,"messages":[{"role":"user","content":"calculate something"},{"role":"assistant","content":"done"},{"role":"user","content":"continue"}],"max_tokens":2}`
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(turn2Payload))
	w2 := httptest.NewRecorder()
	server.Handler().ServeHTTP(w2, req2)

	if w2.Code != http.StatusOK {
		t.Fatalf("turn 2 failed: %d (%s)", w2.Code, w2.Body.String())
	}

	sess2, ok := adapter.ContinuationTracker().GetSession("sess-123")
	if !ok {
		t.Fatal("session not preserved after turn 2")
	}
	if len(sess2.Tokens) <= len(sess.Tokens) {
		t.Fatalf("tokens not accumulated across turns: turn 1=%d, turn 2=%d", len(sess.Tokens), len(sess2.Tokens))
	}
}

func TestExactSuffixTokenization(t *testing.T) {
	tok := &simpleWordTokenizer{}

	// Case 1: Prefix match without boundary collision
	existingText := "the quick brown"
	existingTokens, _ := tok.Encode(existingText)

	reused, delta, err := cudabatch.SpliceContinuation(tok, existingTokens, existingText, " fox jumps", 2)
	if err != nil {
		t.Fatal(err)
	}

	if len(reused) != 3 {
		t.Fatalf("reused tokens = %d, want 3", len(reused))
	}
	if len(delta) != 2 {
		t.Fatalf("delta tokens = %d, want 2", len(delta))
	}

	// Case 2: Matching continuation blocks
	blocks := []int{10, 11, 12, 13}
	matchedBlocks, tokenCount := cudabatch.MatchContinuationBlocks(32, blocks, 16)
	if len(matchedBlocks) != 2 || tokenCount != 32 {
		t.Fatalf("MatchContinuationBlocks = %v (%d tokens), want 2 blocks (32 tokens)", matchedBlocks, tokenCount)
	}
}
