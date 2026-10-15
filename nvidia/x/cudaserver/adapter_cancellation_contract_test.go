package cudaserver

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

type cancellationEngineFunc func(context.Context, *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error)

func (f cancellationEngineFunc) ExecuteForward(ctx context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	return f(ctx, step)
}

type cancellationTokenizer struct {
	mu      sync.Mutex
	decoded []int32
}

func (*cancellationTokenizer) Encode(prompt string) ([]int32, error) {
	if strings.Contains(prompt, "cancel-this") {
		return []int32{1}, nil
	}
	return []int32{2}, nil
}

func (t *cancellationTokenizer) Decode(tokens []int32) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.decoded = append(t.decoded, tokens...)
	return fmt.Sprint(tokens[0]), nil
}

// Listener registration is inspected only as a deterministic admission barrier.
// Execution, cancellation, generation, and terminal settlement all use the real
// public adapter/scheduler interfaces; this test opens no transport or socket.
func awaitCancellationRequests(t *testing.T, ctx context.Context, adapter *BatchEngineAdapter) []string {
	t.Helper()
	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()
	for {
		adapter.mu.Lock()
		var ids []string
		for id := range adapter.listeners {
			ids = append(ids, id)
		}
		adapter.mu.Unlock()
		if len(ids) == 2 {
			ready := true
			for _, id := range ids {
				_, ok := adapter.scheduler.Request(id)
				ready = ready && ok
			}
			if ready {
				return ids
			}
		}
		select {
		case <-ticker.C:
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
}

func TestAdapterCancellationContract(t *testing.T) {
	for _, phase := range []string{"before-schedule", "final-prefill", "decode", "row-error"} {
		t.Run(phase, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			allocator := cudakv.NewBlockAllocator(16, 8)
			scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 2, MaxBudgetTokens: 4, ChunkSize: 2}, cudabatch.NewMemoryBudgetTracker(allocator))
			tokenizer := &cancellationTokenizer{}
			calls, releases := 0, 0
			var canceledID string
			rowFailure := errors.New("row execution failed")
			engine := cancellationEngineFunc(func(_ context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
				calls++
				out := make(map[string]int32)
				rowErrors := make(map[string]error)
				for _, row := range step.Metadata.RowBindings {
					request, _ := scheduler.Request(row.SeqID)
					bad := request.PromptTokens[0] == 1
					if bad {
						canceledID = row.SeqID
					}
					if bad && (phase != "decode" || !row.IsPrefill) {
						if err := scheduler.Cancel(row.SeqID); err != nil {
							t.Fatal(err)
						}
						if allocator.NumFreeBlocks() != 6 || releases != 0 {
							t.Fatal("submitted cancellation released ownership")
						}
						if phase == "row-error" {
							rowErrors[row.SeqID] = rowFailure
						}
					}
					if !row.IsPrefill || row.IsLastPrefill {
						if bad {
							out[row.SeqID] = int32(100 + calls)
						} else {
							out[row.SeqID] = int32(200 + calls)
						}
					}
				}
				return out, rowErrors, nil
			})
			adapter := NewBatchEngineAdapter(scheduler, engine, tokenizer)
			badCtx, cancelBad := context.WithCancel(ctx)
			defer cancelBad()
			badTokens, healthyTokens := make(chan string, 4), make(chan string, 4)
			badDone, healthyDone := make(chan error, 1), make(chan error, 1)
			maxTokens := 1
			if phase == "decode" {
				maxTokens = 2
			}
			go func() {
				badDone <- adapter.Generate(badCtx, ChatCompletionRequest{Messages: []ChatMessage{{Role: "user", Content: "cancel-this"}}, MaxTokens: maxTokens}, badTokens)
			}()
			go func() {
				healthyDone <- adapter.Generate(ctx, ChatCompletionRequest{Messages: []ChatMessage{{Role: "user", Content: "healthy"}}, MaxTokens: maxTokens}, healthyTokens)
			}()
			ids := awaitCancellationRequests(t, ctx, adapter)
			for _, id := range ids {
				request, _ := scheduler.Request(id)
				if request.PromptTokens[0] == 1 {
					canceledID = id
					scheduler.RegisterResourceRelease(id, func() { releases++ })
				}
			}
			if phase == "before-schedule" {
				cancelBad()
				select {
				case err := <-badDone:
					if !errors.Is(err, context.Canceled) {
						t.Fatal(err)
					}
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				}
			}
			if err := adapter.StepIteration(ctx); err != nil {
				t.Fatal(err)
			}
			if phase == "decode" {
				// Both final-prefill samples must be visible before canceling
				// the next submitted decode; preserve that completed output.
				select {
				case token := <-badTokens:
					if token != "101" {
						t.Fatal(token)
					}
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				}
				if err := adapter.StepIteration(ctx); err != nil {
					t.Fatal(err)
				}
			}
			if phase != "before-schedule" {
				select {
				case err := <-badDone:
					want := error(context.Canceled)
					if phase == "row-error" {
						want = rowFailure
					}
					if !errors.Is(err, want) {
						t.Fatalf("canceled/error row became success: %v", err)
					}
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				}
			}
			select {
			case err := <-healthyDone:
				if err != nil {
					t.Fatal(err)
				}
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			if len(badTokens) != 0 || len(healthyTokens) != maxTokens || releases != 1 || allocator.NumFreeBlocks() != 8 {
				t.Fatalf("emission/ownership: bad=%d healthy=%d releases=%d free=%d", len(badTokens), len(healthyTokens), releases, allocator.NumFreeBlocks())
			}
			for i := range maxTokens {
				if token := <-healthyTokens; token != fmt.Sprint(201+i) {
					t.Fatalf("healthy sibling token %d=%q", i, token)
				}
			}
			request, _ := scheduler.Request(canceledID)
			wantConsumed := 0
			if phase == "decode" {
				wantConsumed = 1
			}
			if request.ContextTokens() != wantConsumed {
				t.Fatalf("canceled work advanced input: %+v", request)
			}
			tokenizer.mu.Lock()
			for _, token := range tokenizer.decoded {
				if token < 200 && !(phase == "decode" && token == 101) {
					t.Errorf("discarded canceled/error token decoded for listener: %d", token)
				}
			}
			tokenizer.mu.Unlock()
			before := calls
			if err := adapter.StepIteration(ctx); err != nil || calls != before || len(badTokens) != 0 || len(healthyTokens) != 0 || releases != 1 {
				t.Fatalf("duplicate execution/emission/settlement: calls=%d err=%v", calls, err)
			}
		})
	}
}
