package cudasrv_test

import (
	"context"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudasrv"
)

func TestAdapterRecomputesSessionPrompt(t *testing.T) {
	for _, blocks := range [][]int{nil, {0}, {999}} {
		scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 4, MaxBudgetTokens: 64, ChunkSize: 16}, cudabatch.NewMemoryBudgetTracker(cudakv.NewBlockAllocator(16, 20)))
		tokenizer := &simpleWordTokenizer{}
		history := cudabatch.NewContinuationTracker()
		history.StoreSession("session", []int32{17, 18}, blocks, cudabatch.VisibilityDefault, "old prompt")
		var prefills []int32
		engine := executionFunc(func(_ context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
			result := make(map[string]int32)
			for _, row := range step.Iteration.AllocatedRows {
				if row.IsPrefill {
					prefills = append(prefills, row.Tokens...)
				} else {
					result[row.SeqID] = 201
				}
			}
			return result, nil, nil
		})
		adapter := cudasrv.NewBatchEngineAdapter(scheduler, engine, tokenizer, cudasrv.WithContinuationTracker(history))
		request := cudasrv.ChatCompletionRequest{SessionID: "session", MaxTokens: 1, Messages: []cudasrv.ChatMessage{{Role: "user", Content: "first"}, {Role: "assistant", Content: "answer"}, {Role: "user", Content: "second"}}}
		expected, err := tokenizer.Encode("user: first\nassistant: answer\nuser: second\nassistant: ")
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		done := make(chan error, 1)
		go func() { done <- adapter.Generate(ctx, request, make(chan string, 4)) }()
	loop:
		for {
			select {
			case err := <-done:
				if err != nil {
					cancel()
					t.Fatal(err)
				}
				break loop
			case <-ctx.Done():
				cancel()
				t.Fatal(ctx.Err())
			default:
				if err := adapter.StepIteration(ctx); err != nil {
					cancel()
					t.Fatal(err)
				}
				runtime.Gosched()
			}
		}
		cancel()
		if !reflect.DeepEqual(prefills, expected) {
			t.Fatalf("blocks %v: prompt=%v want=%v", blocks, prefills, expected)
		}
		saved, _ := history.GetSession("session")
		if len(saved.BlockTable) != 0 {
			t.Fatalf("snapshot claims live KV ownership: %v", saved.BlockTable)
		}
	}
}
