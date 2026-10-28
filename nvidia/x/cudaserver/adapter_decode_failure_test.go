package cudaserver_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudaserver"
)

type decodeFailureTokenizer struct{}

func (decodeFailureTokenizer) Encode(string) ([]int32, error) { return []int32{7}, nil }
func (decodeFailureTokenizer) Decode(tokens []int32) (string, error) {
	return fmt.Sprint(tokens[0]), nil
}

type decodeFailureEngine struct {
	requestID string
	calls     int
}

func (e *decodeFailureEngine) ExecuteForward(_ context.Context, step *cudabatch.ExecutionStep) (map[string]int32, map[string]error, error) {
	e.calls++
	out := make(map[string]int32)
	for _, row := range step.Metadata.RowBindings {
		e.requestID = row.SeqID
		out[row.SeqID] = 8
	}
	return out, nil, nil
}

func TestAdapterDecodeExhaustion(t *testing.T) {
	for _, blocks := range []int{1, 2} {
		t.Run(fmt.Sprint(blocks), func(t *testing.T) {
			pool := cudakv.NewBlockAllocator(1, blocks)
			scheduler := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 1, ChunkSize: 1}, cudabatch.NewMemoryBudgetTracker(pool))
			engine := new(decodeFailureEngine)
			adapter := cudaserver.NewBatchEngineAdapter(scheduler, engine, decodeFailureTokenizer{})
			run := func(maxTokens int) error {
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				done := make(chan error, 1)
				go func() {
					done <- adapter.Generate(ctx, cudaserver.ChatCompletionRequest{MaxTokens: maxTokens}, make(chan string, 4))
				}()
				ticker := time.NewTicker(time.Millisecond)
				defer ticker.Stop()
				for {
					select {
					case err := <-done:
						return err
					case <-ticker.C:
						if err := adapter.StepIteration(ctx); err != nil {
							cancel()
							<-done
							return err
						}
					}
				}
			}
			err := run(2)
			if blocks == 1 {
				status, ok := scheduler.Request(engine.requestID)
				if !ok || !status.IsFinished() || status.Err() == nil || !errors.Is(err, status.Err()) || !strings.Contains(err.Error(), "out of KV blocks") {
					t.Fatalf("Generate did not expose terminal cause: err=%v status=%+v", err, status)
				}
				if engine.calls != 1 {
					t.Fatalf("failed growth executed: calls=%d", engine.calls)
				}
			} else if err != nil || engine.calls != 2 {
				t.Fatalf("capacity positive: err=%v calls=%d", err, engine.calls)
			}
			if pool.NumFreeBlocks() != blocks {
				t.Fatalf("free=%d", pool.NumFreeBlocks())
			}
			if err := run(1); err != nil {
				t.Fatalf("unrelated request failed: %v", err)
			}
			if pool.NumFreeBlocks() != blocks {
				t.Fatalf("progress leaked KV: free=%d", pool.NumFreeBlocks())
			}
		})
	}
}
