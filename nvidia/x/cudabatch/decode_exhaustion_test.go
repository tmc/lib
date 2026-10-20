package cudabatch_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func TestDecodeExhaustion(t *testing.T) {
	for _, blocks := range []int{1, 2} {
		t.Run(string(rune('0'+blocks)), func(t *testing.T) {
			pool := cudakv.NewBlockAllocator(1, blocks)
			s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 1, ChunkSize: 1}, cudabatch.NewMemoryBudgetTracker(pool))
			var finished *cudabatch.Request
			finishes, releases := 0, 0
			if err := s.Submit(&cudabatch.Request{ID: "a", PromptTokens: []int32{7}, MaxTokens: 2, OnFinish: func(r *cudabatch.Request) { finished = r; finishes++ }}); err != nil {
				t.Fatal(err)
			}
			s.RegisterResourceRelease("a", func() { releases++ })
			first, err := s.Schedule()
			if err != nil {
				t.Fatal(err)
			}
			if _, err := s.Schedule(); !errors.Is(err, cudabatch.ErrExecutionInFlight) || releases != 0 || pool.NumFreeBlocks() != blocks-1 {
				t.Fatalf("outstanding prefill released: %v", err)
			}
			if err := s.CompleteStep(first, map[string]int32{"a": 8}, nil); err != nil {
				t.Fatal(err)
			}
			step, err := s.Schedule()
			if err != nil {
				t.Fatal(err)
			}
			if blocks == 1 {
				status, _ := s.Request("a")
				if len(step.AllocatedRows) != 0 || !status.IsFinished() || status.State != cudabatch.StateCompleted || status.Err() == nil || !strings.Contains(status.Err().Error(), "out of KV blocks") {
					t.Fatalf("exhausted decode not terminal: rows=%d status=%+v err=%v", len(step.AllocatedRows), status, status.Err())
				}
				if finished == nil || finished.Err() != status.Err() || finishes != 1 || releases != 1 || pool.NumFreeBlocks() != blocks {
					t.Fatalf("failure notification/cleanup: finished=%+v finishes=%d releases=%d free=%d", finished, finishes, releases, pool.NumFreeBlocks())
				}
				if err := s.Cancel("a"); err != nil {
					t.Fatal(err)
				}
				if _, err := s.Schedule(); err != nil || finishes != 1 || releases != 1 {
					t.Fatalf("repeated cleanup: %v", err)
				}
			} else {
				if len(step.AllocatedRows) != 1 || releases != 0 || finished != nil {
					t.Fatalf("capacity control failed: %+v", step)
				}
				if err := s.CompleteStep(step, map[string]int32{"a": 9}, nil); err != nil {
					t.Fatal(err)
				}
				if finished == nil || finished.Err() != nil || finishes != 1 || releases != 1 || pool.NumFreeBlocks() != blocks {
					t.Fatalf("successful completion: %+v", finished)
				}
			}
		})
	}
}

func TestDecodeExhaustionQueuedProgress(t *testing.T) {
	pool := cudakv.NewBlockAllocator(1, 1)
	s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 1, ChunkSize: 1}, cudabatch.NewMemoryBudgetTracker(pool))
	for _, req := range []*cudabatch.Request{{ID: "a", PromptTokens: []int32{7}, MaxTokens: 2}, {ID: "b", PromptTokens: []int32{3}, MaxTokens: 1}} {
		if err := s.Submit(req); err != nil {
			t.Fatal(err)
		}
	}
	first, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if err := s.CompleteStep(first, map[string]int32{"a": 8}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Schedule(); err != nil {
		t.Fatal(err)
	}
	next, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if len(next.AllocatedRows) != 1 || next.AllocatedRows[0].SeqID != "b" {
		t.Fatalf("queued request did not progress: %+v", next)
	}
	if err := s.CompleteStep(next, map[string]int32{"b": 4}, nil); err != nil {
		t.Fatal(err)
	}
	if pool.NumFreeBlocks() != 1 {
		t.Fatalf("free=%d", pool.NumFreeBlocks())
	}
}

func TestDecodeExhaustionRetainsBorrowedBlock(t *testing.T) {
	pool := cudakv.NewBlockAllocator(1, 1)
	s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 1, MaxBudgetTokens: 1, ChunkSize: 1}, cudabatch.NewMemoryBudgetTracker(pool))
	if err := s.Submit(&cudabatch.Request{ID: "a", PromptTokens: []int32{7}, MaxTokens: 2}); err != nil {
		t.Fatal(err)
	}
	first, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	borrowed := first.AllocatedRows[0].BlockTable
	if err := pool.Retain(borrowed); err != nil {
		t.Fatal(err)
	}
	if err := s.CompleteStep(first, map[string]int32{"a": 8}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Schedule(); err != nil {
		t.Fatal(err)
	}
	status, _ := s.Request("a")
	holders, err := pool.Holders(borrowed[0])
	if err != nil || !status.IsFinished() || status.Err() == nil || holders != 1 || pool.NumFreeBlocks() != 0 {
		t.Fatalf("borrowed block released: status=%+v holders=%d err=%v", status, holders, err)
	}
	if err := pool.Release(borrowed); err != nil {
		t.Fatal(err)
	}
	if pool.NumFreeBlocks() != 1 {
		t.Fatal("last borrower did not free block")
	}
}
