package cudabatch_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabatch"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func prioritySubmit(t *testing.T, s *cudabatch.BatchScheduler, id string, prompt, generated int) {
	t.Helper()
	if err := s.Submit(&cudabatch.Request{ID: id, PromptTokens: make([]int32, prompt), MaxTokens: generated}); err != nil {
		t.Fatal(err)
	}
}

func prioritySchedule(t *testing.T, s *cudabatch.BatchScheduler, rows, budget int) *cudabatch.ScheduledIteration {
	t.Helper()
	iter, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	if len(iter.AllocatedRows) > rows || iter.TotalTokens > budget {
		t.Fatalf("iteration exceeds limits: rows=%d tokens=%d", len(iter.AllocatedRows), iter.TotalTokens)
	}
	return iter
}

func priorityComplete(t *testing.T, s *cudabatch.BatchScheduler, iter *cudabatch.ScheduledIteration) {
	t.Helper()
	tokens := make(map[string]int32)
	for _, row := range iter.AllocatedRows {
		if !row.IsPrefill || row.IsLastPrefill {
			tokens[row.SeqID] = 42
		}
	}
	if err := s.CompleteStep(iter, tokens, nil); err != nil {
		t.Fatal(err)
	}
}

func prioritySeed(t *testing.T, s *cudabatch.BatchScheduler, decodes, rows, budget int) {
	t.Helper()
	for i := 0; i < decodes; i++ {
		prioritySubmit(t, s, fmt.Sprintf("d%d", i), 1, 100)
	}
	for i := 0; i < decodes; i++ {
		priorityComplete(t, s, prioritySchedule(t, s, rows, budget))
		ready := true
		for j := 0; j < decodes; j++ {
			r, _ := s.Request(fmt.Sprintf("d%d", j))
			ready = ready && r.State == cudabatch.StateDecoding
		}
		if ready {
			return
		}
	}
	t.Fatal("seed requests did not enter decode")
}

func TestPriorityDecodePrefillProgress(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		rows, budget, decodes int
		pool                  bool
	}{
		{"spare row", 3, 2, 2, false},
		{"full rows", 2, 2, 2, false},
		{"decode priority", 4, 4, 3, false},
		{"one token", 2, 1, 1, false},
		{"one row", 1, 2, 1, false},
		{"allocated KV", 2, 2, 2, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var tracker *cudabatch.MemoryBudgetTracker
			var pool *cudakv.BlockAllocator
			if tc.pool {
				pool = cudakv.NewBlockAllocator(16, tc.decodes+1)
				tracker = cudabatch.NewMemoryBudgetTracker(pool)
			}
			s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: tc.rows, MaxBudgetTokens: tc.budget, ChunkSize: 3, FairnessPolicy: cudabatch.PolicyPriorityDecode}, tracker)
			prioritySeed(t, s, tc.decodes, tc.rows, tc.budget)
			prioritySubmit(t, s, "long", 4, 1)
			wantDecodes := min(tc.decodes, tc.rows-1, tc.budget-1)
			for round := 0; round < 4; round++ {
				iter := prioritySchedule(t, s, tc.rows, tc.budget)
				if len(iter.PrefillChunks) != 1 || iter.PrefillChunks[0].SeqID != "long" || len(iter.PrefillChunks[0].Tokens) == 0 {
					t.Fatalf("round %d: feasible prefill made no progress: %+v", round, iter)
				}
				if len(iter.DecodeSeqIDs) != wantDecodes {
					t.Fatalf("round %d: decodes=%d want %d", round, len(iter.DecodeSeqIDs), wantDecodes)
				}
				priorityComplete(t, s, iter)
				r, _ := s.Request("long")
				if r.State == cudabatch.StateCompleted {
					break
				}
			}
			r, _ := s.Request("long")
			if r.State != cudabatch.StateCompleted || r.PrefilledTokens != 4 {
				t.Fatalf("long prefill did not complete: %+v", r)
			}
			iter := prioritySchedule(t, s, tc.rows, tc.budget)
			if len(iter.DecodeSeqIDs) != min(tc.decodes, tc.rows, tc.budget) || len(iter.PrefillChunks) != 0 {
				t.Fatalf("decode budget stayed reserved after prefill completed: %+v", iter)
			}
			priorityComplete(t, s, iter)
			for i := 0; i < tc.decodes; i++ {
				if err := s.Cancel(fmt.Sprintf("d%d", i)); err != nil {
					t.Fatal(err)
				}
			}
			if pool != nil && pool.NumFreeBlocks() != tc.decodes+1 {
				t.Fatal("KV blocks not released after completed steps and cancellation")
			}
		})
	}
}

func TestPriorityDecodeKVUnavailable(t *testing.T) {
	pool := cudakv.NewBlockAllocator(16, 2)
	s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 2, MaxBudgetTokens: 2, ChunkSize: 1, FairnessPolicy: cudabatch.PolicyPriorityDecode}, cudabatch.NewMemoryBudgetTracker(pool))
	prioritySeed(t, s, 2, 2, 2)
	prioritySubmit(t, s, "long", 4, 1)
	for round := 0; round < 3; round++ {
		iter := prioritySchedule(t, s, 2, 2)
		r, _ := s.Request("long")
		if len(iter.PrefillChunks) != 0 || len(iter.DecodeSeqIDs) != 2 || iter.TotalTokens != 2 || r.State != cudabatch.StateWaiting || pool.NumFreeBlocks() != 0 {
			t.Fatalf("infeasible KV admission changed decode progress: %+v request=%+v", iter, r)
		}
		priorityComplete(t, s, iter)
	}
	for _, id := range []string{"d0", "d1", "long"} {
		if err := s.Cancel(id); err != nil {
			t.Fatal(err)
		}
	}
	if pool.NumFreeBlocks() != 2 {
		t.Fatal("KV cleanup lost blocks")
	}
}

func TestPriorityDecodeAdmissionBound(t *testing.T) {
	s := cudabatch.NewBatchScheduler(cudabatch.SchedulerConfig{MaxBatchSize: 2, MaxBudgetTokens: 2, ChunkSize: 1, FairnessPolicy: cudabatch.PolicyPriorityDecode}, nil)
	prioritySeed(t, s, 2, 2, 2)
	prioritySubmit(t, s, "first", 8, 1)
	prioritySubmit(t, s, "second", 8, 1)
	for round := 0; round < 3; round++ {
		iter := prioritySchedule(t, s, 2, 2)
		r, _ := s.Request("second")
		if len(iter.PrefillChunks) != 1 || iter.PrefillChunks[0].SeqID != "first" || r.State != cudabatch.StateWaiting {
			t.Fatalf("extra admission was not bounded: %+v second=%+v", iter, r)
		}
		priorityComplete(t, s, iter)
	}
	if err := s.Cancel("first"); err != nil {
		t.Fatal(err)
	}
	iter := prioritySchedule(t, s, 2, 2)
	if len(iter.PrefillChunks) != 1 || iter.PrefillChunks[0].SeqID != "second" {
		t.Fatalf("freed admission slot did not enable next prefill: %+v", iter)
	}
	priorityComplete(t, s, iter)
	for _, id := range []string{"d0", "d1", "second"} {
		if err := s.Cancel(id); err != nil {
			t.Fatal(err)
		}
	}
}
