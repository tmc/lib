package cudabatch

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func TestAllocatedRowsAdmission(t *testing.T) {
	pool := cudakv.NewBlockAllocator(2, 8)
	if err := pool.AllocSequence(11, nil, 3); err != nil {
		t.Fatal(err)
	}
	if err := pool.AllocSequence(12, nil, 3); err != nil {
		t.Fatal(err)
	}
	row := BatchRowBinding{SeqID: "a", IsPrefill: true, Tokens: []int32{1, 2, 3}, SequenceLength: 3, BlockTable: pool.Table(11)}
	bucket := ShapeBucket{1, 4, 3}
	for _, name := range []string{"valid", "nil-pool", "wrong-owner", "out-of-pool", "foreign-held", "duplicate", "capacity"} {
		t.Run(name, func(t *testing.T) {
			r := row
			r.BlockTable = append([]int(nil), row.BlockTable...)
			p := pool
			id := int64(11)
			switch name {
			case "nil-pool":
				p = nil
			case "wrong-owner":
				id = 99
			case "out-of-pool":
				r.BlockTable[0] = pool.TotalBlocks()
			case "foreign-held":
				r.BlockTable = pool.Table(12)
			case "duplicate":
				r.BlockTable[1] = r.BlockTable[0]
			case "capacity":
				r.SequenceLength = 4
				r.Tokens = append(r.Tokens, 4)
			}
			free := pool.NumFreeBlocks()
			m, err := BindAllocatedRows([]BatchRowBinding{r}, bucket, p, []int64{id})
			if name == "valid" {
				if err != nil || m == nil {
					t.Fatal(err)
				}
			} else if m != nil || !errors.Is(err, ErrCorruptBlockTable) {
				t.Fatalf("admitted %s: %v %v", name, m, err)
			}
			if pool.NumFreeBlocks() != free {
				t.Fatal("binding changed allocation")
			}
		})
	}
}

func TestPlanEngineAllocationBeforeBinder(t *testing.T) {
	for _, name := range []string{"shape-only", "changed-table", "freed", "reallocated", "valid"} {
		t.Run(name, func(t *testing.T) {
			pool := cudakv.NewBlockAllocator(1, 1)
			if err := pool.AllocSequence(1, nil, 1); err != nil {
				t.Fatal(err)
			}
			row := BatchRowBinding{SeqID: "a", Tokens: []int32{1}, SequenceLength: 1, BlockTable: pool.Table(1)}
			m, err := BindAllocatedRows([]BatchRowBinding{row}, ShapeBucket{1, 1, 1}, pool, []int64{1})
			if err != nil {
				t.Fatal(err)
			}
			switch name {
			case "shape-only":
				m, err = ValidateAndBindRows([]BatchRowBinding{row}, ShapeBucket{1, 1, 1}, 1)
				if err != nil {
					t.Fatal(err)
				}
			case "changed-table":
				m.RowBindings[0].BlockTable[0] = 1
			case "freed":
				pool.FreeSequence(1)
			case "reallocated":
				pool.FreeSequence(1)
				if err := pool.AllocSequence(1, nil, 1); err != nil {
					t.Fatal(err)
				}
			}
			var events []string
			e := planTestEngine(&controlledPlanCompletion{done: true}, nil, &events)
			_, _, err = e.ExecuteForward(context.Background(), &ExecutionStep{StepID: 1, Metadata: m})
			if name == "valid" {
				if err != nil || len(events) == 0 {
					t.Fatal(err, events)
				}
			} else if !errors.Is(err, ErrCorruptBlockTable) || len(events) != 0 {
				t.Fatalf("side effects before allocation admission: %v %v", err, events)
			}
		})
	}
}

func TestSchedulerBindingTracksOutstandingOwner(t *testing.T) {
	pool := cudakv.NewBlockAllocator(2, 8)
	s := NewBatchScheduler(SchedulerConfig{}, NewMemoryBudgetTracker(pool))
	if err := s.Submit(&Request{ID: "a", PromptTokens: []int32{1, 2}, MaxTokens: 1}); err != nil {
		t.Fatal(err)
	}
	iter, err := s.Schedule()
	if err != nil {
		t.Fatal(err)
	}
	m, err := s.BindIteration(iter, ShapeBucket{1, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	iter.AllocatedRows[0].BlockTable[0] = pool.TotalBlocks()
	if _, err := s.BindIteration(iter, ShapeBucket{1, 2, 2}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatal(err)
	}
	if err := s.Cancel("a"); err != nil {
		t.Fatal(err)
	}
	if err := m.validateAllocation(); err != nil {
		t.Fatal("cancellation released in-flight owner", err)
	}
	if err := s.CompleteStep(iter, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := m.validateAllocation(); !errors.Is(err, ErrCorruptBlockTable) {
		t.Fatal("completed owner still admitted", err)
	}
}

type unownedBlockManager struct{ KVBlockManager }

func TestTrackerRequiresActualPool(t *testing.T) {
	target := NewMemoryBudgetTracker(unownedBlockManager{})
	if target.CanAdmit(1, nil) {
		t.Fatal("unknown pool admitted")
	}
	if _, err := target.AllocSequence(1, nil, 1); !errors.Is(err, ErrCorruptBlockTable) {
		t.Fatal(err)
	}
	if err := target.AppendToken(1); !errors.Is(err, ErrCorruptBlockTable) {
		t.Fatal(err)
	}
}

func ExampleBindAllocatedRows() {
	pool := cudakv.NewBlockAllocator(2, 1)
	_ = pool.AllocSequence(7, nil, 1)
	row := BatchRowBinding{SeqID: "a", Tokens: []int32{1}, SequenceLength: 1, BlockTable: pool.Table(7)}
	m, err := BindAllocatedRows([]BatchRowBinding{row}, ShapeBucket{1, 2, 1}, pool, []int64{7})
	fmt.Println(m.TotalTokens, err)
	// Output: 1 <nil>
}

func ExampleBatchScheduler_BindIteration() {
	s := NewBatchScheduler(SchedulerConfig{}, NewMemoryBudgetTracker(cudakv.NewBlockAllocator(2, 1)))
	_ = s.Submit(&Request{ID: "a", PromptTokens: []int32{1}, MaxTokens: 1})
	iteration, _ := s.Schedule()
	m, err := s.BindIteration(iteration, ShapeBucket{1, 2, 1})
	fmt.Println(m.TotalTokens, err)
	// Output: 1 <nil>
}
