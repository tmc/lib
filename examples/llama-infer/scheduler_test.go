package main

import (
	"context"
	"testing"
	"time"
)

func TestSequenceLifecycle(t *testing.T) {
	promptTokens := []int32{1, 2, 3}
	seq := NewSequence("seq-1", "test prompt", promptTokens, 3, nil)

	if seq.Status != SeqPendingPrefill {
		t.Errorf("expected initial status PendingPrefill, got %v", seq.Status)
	}

	cont := seq.AppendToken(10, " tok1")
	if !cont {
		t.Errorf("expected sequence to continue after token 1")
	}

	cont = seq.AppendToken(20, " tok2")
	if !cont {
		t.Errorf("expected sequence to continue after token 2")
	}

	cont = seq.AppendToken(30, " tok3")
	if cont {
		t.Errorf("expected sequence to finish after reaching max tokens")
	}

	if seq.Status != SeqFinished {
		t.Errorf("expected status Finished, got %v", seq.Status)
	}

	select {
	case <-seq.DoneCh:
	case <-time.After(50 * time.Millisecond):
		t.Errorf("expected DoneCh to be closed")
	}
}

func TestSchedulerCapacityAndStep(t *testing.T) {
	sch := NewScheduler(nil, Config{}, 2)

	seq1 := NewSequence("seq-1", "prompt 1", []int32{1}, 5, nil)
	seq2 := NewSequence("seq-2", "prompt 2", []int32{2}, 5, nil)
	seq3 := NewSequence("seq-3", "prompt 3", []int32{3}, 5, nil)

	if err := sch.Submit(seq1); err != nil {
		t.Fatalf("unexpected error submitting seq1: %v", err)
	}
	if err := sch.Submit(seq2); err != nil {
		t.Fatalf("unexpected error submitting seq2: %v", err)
	}
	if err := sch.Submit(seq3); err == nil {
		t.Errorf("expected capacity error when submitting beyond maxBatch")
	}

	if sch.ActiveCount() != 2 {
		t.Errorf("expected active count 2, got %d", sch.ActiveCount())
	}

	ctx := context.Background()
	_ = sch.Step(ctx)
}
