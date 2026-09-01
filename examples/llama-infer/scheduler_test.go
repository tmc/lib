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

	seq.Status = SeqDecoding
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
		t.Errorf("expected 2 active sequences, got %d", sch.ActiveCount())
	}

	ctx := context.Background()
	n, err := sch.Step(ctx)
	if err != nil {
		t.Fatalf("unexpected error stepping scheduler: %v", err)
	}
	if sch.ActiveCount() != 0 {
		t.Errorf("expected sequences without engine to finish on prefill, remaining %d (stepped %d)", sch.ActiveCount(), n)
	}
}

func TestSchedulerCancellation(t *testing.T) {
	sch := NewScheduler(nil, Config{}, 4)

	seq1 := NewSequence("seq-cancel", "prompt", []int32{1, 2}, 10, nil)
	if err := sch.Submit(seq1); err != nil {
		t.Fatalf("submit: %v", err)
	}

	seq1.Cancel()
	if seq1.Status != SeqCanceled {
		t.Errorf("expected status Canceled, got %v", seq1.Status)
	}

	ctx := context.Background()
	_, err := sch.Step(ctx)
	if err != nil {
		t.Fatalf("step: %v", err)
	}

	if sch.ActiveCount() != 0 {
		t.Errorf("expected canceled sequence to be pruned, got %d active", sch.ActiveCount())
	}
}

func TestSchedulerCapacityReuse(t *testing.T) {
	sch := NewScheduler(nil, Config{}, 1)

	seq1 := NewSequence("seq-1", "p1", []int32{1}, 2, nil)
	if err := sch.Submit(seq1); err != nil {
		t.Fatalf("submit seq1: %v", err)
	}

	// Step to finish seq1
	ctx := context.Background()
	_, _ = sch.Step(ctx)

	if sch.ActiveCount() != 0 {
		t.Fatalf("expected 0 active, got %d", sch.ActiveCount())
	}

	// Should now be able to admit seq2
	seq2 := NewSequence("seq-2", "p2", []int32{2}, 2, nil)
	if err := sch.Submit(seq2); err != nil {
		t.Fatalf("expected capacity to be reused for seq2: %v", err)
	}
	if sch.ActiveCount() != 1 {
		t.Errorf("expected 1 active sequence, got %d", sch.ActiveCount())
	}
}

func TestSchedulerEngineStepping(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	sch := NewScheduler(eng, cfg, 4)

	var emittedTokens []int32
	onTok := func(tok int32, txt string) bool {
		emittedTokens = append(emittedTokens, tok)
		return true
	}

	seq := NewSequence("seq-real", "test", []int32{10, 20, 30}, 3, onTok)
	if err := sch.Submit(seq); err != nil {
		t.Fatalf("submit: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Step through prefill and 3 decode steps
	for step := 0; step < 10 && sch.ActiveCount() > 0; step++ {
		n, err := sch.Step(ctx)
		if err != nil {
			t.Fatalf("step %d failed: %v", step, err)
		}
		t.Logf("Step %d: processed %d callbacks, active=%d", step, n, sch.ActiveCount())
	}

	if seq.Status != SeqFinished {
		t.Fatalf("expected sequence to finish, got status %v", seq.Status)
	}

	if len(seq.GeneratedTokens) != 3 {
		t.Fatalf("expected 3 generated tokens, got %d: %v", len(seq.GeneratedTokens), seq.GeneratedTokens)
	}

	if len(emittedTokens) != 3 {
		t.Fatalf("expected 3 emitted tokens from callbacks, got %d: %v", len(emittedTokens), emittedTokens)
	}

	select {
	case <-seq.DoneCh:
	default:
		t.Errorf("expected DoneCh to be closed upon completion")
	}
}

func TestSchedulerMultiSequenceInterleavedEquivalence(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatalf("GetOrCompileEngine failed: %v", err)
	}

	promptA := []int32{10, 20, 30}
	promptB := []int32{40, 50}
	maxTokA := 4
	maxTokB := 3

	// 1. Run Sequence A in isolation with fresh KV
	kvA, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatalf("new sequence kv A: %v", err)
	}
	defer kvA.Free()
	nextA, err := GPUPrefillPassKV(eng, cfg, kvA, promptA)
	if err != nil {
		t.Fatalf("prefill A: %v", err)
	}
	wantA := []int32{nextA}
	curA := nextA
	for step := 0; step < maxTokA-1; step++ {
		curA, err = GPUForwardPassKV(eng, cfg, kvA, curA, len(promptA)+step)
		if err != nil {
			t.Fatalf("decode A step %d: %v", step, err)
		}
		wantA = append(wantA, curA)
	}

	// 2. Run Sequence B in isolation with fresh KV
	kvB, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatalf("new sequence kv B: %v", err)
	}
	defer kvB.Free()
	nextB, err := GPUPrefillPassKV(eng, cfg, kvB, promptB)
	if err != nil {
		t.Fatalf("prefill B: %v", err)
	}
	wantB := []int32{nextB}
	curB := nextB
	for step := 0; step < maxTokB-1; step++ {
		curB, err = GPUForwardPassKV(eng, cfg, kvB, curB, len(promptB)+step)
		if err != nil {
			t.Fatalf("decode B step %d: %v", step, err)
		}
		wantB = append(wantB, curB)
	}

	// 3. Run both sequences simultaneously through Scheduler
	sch := NewScheduler(eng, cfg, 4)
	seqA := NewSequence("seq-A", "prompt A", promptA, maxTokA, nil)
	seqB := NewSequence("seq-B", "prompt B", promptB, maxTokB, nil)

	if err := sch.Submit(seqA); err != nil {
		t.Fatalf("submit seqA: %v", err)
	}
	if err := sch.Submit(seqB); err != nil {
		t.Fatalf("submit seqB: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	for step := 0; step < 20 && sch.ActiveCount() > 0; step++ {
		_, err := sch.Step(ctx)
		if err != nil {
			t.Fatalf("interleaved step %d failed: %v", step, err)
		}
	}

	if seqA.Status != SeqFinished {
		t.Fatalf("seqA expected Finished, got %v", seqA.Status)
	}
	if seqB.Status != SeqFinished {
		t.Fatalf("seqB expected Finished, got %v", seqB.Status)
	}

	if len(seqA.GeneratedTokens) != len(wantA) {
		t.Fatalf("seqA length mismatch: got %d, want %d", len(seqA.GeneratedTokens), len(wantA))
	}
	for i := range wantA {
		if seqA.GeneratedTokens[i] != wantA[i] {
			t.Errorf("seqA token %d mismatch: got %d, want %d", i, seqA.GeneratedTokens[i], wantA[i])
		}
	}

	if len(seqB.GeneratedTokens) != len(wantB) {
		t.Fatalf("seqB length mismatch: got %d, want %d", len(seqB.GeneratedTokens), len(wantB))
	}
	for i := range wantB {
		if seqB.GeneratedTokens[i] != wantB[i] {
			t.Errorf("seqB token %d mismatch: got %d, want %d", i, seqB.GeneratedTokens[i], wantB[i])
		}
	}
}
