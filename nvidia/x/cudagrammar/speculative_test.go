// SPDX-License-Identifier: Apache-2.0

package cudagrammar_test

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagrammar"
)

func buildSpecVocab() *cudagrammar.Vocabulary {
	return cudagrammar.NewVocabulary([]string{
		0:  `{`,
		1:  `"n"`,
		2:  `:`,
		3:  `1`,
		4:  `2`,
		5:  `,`,
		6:  `"s"`,
		7:  `"ok"`,
		8:  `}`,
		9:  `true`,
		10: `[`,
		11: `]`,
	})
}

func TestMatcherSaveRestoreRollback(t *testing.T) {
	const schema = `{"type":"object","properties":{"n":{"type":"integer"},"s":{"type":"string"}}}`
	g, err := cudagrammar.CompileString(schema)
	if err != nil {
		t.Fatalf("CompileString: %v", err)
	}
	v := buildSpecVocab()

	// Full valid token sequence: { "n" : 1 , "s" : "ok" }
	seq := []int32{0, 1, 2, 3, 5, 6, 2, 7, 8}

	m := g.Match()
	checkpoints := make([]cudagrammar.Checkpoint, len(seq)+1)
	masksAtStep := make([][]uint32, len(seq)+1)

	bm0, err := m.Bitmask(v)
	if err != nil {
		t.Fatalf("Bitmask(0): %v", err)
	}
	checkpoints[0] = m.Save()
	masksAtStep[0] = append([]uint32(nil), bm0.Words...)

	for i, tok := range seq {
		if err := m.AcceptToken(v, tok); err != nil {
			t.Fatalf("step %d AcceptToken(%d): %v", i, tok, err)
		}
		checkpoints[i+1] = m.Save()
		bm, err := m.Bitmask(v)
		if err != nil {
			t.Fatalf("Bitmask(%d): %v", i+1, err)
		}
		masksAtStep[i+1] = append([]uint32(nil), bm.Words...)
	}
	if !m.Done() {
		t.Fatal("expected matcher to be Done after full sequence")
	}

	// Error cases on Rollback.
	if err := m.Rollback(-1); err == nil {
		t.Fatal("expected error for negative rollback steps")
	}
	if err := m.Rollback(len(seq) + 1); err == nil {
		t.Fatal("expected error when rollback steps exceed history length")
	}
	if err := m.Rollback(0); err != nil {
		t.Fatalf("Rollback(0) failed: %v", err)
	}

	// Roll back step by step and compare against fresh oracle matchers run from root.
	for step := len(seq) - 1; step >= 0; step-- {
		if err := m.Rollback(1); err != nil {
			t.Fatalf("Rollback(1) to step %d: %v", step, err)
		}
		oracle := g.Match()
		for j := 0; j < step; j++ {
			if err := oracle.AcceptToken(v, seq[j]); err != nil {
				t.Fatalf("oracle AcceptToken(%d): %v", j, err)
			}
		}
		if m.Pos() != oracle.Pos() || m.Done() != oracle.Done() {
			t.Fatalf("step %d: after Rollback Pos=%d Done=%v, want Pos=%d Done=%v",
				step, m.Pos(), m.Done(), oracle.Pos(), oracle.Done())
		}
		gotBM, err := m.Bitmask(v)
		if err != nil {
			t.Fatalf("step %d Bitmask: %v", step, err)
		}
		if !reflect.DeepEqual(gotBM.Words, masksAtStep[step]) {
			t.Fatalf("step %d: bitmask words after Rollback = %v, want %v", step, gotBM.Words, masksAtStep[step])
		}
	}

	// Now test Save/Restore at arbitrary checkpoints and multi-step Rollback from restored state.
	for targetStep := len(seq); targetStep >= 0; targetStep-- {
		if err := m.Restore(checkpoints[targetStep]); err != nil {
			t.Fatalf("Restore(step %d): %v", targetStep, err)
		}
		gotBM, err := m.Bitmask(v)
		if err != nil {
			t.Fatalf("Bitmask after Restore(%d): %v", targetStep, err)
		}
		if !reflect.DeepEqual(gotBM.Words, masksAtStep[targetStep]) {
			t.Fatalf("Restore(%d) bitmask = %v, want %v", targetStep, gotBM.Words, masksAtStep[targetStep])
		}
		if targetStep >= 3 {
			if err := m.Rollback(3); err != nil {
				t.Fatalf("Rollback(3) from restored step %d: %v", targetStep, err)
			}
			rewoundBM, err := m.Bitmask(v)
			if err != nil {
				t.Fatalf("Bitmask after Rollback(3): %v", err)
			}
			if !reflect.DeepEqual(rewoundBM.Words, masksAtStep[targetStep-3]) {
				t.Fatalf("Rollback(3) from %d: got bitmask %v, want %v",
					targetStep, rewoundBM.Words, masksAtStep[targetStep-3])
			}
		}
	}

	// Invalid zero-value Checkpoint must be rejected by Restore.
	if err := m.Restore(cudagrammar.Checkpoint{}); err == nil {
		t.Fatal("expected Restore on zero Checkpoint to return an error")
	}
}

func TestSpeculativeMasksAndCommitPrefix(t *testing.T) {
	const schema = `{"type":"object","properties":{"n":{"type":"integer"},"s":{"type":"string"}}}`
	g, err := cudagrammar.CompileString(schema)
	if err != nil {
		t.Fatalf("CompileString: %v", err)
	}
	v := buildSpecVocab()
	m := g.Match()

	// Advance prefix: { "n" :
	if err := m.CommitPrefix(v, []int32{0, 1, 2}); err != nil {
		t.Fatalf("CommitPrefix initial: %v", err)
	}
	posBefore := m.Pos()
	bmBefore, err := m.Bitmask(v)
	if err != nil {
		t.Fatalf("Bitmask before SpeculativeMasks: %v", err)
	}

	// Draft chain where tokens 0..2 ([3="1", 5=",", 6="\"s\""]) are valid,
	// token 3 (9="true" instead of 2=":") is INVALID, and token 4 (7="\"ok\"") follows.
	draft := []int32{3, 5, 6, 9, 7}
	masks, err := m.SpeculativeMasks(v, draft)
	if err != nil {
		t.Fatalf("SpeculativeMasks: %v", err)
	}
	if len(masks) != len(draft)+1 {
		t.Fatalf("len(masks) = %d, want %d", len(masks), len(draft)+1)
	}

	// Verify that m was rolled back to its exact state before SpeculativeMasks.
	if m.Pos() != posBefore || m.Done() {
		t.Fatalf("after SpeculativeMasks: Pos=%d Done=%v, want Pos=%d Done=false", m.Pos(), m.Done(), posBefore)
	}
	bmAfter, err := m.Bitmask(v)
	if err != nil {
		t.Fatalf("Bitmask after SpeculativeMasks: %v", err)
	}
	if !reflect.DeepEqual(bmAfter.Words, bmBefore.Words) {
		t.Fatalf("matcher bitmask mutated by SpeculativeMasks: got %v, want %v", bmAfter.Words, bmBefore.Words)
	}

	// Verify masks[0..3] match step-by-step oracle matchers:
	// mask[0]: state after `{ "n" :` -> admits `1` (token 3) and `2` (token 4).
	if !masks[0].IsAllowed(3) || !masks[0].IsAllowed(4) || masks[0].IsAllowed(9) {
		t.Fatalf("masks[0] unexpected allowed set: words=%#x", masks[0].Words)
	}
	// mask[1]: state after `{ "n" : 1` -> admits `,` (token 5) and more digits (3, 4).
	if !masks[1].IsAllowed(5) {
		t.Fatalf("masks[1] should allow ',' (token 5): words=%#x", masks[1].Words)
	}
	// mask[2]: state after `{ "n" : 1 ,` -> admits `"s"` (token 6).
	if !masks[2].IsAllowed(6) || masks[2].IsAllowed(1) {
		t.Fatalf("masks[2] should allow '\"s\"' (token 6) and not '\"n\"': words=%#x", masks[2].Words)
	}
	// mask[3]: state after `{ "n" : 1 , "s"` -> admits `:` (token 2) ONLY.
	if !masks[3].IsAllowed(2) || masks[3].IsAllowed(9) {
		t.Fatalf("masks[3] should allow ':' (token 2) and forbid 'true' (token 9): words=%#x", masks[3].Words)
	}
	// Since draft[3] = 9 ("true") is rejected at step 3, masks[4] and masks[5] must be all-zero.
	for rem := 4; rem < len(masks); rem++ {
		if masks[rem].CountAllowed() != 0 {
			t.Fatalf("masks[%d] after rejected draft token has %d allowed tokens, want 0", rem, masks[rem].CountAllowed())
		}
	}

	// CommitPrefix with an invalid token in the slice must fail and roll back all changes from that call.
	if err := m.CommitPrefix(v, []int32{3, 5, 9}); err == nil {
		t.Fatal("expected CommitPrefix with invalid token 9 to fail")
	}
	if m.Pos() != posBefore {
		t.Fatalf("failed CommitPrefix left Pos=%d, want %d", m.Pos(), posBefore)
	}

	// Commit the valid prefix [3, 5, 6] plus bonus token 2 (":"), then finish with [7, 8].
	if err := m.CommitPrefix(v, []int32{3, 5, 6, 2}); err != nil {
		t.Fatalf("CommitPrefix valid: %v", err)
	}
	if err := m.CommitPrefix(v, []int32{7, 8}); err != nil {
		t.Fatalf("CommitPrefix finish: %v", err)
	}
	if !m.Done() {
		t.Fatal("expected matcher to be Done after committing remaining tokens")
	}
}
