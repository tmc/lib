// SPDX-License-Identifier: Apache-2.0

package cudagrammar

import "testing"

// TestApplyMaskCoversLogitsPastVocab covers a logits slice longer than the
// table's vocabulary. The table has no bit for those tokens, so they were never
// permitted, but ApplyMask used to skip them and leave them at their sampled
// value: a decoder could then pick a token the grammar never allowed.
func TestApplyMaskCoversLogitsPastVocab(t *testing.T) {
	// A vocabulary of 70 needs two 64-bit words, so the table addresses 128
	// bits while only 70 are meaningful.
	bt := NewBitmaskTable(2, 70)
	bt.SetValid(0, 5)

	logits := make([]float32, 200)
	for i := range logits {
		logits[i] = 1
	}
	bt.ApplyMask(0, logits)

	if logits[5] != 1 {
		t.Errorf("permitted token 5 was masked: %g", logits[5])
	}
	for i := range logits {
		if i == 5 {
			continue
		}
		if logits[i] != MaskedLogit {
			t.Fatalf("token %d = %g, want %g (VocabSize is %d)", i, logits[i], MaskedLogit, bt.VocabSize)
		}
	}
}

// TestApplyMaskPaddingBits checks the bits between VocabSize and the end of the
// final word, which SetValid refuses to set and IsValid reports as invalid.
func TestApplyMaskPaddingBits(t *testing.T) {
	bt := NewBitmaskTable(1, 70)
	for tok := 0; tok < 70; tok++ {
		bt.SetValid(0, tok)
	}

	// SetValid rejects anything past the vocabulary.
	bt.SetValid(0, 70)
	bt.SetValid(0, 127)
	if bt.IsValid(0, 70) || bt.IsValid(0, 127) {
		t.Error("a token past VocabSize reported as valid")
	}

	logits := make([]float32, 128)
	for i := range logits {
		logits[i] = 2
	}
	bt.ApplyMask(0, logits)

	for i := 0; i < 70; i++ {
		if logits[i] != 2 {
			t.Errorf("token %d was masked but is within the vocabulary", i)
		}
	}
	for i := 70; i < 128; i++ {
		if logits[i] != MaskedLogit {
			t.Errorf("padding token %d = %g, want %g", i, logits[i], MaskedLogit)
		}
	}
}
