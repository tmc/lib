// SPDX-License-Identifier: Apache-2.0

package cudagrammar

// BitmaskTable stores pre-computed 64-bit boolean bitmasks for vocab token validity per FSM state.
type BitmaskTable struct {
	NumStates int
	VocabSize int
	Words     int // (VocabSize + 63) / 64
	Table     [][]uint64
}

// NewBitmaskTable initializes a bitmask lookup table for a given vocabulary size and state count.
func NewBitmaskTable(numStates, vocabSize int) *BitmaskTable {
	words := (vocabSize + 63) / 64
	table := make([][]uint64, numStates)
	for i := 0; i < numStates; i++ {
		table[i] = make([]uint64, words)
	}
	return &BitmaskTable{
		NumStates: numStates,
		VocabSize: vocabSize,
		Words:     words,
		Table:     table,
	}
}

// SetValid marks tokenID as permissible in state.
func (b *BitmaskTable) SetValid(state, tokenID int) {
	if state < 0 || state >= b.NumStates || tokenID < 0 || tokenID >= b.VocabSize {
		return
	}
	wordIdx := tokenID / 64
	bitIdx := uint(tokenID % 64)
	b.Table[state][wordIdx] |= (1 << bitIdx)
}

// IsValid checks if tokenID is permissible in state.
func (b *BitmaskTable) IsValid(state, tokenID int) bool {
	if state < 0 || state >= b.NumStates || tokenID < 0 || tokenID >= b.VocabSize {
		return false
	}
	wordIdx := tokenID / 64
	bitIdx := uint(tokenID % 64)
	return (b.Table[state][wordIdx] & (1 << bitIdx)) != 0
}

// MaskedLogit is the value assigned to a token the grammar forbids. It is a
// large finite penalty rather than negative infinity, so that a state which
// happens to forbid every token yields a uniform distribution instead of NaN.
const MaskedLogit = float32(-1e9)

// ApplyMask sets the logit of every token the grammar forbids in state to
// MaskedLogit. Tokens at or beyond VocabSize are forbidden: the table has no
// bit for them, so they were never permitted.
func (b *BitmaskTable) ApplyMask(state int, logits []float32) {
	if state < 0 || state >= b.NumStates || len(logits) == 0 {
		return
	}
	mask := b.Table[state]
	for i := range logits {
		if i >= b.VocabSize {
			logits[i] = MaskedLogit
			continue
		}
		if mask[i/64]&(1<<uint(i%64)) == 0 {
			logits[i] = MaskedLogit
		}
	}
}
