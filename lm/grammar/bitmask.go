// SPDX-License-Identifier: Apache-2.0

package grammar

// BitmaskTable stores pre-computed 64-bit boolean bitmasks for vocab token validity per FSM state.
type BitmaskTable struct {
	NumStates int
	VocabSize int
	Words     int // (VocabSize + 63) / 64
	Table     [][]uint64
}

// NewBitmaskTable initializes a bitmask lookup table for a given vocabulary
// size and state count, with every token forbidden. A negative size is treated
// as zero.
func NewBitmaskTable(numStates, vocabSize int) *BitmaskTable {
	numStates = max(numStates, 0)
	vocabSize = max(vocabSize, 0)
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

// Masked is the logit a forbidden token is given. It is a large finite value
// rather than -Inf so that a softmax over the masked logits stays finite.
const Masked = -1e9

// ApplyMask sets the logits of every token forbidden in state to [Masked].
// Tokens past the table's vocabulary are left alone.
func (b *BitmaskTable) ApplyMask(state int, logits []float32) {
	if state < 0 || state >= b.NumStates || len(logits) == 0 {
		return
	}
	mask := b.Table[state]
	for i := 0; i < len(logits); i++ {
		wordIdx := i / 64
		bitIdx := uint(i % 64)
		if wordIdx < len(mask) {
			if (mask[wordIdx] & (1 << bitIdx)) == 0 {
				logits[i] = Masked
			}
		}
	}
}

// JSONSchemaFSM holds a bitmask table indexed by a position in a schema's
// property list.
//
// The table it builds permits every token in every state: it reserves the
// shape of a schema-constrained decoder without yet enforcing a schema. Use
// [JSONConstraint] for the syntax checking that is implemented.
type JSONSchemaFSM struct {
	Properties   []string
	BitmaskTable *BitmaskTable
	CurrentState int
}

// NewJSONSchemaFSM creates an FSM constraining generation to specific schema keys and values.
func NewJSONSchemaFSM(properties []string, vocabSize int) *JSONSchemaFSM {
	numStates := len(properties)*2 + 2
	bt := NewBitmaskTable(numStates, vocabSize)

	// Populate basic bitmasks
	for s := 0; s < numStates; s++ {
		for t := 0; t < vocabSize; t++ {
			bt.SetValid(s, t)
		}
	}

	return &JSONSchemaFSM{
		Properties:   properties,
		BitmaskTable: bt,
		CurrentState: 0,
	}
}

// FilterLogits applies the bitmask constraint to output logits.
func (f *JSONSchemaFSM) FilterLogits(logits []float32) {
	f.BitmaskTable.ApplyMask(f.CurrentState, logits)
}

// Step advances to the next state. The chosen token does not yet affect the
// transition; see [JSONSchemaFSM].
func (f *JSONSchemaFSM) Step() {
	if f.CurrentState < f.BitmaskTable.NumStates-1 {
		f.CurrentState++
	}
}
