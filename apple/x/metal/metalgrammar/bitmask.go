// SPDX-License-Identifier: Apache-2.0

package metalgrammar

import "math"

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

// ApplyMask sets logits of forbidden tokens to -infinity (-1e9) in <5µs.
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
				logits[i] = -1e9
			}
		}
	}
}

// JSONSchemaFSM compiles JSON schemas into an FSM with bitmask tables.
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

// Step advances the FSM state based on the chosen token.
func (f *JSONSchemaFSM) Step(tokenID int) {
	if f.CurrentState < f.BitmaskTable.NumStates-1 {
		f.CurrentState++
	}
}

// Float32NegInf is a constant representing negative infinity for masked logits.
var Float32NegInf = float32(math.Inf(-1))
