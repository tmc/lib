// SPDX-License-Identifier: Apache-2.0

package cudagrammar

// Vocab is a view of a tokenizer's vocabulary: it returns the bytes a token id
// stands for, or nil for an id the tokenizer does not decode to text, such as a
// control token. Taking a function rather than a table keeps the grammar
// independent of any one tokenizer.
type Vocab func(id int) []byte

// Admits reports whether the grammar can accept every byte of token id from the
// matcher's current state. The matcher is not advanced.
//
// A token that decodes to no bytes is never admitted: it carries no evidence
// about the document, so a decoder that picked it would make no progress.
func (m *Matcher) Admits(v Vocab, id int) bool {
	b := v(id)
	if len(b) == 0 {
		return false
	}
	probe := m.Clone()
	return probe.Accept(b) == nil
}

// TokenMask returns the ids below vocabSize that the grammar admits, in
// increasing order. dst is appended to and may be nil.
//
// The result is the set a decoder may sample from. It is empty only if the
// vocabulary cannot continue the document at all, which is the caller's cue to
// stop rather than to sample.
func (m *Matcher) TokenMask(v Vocab, vocabSize int, dst []int) []int {
	for id := 0; id < vocabSize; id++ {
		if m.Admits(v, id) {
			dst = append(dst, id)
		}
	}
	return dst
}

// ApplyMask sets the logit of every token the grammar forbids to MaskedLogit,
// leaving the admitted ones as they are.
func (m *Matcher) ApplyMask(v Vocab, logits []float32) {
	for id := range logits {
		if !m.Admits(v, id) {
			logits[id] = MaskedLogit
		}
	}
}

// FillBitmask marks in table, for the given state row, every token the matcher
// admits. It lets a caller precompute a row once and apply it on the device
// with [BitmaskTable.ApplyMask].
func (m *Matcher) FillBitmask(v Vocab, table *BitmaskTable, state int) {
	for id := 0; id < table.VocabSize; id++ {
		if m.Admits(v, id) {
			table.SetValid(state, id)
		}
	}
}
