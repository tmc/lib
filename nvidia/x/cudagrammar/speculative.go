// SPDX-License-Identifier: Apache-2.0

package cudagrammar

import (
	"fmt"
	"math/bits"
)

type matcherSnapshot struct {
	stack []frame
	pos   int
}

func cloneFrames(src []frame) []frame {
	if len(src) == 0 {
		return nil
	}
	dst := make([]frame, len(src))
	copy(dst, src)
	for i := range dst {
		if dst[i].cands != nil {
			c := make([]int, len(dst[i].cands))
			copy(c, dst[i].cands)
			dst[i].cands = c
		}
	}
	return dst
}

func (m *Matcher) snapshot() matcherSnapshot {
	return matcherSnapshot{
		stack: cloneFrames(m.stack),
		pos:   m.pos,
	}
}

func (m *Matcher) restoreSnapshot(s matcherSnapshot) {
	m.stack = cloneFrames(s.stack)
	m.pos = s.pos
}

// Checkpoint captures the complete state and transition history of a [Matcher]
// so it can be restored in O(1) without re-running from the root of the document.
type Checkpoint struct {
	state     int
	history   []int
	snap      matcherSnapshot
	snapshots []matcherSnapshot
	valid     bool
}

// Save captures the current state and transition history of m in O(1).
func (m *Matcher) Save() Checkpoint {
	histOffsets := make([]int, len(m.history))
	snaps := make([]matcherSnapshot, len(m.history))
	for i, h := range m.history {
		histOffsets[i] = h.pos
		snaps[i] = h
	}
	return Checkpoint{
		state:     m.pos,
		history:   histOffsets,
		snap:      m.snapshot(),
		snapshots: snaps,
		valid:     true,
	}
}

// Restore resets m to the state and transition history captured in cp.
func (m *Matcher) Restore(cp Checkpoint) error {
	if m == nil {
		return fmt.Errorf("cudagrammar: restore on nil matcher")
	}
	if !cp.valid || cp.state < 0 || len(cp.history) != len(cp.snapshots) {
		return fmt.Errorf("cudagrammar: invalid checkpoint")
	}
	m.restoreSnapshot(cp.snap)
	if len(cp.snapshots) == 0 {
		m.history = nil
	} else {
		m.history = make([]matcherSnapshot, len(cp.snapshots))
		copy(m.history, cp.snapshots)
	}
	return nil
}

// Rollback rewinds steps accepted transitions along m's transition history,
// matching XGrammar's matcher.rollback(n) in DeepInfer's speculative sampler.
func (m *Matcher) Rollback(steps int) error {
	if m == nil {
		return fmt.Errorf("cudagrammar: rollback on nil matcher")
	}
	if steps < 0 {
		return fmt.Errorf("cudagrammar: negative rollback steps %d", steps)
	}
	if steps == 0 {
		return nil
	}
	if steps > len(m.history) {
		return fmt.Errorf("cudagrammar: rollback %d steps exceeds history length %d", steps, len(m.history))
	}
	targetIdx := len(m.history) - steps
	target := m.history[targetIdx]
	m.restoreSnapshot(target)
	m.history = m.history[:targetIdx]
	return nil
}

// Vocabulary maps token IDs in [0, Size) to their decoded byte strings.
type Vocabulary struct {
	tokens []string
}

// NewVocabulary constructs a Vocabulary from a slice of token strings indexed by token ID.
func NewVocabulary(tokens []string) *Vocabulary {
	copied := make([]string, len(tokens))
	copy(copied, tokens)
	return &Vocabulary{tokens: copied}
}

// Size returns the number of tokens in v.
func (v *Vocabulary) Size() int {
	if v == nil {
		return 0
	}
	return len(v.tokens)
}

// Token returns the string representation of token ID id, or "" if id is out of bounds.
func (v *Vocabulary) Token(id int) string {
	if v == nil || id < 0 || id >= len(v.tokens) {
		return ""
	}
	return v.tokens[id]
}

// Vocab returns a [Vocab] function view of v for use with [Matcher.Admits],
// [Matcher.TokenMask], and [Matcher.ApplyMask].
func (v *Vocabulary) Vocab() Vocab {
	return func(id int) []byte {
		if v == nil || id < 0 || id >= len(v.tokens) {
			return nil
		}
		s := v.tokens[id]
		if len(s) == 0 {
			return nil
		}
		return []byte(s)
	}
}

// Bitmask holds a packed 32-bit word token bitmask ([ceil(VocabSize/32)]uint32)
// representing the set of vocabulary tokens admitted at a grammar step.
type Bitmask struct {
	VocabSize int
	Words     []uint32
}

// NewBitmask allocates an all-zero Bitmask for vocabSize tokens.
func NewBitmask(vocabSize int) *Bitmask {
	if vocabSize < 0 {
		vocabSize = 0
	}
	words := (vocabSize + 31) / 32
	return &Bitmask{
		VocabSize: vocabSize,
		Words:     make([]uint32, words),
	}
}

// Set marks tokenID as allowed (or disallowed when allowed is false).
func (b *Bitmask) Set(tokenID int, allowed bool) {
	if b == nil || tokenID < 0 || tokenID >= b.VocabSize {
		return
	}
	w := tokenID / 32
	bit := uint32(1) << uint(tokenID%32)
	if allowed {
		b.Words[w] |= bit
	} else {
		b.Words[w] &^= bit
	}
}

// IsAllowed reports whether tokenID is marked as valid in b.
func (b *Bitmask) IsAllowed(tokenID int) bool {
	if b == nil || tokenID < 0 || tokenID >= b.VocabSize {
		return false
	}
	w := tokenID / 32
	if w >= len(b.Words) {
		return false
	}
	return (b.Words[w] & (uint32(1) << uint(tokenID%32))) != 0
}

// CountAllowed returns the number of allowed tokens in b.
func (b *Bitmask) CountAllowed() int {
	if b == nil {
		return 0
	}
	total := 0
	for _, w := range b.Words {
		total += bits.OnesCount32(w)
	}
	return total
}

// ApplyInPlace sets logits[i] to [MaskedLogit] for every token i not allowed by b.
func (b *Bitmask) ApplyInPlace(logits []float32) {
	if b == nil {
		return
	}
	for i := range logits {
		if !b.IsAllowed(i) {
			logits[i] = MaskedLogit
		}
	}
}

// Bitmask computes the packed [Bitmask] of tokens in v admitted from m's current state.
func (m *Matcher) Bitmask(v *Vocabulary) (*Bitmask, error) {
	if m == nil {
		return nil, fmt.Errorf("cudagrammar: nil matcher")
	}
	if v == nil || v.Size() == 0 {
		return nil, fmt.Errorf("cudagrammar: empty vocabulary")
	}
	bm := NewBitmask(v.Size())
	if m.Done() {
		return bm, nil
	}
	vocabFn := v.Vocab()
	for id := 0; id < v.Size(); id++ {
		if m.Admits(vocabFn, id) {
			bm.Set(id, true)
		}
	}
	return bm, nil
}

// AcceptToken validates and advances m by the bytes of tokenID in v as a single
// atomic transition step. If tokenID is out of range, empty, or rejected by the
// grammar, m is left completely unchanged and an error is returned.
func (m *Matcher) AcceptToken(v *Vocabulary, tokenID int32) error {
	if m == nil {
		return fmt.Errorf("cudagrammar: nil matcher")
	}
	if v == nil {
		return fmt.Errorf("cudagrammar: nil vocabulary")
	}
	if m.Done() {
		return fmt.Errorf("cudagrammar: matcher already terminated at offset %d", m.pos)
	}
	tok := v.Token(int(tokenID))
	if len(tok) == 0 {
		return fmt.Errorf("cudagrammar: invalid or empty token id %d", tokenID)
	}
	snap := m.snapshot()
	for i := 0; i < len(tok); i++ {
		if err := m.acceptByteNoHistory(tok[i]); err != nil {
			m.restoreSnapshot(snap)
			return err
		}
	}
	m.history = append(m.history, snap)
	return nil
}

// SpeculativeMasks implements DeepInfer's _get_so_mask loop for an N-token
// speculative draft chain:
//  1. Emits mask 0 for the current matcher state (validating draftTokens[0]).
//  2. For each draft token t_k (k in [0, len(draftTokens))): attempts to advance
//     m with v.Token(int(t_k)). If accepted, emits mask k+1 from the new matcher
//     state; if rejected (or m already terminated), fills remaining masks
//     k+1..len(draftTokens) with an all-zero bitmask and stops advancing.
//  3. Automatically calls [Matcher.Rollback] for the exact number of accepted
//     draft transitions before returning so m's state is unchanged until
//     verification commits the accepted prefix via [Matcher.CommitPrefix].
func (m *Matcher) SpeculativeMasks(v *Vocabulary, draftTokens []int32) ([]*Bitmask, error) {
	if m == nil {
		return nil, fmt.Errorf("cudagrammar: nil matcher")
	}
	if v == nil || v.Size() == 0 {
		return nil, fmt.Errorf("cudagrammar: empty vocabulary")
	}

	masks := make([]*Bitmask, len(draftTokens)+1)
	mask0, err := m.Bitmask(v)
	if err != nil {
		return nil, err
	}
	masks[0] = mask0

	acceptedSteps := 0
	for k, tk := range draftTokens {
		if m.Done() {
			for rem := k + 1; rem <= len(draftTokens); rem++ {
				masks[rem] = NewBitmask(v.Size())
			}
			break
		}
		if err := m.AcceptToken(v, tk); err != nil {
			for rem := k + 1; rem <= len(draftTokens); rem++ {
				masks[rem] = NewBitmask(v.Size())
			}
			break
		}
		acceptedSteps++
		nextMask, err := m.Bitmask(v)
		if err != nil {
			_ = m.Rollback(acceptedSteps)
			return nil, err
		}
		masks[k+1] = nextMask
	}

	if err := m.Rollback(acceptedSteps); err != nil {
		return nil, err
	}
	return masks, nil
}

// CommitPrefix advances m across each token in acceptedTokens after speculative
// verification. If any token in acceptedTokens is rejected by the grammar,
// CommitPrefix rolls back all tokens accepted during this call and returns an error.
func (m *Matcher) CommitPrefix(v *Vocabulary, acceptedTokens []int32) error {
	if m == nil {
		return fmt.Errorf("cudagrammar: nil matcher")
	}
	if v == nil || v.Size() == 0 {
		return fmt.Errorf("cudagrammar: empty vocabulary")
	}
	committed := 0
	for i, tk := range acceptedTokens {
		if err := m.AcceptToken(v, tk); err != nil {
			_ = m.Rollback(committed)
			return fmt.Errorf("cudagrammar: commit prefix token %d (id=%d): %w", i, tk, err)
		}
		committed++
	}
	return nil
}
