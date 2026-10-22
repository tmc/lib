package main

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// legacyMTPDrafter reads actual target embedding and output weight tensors;
// it never calls the target evaluator to invent a support prediction.
type legacyMTPDrafter struct {
	state                     *legacyMTPState
	target                    *dsparkModel
	hidden                    []float32
	pending                   []*legacyMTPState
	pendingHidden             [][]float32
	margin                    float32
	prepared                  bool
	seed, seedPosition, first int
	firstLogits               []float32
	eos                       int
	historyHash               string
}

var _ dsv4.DraftLifecycle = (*legacyMTPDrafter)(nil)

// PreparedSeed reports the seed bound to a detached committed-history hash.
// Raw prepare is a value-reference helper; production verification requires
// prepareHistory and a currently published preparation.
func (d *legacyMTPDrafter) PreparedSeed() (int, int, error) {
	if d == nil || d.state == nil || !d.prepared || len(d.historyHash) != 64 || d.seedPosition < 0 || d.state.next != d.seedPosition+1 {
		return 0, 0, fmt.Errorf("legacy mtp: no prepared history frontier")
	}
	return d.seed, d.seedPosition + 1, nil
}

// CommitDrafts retains the accepted support prefix after target commitment.
func (d *legacyMTPDrafter) CommitDrafts(n int) error {
	if _, _, err := d.PreparedSeed(); err != nil {
		return err
	}
	if len(d.pending) != len(d.pendingHidden) {
		return fmt.Errorf("legacy mtp: incomplete pending support frontier")
	}
	return d.commit(n)
}

// DiscardDrafts removes lookahead while retaining the prepared seed row.
// It remains usable after a failed Draft or CommitDrafts operation.
func (d *legacyMTPDrafter) DiscardDrafts() error {
	return d.commit(0)
}

// prepareHistory binds the production proposal to the entire committed
// prefix, using the same zero-based position/hash convention as receipts.
func (d *legacyMTPDrafter) prepareHistory(history []int, targetHC []float32) error {
	if len(history) == 0 {
		return fmt.Errorf("legacy mtp prepare: missing committed history")
	}
	for _, token := range history {
		if token < 0 || token >= Flash.Vocab {
			return fmt.Errorf("legacy mtp prepare: history token outside vocabulary")
		}
	}
	hash, err := dsv4.SpeculativeHistorySHA256(history)
	if err != nil {
		return fmt.Errorf("legacy mtp prepare history: %w", err)
	}
	if err := d.prepare(history[len(history)-1], len(history)-1, targetHC); err != nil {
		return err
	}
	d.historyHash = hash
	return nil
}

func newLegacyMTPDrafter(s *legacyMTPState, target *dsparkModel, hidden []float32) (*legacyMTPDrafter, error) {
	if s == nil || target == nil || target.reader == nil || len(hidden) != Flash.Embed*Flash.HC || !allFinite(hidden) {
		return nil, errShape
	}
	for _, name := range []string{"token_embd.weight", "output.weight"} {
		t, ok := target.tensor(name)
		if !ok || len(t.Dimensions) != 2 || t.Dimensions[0] != uint64(Flash.Embed) || t.Dimensions[1] != uint64(Flash.Vocab) || (t.TypeID != 0 && t.TypeID != 1 && t.TypeID != 8) {
			return nil, fmt.Errorf("legacy mtp: unsupported target tensor %s", name)
		}
	}
	return &legacyMTPDrafter{state: s.clone(), target: target, hidden: append([]float32(nil), hidden...), margin: 3, eos: -1}, nil
}

// prepare consumes a freshly committed target HC/token row before proposal
// verification. Its one support row survives a free first-token miss, as
// ds4_session_prepare_legacy_mtp_draft does during ordinary target Eval.
func (d *legacyMTPDrafter) prepare(token, position int, targetHC []float32) error {
	if d == nil || d.state == nil || d.target == nil || token < 0 || token >= Flash.Vocab || position < 0 || len(d.pending) != 0 || len(targetHC) != Flash.Embed*Flash.HC || !allFinite(targetHC) {
		return errShape
	}
	work := d.state.clone()
	// A fresh support ring can begin at any committed target frontier. Later
	// preparation must continue the admitted support timeline exactly.
	if work.count == 0 {
		work.next = position
	}
	if work.next != position {
		return fmt.Errorf("legacy mtp prepare: target/support frontier mismatch")
	}
	embedding, err := d.target.denseRow("token_embd.weight", token)
	if err != nil {
		return err
	}
	head, hidden, err := work.step(embedding, targetHC, position)
	if err != nil {
		return err
	}
	logits, err := d.target.denseMatvec("output.weight", head)
	if err != nil {
		return err
	}
	first := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[first] {
			first = i
		}
	}
	d.state, d.hidden = work, append([]float32(nil), hidden...)
	d.seed, d.seedPosition, d.first, d.firstLogits, d.prepared = token, position, first, logits, true
	d.historyHash = ""
	return nil
}

func (d *legacyMTPDrafter) Draft(token, position, maxDrafts int) ([]int, error) {
	// The verifier's position is the next target position. The support
	// feature/token pair belongs to the preceding committed row, exactly as
	// prepare_support_draft(checkpoint.len-1) does at ds4.c:72244.
	if d == nil || d.state == nil || !d.prepared || maxDrafts < 1 || maxDrafts > 16 || token != d.seed || position != d.seedPosition+1 || position != d.state.next {
		return nil, errShape
	}
	d.pending, d.pendingHidden = nil, nil
	work := d.state.clone()
	hidden := append([]float32(nil), d.hidden...)
	tokens := []int{d.first}
	states := []*legacyMTPState{work.clone()}
	hiddens := [][]float32{append([]float32(nil), hidden...)}
	lastLogits := d.firstLogits
	token = d.first
	for i := 1; i < maxDrafts && token != d.eos; i++ {
		embedding, err := d.target.denseRow("token_embd.weight", token)
		if err != nil {
			return nil, err
		}
		head, next, err := work.step(embedding, hidden, position+i-1)
		if err != nil {
			return nil, err
		}
		logits, err := d.target.denseMatvec("output.weight", head)
		if err != nil {
			return nil, err
		}
		lastLogits = logits
		token = 0
		for j := 1; j < len(logits); j++ {
			if logits[j] > logits[token] {
				token = j
			}
		}
		tokens = append(tokens, token)
		hidden = next
		states = append(states, work.clone())
		hiddens = append(hiddens, append([]float32(nil), hidden...))
	}
	allow, err := legacyMTPMargin(lastLogits, len(tokens), d.margin)
	if err != nil {
		return nil, err
	}
	if !allow {
		tokens, states, hiddens = tokens[:1], states[:1], hiddens[:1]
	}
	d.pending, d.pendingHidden = states, hiddens
	return tokens, nil
}

// commit retains exactly the accepted support frontier. Zero discards the
// proposal; an invalid prefix cannot mutate the active support state.
func (d *legacyMTPDrafter) commit(n int) error {
	if d == nil || n < 0 || n > len(d.pending) {
		return errShape
	}
	if n > 0 {
		d.state = d.pending[n-1].clone()
		d.hidden = append([]float32(nil), d.pendingHidden[n-1]...)
	}
	d.pending, d.pendingHidden = nil, nil
	d.prepared = false
	return nil
}
