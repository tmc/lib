package main

import (
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/lm/speculative"
	"github.com/tmc/lib/nvidia/x/cudamtp"
)

// speculativeStep runs one step of speculative decoding with greedy
// verification. next is the pending token: predicted, but neither emitted
// nor fed. speculativeStep feeds next at pos and then as many of drafts as
// fit both the KV cache of maxSeq positions and the budget of output tokens
// left, keeps the longest prefix of drafts the model agrees with, and
// returns the tokens to emit (next, then the accepted drafts), the new
// pending token and the new position.
//
// The new position is just past the last accepted token. Rejected drafts
// were written to cache positions at or beyond it, which attention does not
// read before they are overwritten.
//
// A zero budget feeds nothing and returns next and pos unchanged. If
// forward fails, speculativeStep returns its error, and the cache may hold
// entries written before the failure.
func speculativeStep(forward func(tok int32, pos int) (int32, error), next int32, drafts []int32, pos, maxSeq, budget int) (emit []int32, newNext int32, newPos int, err error) {
	switch {
	case forward == nil:
		return nil, next, pos, errors.New("speculative step: no forward function")
	case maxSeq <= 0:
		return nil, next, pos, fmt.Errorf("speculative step: kv cache of %d positions", maxSeq)
	case pos < 0 || pos >= maxSeq:
		return nil, next, pos, fmt.Errorf("speculative step: position %d outside the kv cache of %d", pos, maxSeq)
	case budget < 0:
		return nil, next, pos, fmt.Errorf("speculative step: budget %d", budget)
	case budget == 0:
		return nil, next, pos, nil
	}
	k := min(len(drafts), maxSeq-pos-1, budget-1)
	drafts = drafts[:k]
	targets := make([]int32, 0, k+1)
	for i, tok := range append([]int32{next}, drafts...) {
		pred, err := forward(tok, pos+i)
		if err != nil {
			return nil, next, pos, fmt.Errorf("position %d: %w", pos+i, err)
		}
		targets = append(targets, pred)
	}
	accepted := speculative.VerifyGreedy(targets, drafts).AcceptedTokens
	emit = append([]int32{next}, accepted...)
	return emit, targets[len(accepted)], pos + 1 + len(accepted), nil
}

// decodeState is the token bookkeeping of a decode loop. Every emitted token
// is appended to generated and to context, which starts as the prompt, and
// is fed to drafter, so the drafter's history is always context.
type decodeState struct {
	generated []int32
	context   []int32
	drafter   *speculative.NgramDrafter // nil when not decoding speculatively
	sched     *cudamtp.VarlenScheduler  // optional confidence-aware draft length scheduler
}

// newDecodeState returns a decodeState for prompt with room for steps
// generated tokens. A non-nil drafter is fed prompt.
func newDecodeState(prompt []int32, steps int, drafter *speculative.NgramDrafter) *decodeState {
	if drafter != nil {
		drafter.Feed(prompt)
	}
	return &decodeState{
		generated: make([]int32, 0, steps),
		context:   append([]int32(nil), prompt...),
		drafter:   drafter,
	}
}

// setScheduler attaches an optional variable-length MTP scheduler to s.
func (s *decodeState) setScheduler(sched *cudamtp.VarlenScheduler) {
	if s == nil {
		return
	}
	s.sched = sched
}

// emit records tok as emitted.
func (s *decodeState) emit(tok int32) {
	s.generated = append(s.generated, tok)
	s.context = append(s.context, tok)
	if s.drafter != nil {
		s.drafter.Feed([]int32{tok})
	}
}

// draft returns up to n tokens drafted to follow the pending token next,
// which is fed before them, so they are drafted from context then next.
func (s *decodeState) draft(next int32, n int) []int32 {
	if s.drafter == nil {
		return nil
	}
	return s.drafter.Draft(append(slices.Clip(s.context), next), n)
}

// speculate drafts up to n tokens to follow the pending token next and runs
// speculativeStep with them from pos, then emits the tokens it returns. It
// returns those tokens, the new pending token and the new position. If
// there are no drafts or budget is zero, speculate feeds and emits nothing
// and returns nil, next and pos.
func (s *decodeState) speculate(forward func(tok int32, pos int) (int32, error), next int32, n, pos, maxSeq, budget int) (emitted []int32, newNext int32, newPos int, err error) {
	draftBudget := n
	if s.sched != nil && draftBudget > 0 {
		scheduled, serr := scheduleDraftBudget(s.sched, 0, len(s.context)+1, draftBudget)
		if serr != nil {
			return nil, next, pos, serr
		}
		draftBudget = scheduled
	}
	drafts := s.draft(next, draftBudget)
	if len(drafts) == 0 {
		return nil, next, pos, nil
	}
	emitted, newNext, newPos, err = speculativeStep(forward, next, drafts, pos, maxSeq, budget)
	if err != nil {
		return nil, next, pos, err
	}
	if s.sched != nil && len(emitted) > 0 {
		s.sched.UpdateAcceptance(0, len(drafts), len(emitted)-1)
	}
	for _, tok := range emitted {
		s.emit(tok)
	}
	return emitted, newNext, newPos, nil
}
