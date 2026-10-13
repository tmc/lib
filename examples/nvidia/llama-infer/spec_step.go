package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/lm/speculative"
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
