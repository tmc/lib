package dsv4

import (
	"errors"
	"fmt"
	"io"
	"math"
)

// RunSpeculativeCase produces a strict and speculative trace using separate
// request States. The supplied drafter must be the production drafter; a
// strict-trace replay drafter is a verifier test, not a DSpark parity run.
// Sampling is greedy. The caller pins model, support model, prompt tokens,
// context, draft count and generated count in its external manifest.
// On cleanup failure the returned error implements io.Closer. The caller must
// retry that Close before releasing the backend or supplied drafter.
func RunSpeculativeCase(backend Backend, drafter Drafter, name string, prompt []int, context, count, drafts int) (r SpeculativeCaseReceipt, err error) {
	r = SpeculativeCaseReceipt{Name: name, Prompt: append([]int(nil), prompt...)}
	if backend == nil || drafter == nil || name == "" || len(prompt) == 0 || count <= 0 || drafts <= 0 || drafts > DefaultSpeculativeLimits.MaxDrafts || context < len(prompt) || count > context-len(prompt) {
		return r, fmt.Errorf("speculative run: invalid case parameters")
	}
	owner := new(speculativeRunOwner)
	defer func() {
		owner.pending = speculativeErrorOwner(err)
		if closeErr := owner.Close(); closeErr != nil {
			err = &speculativeCleanupError{cause: errors.Join(err, closeErr), owner: owner}
		}
	}()
	strict, err := backend.NewState(context)
	owner.strict = strict
	if err != nil {
		return r, err
	}
	if strict == nil {
		return r, fmt.Errorf("speculative run: nil strict state")
	}
	state, err := backend.NewState(context)
	owner.state = state
	if err != nil {
		return r, err
	}
	if state == nil {
		return r, fmt.Errorf("speculative run: nil speculative state")
	}
	for i, token := range prompt {
		if _, err := SpeculativeHistorySHA256([]int{token}); err != nil {
			return r, err
		}
		if err := backend.Eval(strict, token, i); err != nil {
			return r, err
		}
		if err := backend.Eval(state, token, i); err != nil {
			return r, err
		}
	}
	history := append([]int(nil), prompt...)
	for i := 0; i < count; i++ {
		token, margin, err := speculativeTopMargin(backend.Logits(strict))
		if err != nil {
			return r, err
		}
		hash, _ := SpeculativeHistorySHA256(history)
		r.StrictSteps = append(r.StrictSteps, SpeculativeStep{len(history), token, hash})
		r.Trace.Strict = append(r.Trace.Strict, token)
		r.Trace.Margins = append(r.Trace.Margins, margin)
		if err := backend.Eval(strict, token, len(history)); err != nil {
			return r, err
		}
		history = append(history, token)
	}
	spec, err := NewSpeculatorForBackend(backend, state, drafter)
	owner.spec = spec
	if err != nil {
		return r, err
	}
	history = append([]int(nil), prompt...)
	for len(r.Trace.Speculative) < count {
		n := min(drafts, count-len(r.Trace.Speculative))
		result, err := spec.Verify(history[len(history)-1], len(history), n)
		if err != nil {
			return r, err
		}
		r.Trace.Cycles++
		if result.Committed != result.Accepted || result.Accepted < 0 || result.Accepted > len(result.Drafts) {
			return r, fmt.Errorf("speculative run: invalid committed prefix")
		}
		tokens := result.Drafts[:result.Accepted]
		if len(tokens) == 0 {
			token, _, err := speculativeTopMargin(backend.Logits(state))
			if err != nil {
				return r, err
			}
			if err := backend.Eval(state, token, len(history)); err != nil {
				return r, err
			}
			tokens = []int{token}
		}
		r.Trace.AcceptedDrafts += result.Accepted
		for _, token := range tokens {
			hash, err := SpeculativeHistorySHA256(history)
			if err != nil {
				return r, err
			}
			r.SpeculativeSteps = append(r.SpeculativeSteps, SpeculativeStep{len(history), token, hash})
			r.Trace.Speculative = append(r.Trace.Speculative, token)
			history = append(history, token)
		}
	}
	return r, nil
}

// Keep all prerequisites alive until the retained frontier has been released.
type speculativeRunOwner struct {
	pending io.Closer
	spec    *Speculator
	state   State
	strict  State
}

func (o *speculativeRunOwner) Close() (err error) {
	defer func() {
		if err != nil {
			err = &speculativeCleanupError{cause: err, owner: o}
		}
	}()
	if o.pending != nil {
		if err := o.pending.Close(); err != nil {
			return err
		}
		o.pending = nil
	}
	if o.spec != nil {
		if err := o.spec.Close(); err != nil {
			return err
		}
		o.spec = nil
	}
	if o.state != nil {
		if err := o.state.Close(); err != nil {
			return fmt.Errorf("close speculative state: %w", err)
		}
		o.state = nil
	}
	if o.strict != nil {
		if err := o.strict.Close(); err != nil {
			return fmt.Errorf("close strict state: %w", err)
		}
		o.strict = nil
	}
	return nil
}

func speculativeErrorOwner(err error) io.Closer {
	var owner io.Closer
	if errors.As(err, &owner) {
		return owner
	}
	return nil
}

func speculativeTopMargin(logits []float32) (int, float64, error) {
	if len(logits) < 2 {
		return 0, 0, fmt.Errorf("speculative run: missing vocabulary")
	}
	first, second := math.Inf(-1), math.Inf(-1)
	index := 0
	for i, v := range logits {
		x := float64(v)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return 0, 0, fmt.Errorf("speculative run: nonfinite logit %d", i)
		}
		if x > first {
			second, first, index = first, x, i
		} else if x > second {
			second = x
		}
	}
	return index, first - second, nil
}
