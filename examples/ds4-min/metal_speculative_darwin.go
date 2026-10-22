//go:build darwin && arm64

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// metalSpeculativeVerifier defaults to the scalar speculative contract. The
// opt-in grouped route submits device-only forward work together and batches
// every output row. CommitPrefix restores the frontier and replays only the
// accepted prefix with ordinary Eval in either mode.
type metalSpeculativeVerifier struct {
	adapter *metalStateAdapter
	state   *metalRequestState

	frontier      dsv4.Snapshot
	seed          int
	drafts        []int
	pos           int
	groupedHooks  *metalGroupedExecution // internal post-encoding fault seam
	lastGrouped   bool
	rowLogits     [][]float32
	rowEpoch      uint64
	closed        bool
	cleanupFailed bool
	failed        bool // execution remains disabled after cleanup succeeds
}

var _ dsv4.SpeculativeVerifier = (*metalSpeculativeVerifier)(nil)

func (v *metalSpeculativeVerifier) CurrentLogits() []float32 {
	if v.valid() != nil || v.cleanupFailed {
		return nil
	}
	return v.adapter.Logits(v.state)
}

func (v *metalSpeculativeVerifier) Snapshot() (dsv4.Snapshot, error) {
	if err := v.valid(); err != nil {
		return nil, err
	}
	return v.state.Snapshot()
}

func (v *metalSpeculativeVerifier) Restore(snapshot dsv4.Snapshot) error {
	// Rollback may finish after execution has failed, without reopening it.
	if err := v.validOwner(); err != nil {
		return err
	}
	if err := v.state.Restore(snapshot); err != nil {
		return err
	}
	v.drafts, v.rowLogits = nil, nil
	return v.releaseFrontier()
}

func (v *metalSpeculativeVerifier) valid() error {
	if err := v.validOwner(); err != nil {
		return err
	}
	if v.failed {
		return dsv4.ErrDesynchronized
	}
	return nil
}

func (v *metalSpeculativeVerifier) validOwner() error {
	if v == nil || v.closed || v.adapter == nil || v.adapter.closed || v.state == nil || v.state.closed || v.state.runtime == nil {
		return dsv4.ErrClosed
	}
	if v.state.owner != v.adapter {
		return dsv4.ErrUnimplemented
	}
	return metalStatePoison(v.state.runtime)
}

// Close retains an internal snapshot after a failed release and can be retried.
// It never closes the request State or its backend.
func (v *metalSpeculativeVerifier) Close() error {
	if v == nil {
		return nil
	}
	v.closed = true
	v.drafts, v.rowLogits = nil, nil
	return v.releaseFrontier()
}

func (v *metalSpeculativeVerifier) releaseFrontier() error {
	if v.frontier == nil {
		return nil
	}
	if err := v.frontier.Close(); err != nil {
		v.cleanupFailed = true
		v.failed = true
		return errors.Join(dsv4.ErrDesynchronized, fmt.Errorf("metal speculative verifier close frontier: %w", err))
	}
	v.frontier = nil
	v.cleanupFailed = false
	return nil
}

func (v *metalSpeculativeVerifier) VerifyBatch(seed int, drafts []int, pos int) ([]int, error) {
	if err := v.valid(); err != nil {
		return nil, err
	}
	if v.cleanupFailed {
		return nil, dsv4.ErrDesynchronized
	}
	if len(drafts) == 0 || len(drafts) > dsv4.DefaultSpeculativeLimits.MaxDrafts {
		return nil, fmt.Errorf("Metal speculative verifier: invalid draft count %d", len(drafts))
	}
	if v.frontier != nil {
		if err := v.releaseFrontier(); err != nil {
			return nil, err
		}
	}
	frontier, err := v.state.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("Metal speculative verifier snapshot: %w", err)
	}
	v.frontier = frontier
	v.seed, v.pos = seed, pos
	v.drafts = append(v.drafts[:0], drafts...)
	v.lastGrouped = false
	v.rowLogits = nil
	if os.Getenv("DS4_METAL_SPECULATIVE_GROUPED") == "1" {
		v.lastGrouped = true
		first, err := metalSpeculativeFrontier(v.CurrentLogits())
		if err != nil {
			return nil, v.abort(err)
		}
		rows, err := v.state.runtime.evalGroupedWithHooks(drafts, pos, v.groupedHooks)
		if err != nil {
			return nil, v.abort(err)
		}
		v.rowLogits, err = detachMetalSpeculativeRows(rows, len(drafts))
		if err != nil {
			return nil, v.abort(err)
		}
		tops := make([]int, len(drafts))
		v.rowEpoch = v.state.runtime.speculativeEpoch
		tops[0] = first
		for i := 1; i < len(drafts); i++ {
			top, err := metalSpeculativeFrontier(rows[i-1])
			if err != nil {
				return nil, v.abort(err)
			}
			tops[i] = top
		}
		return tops, nil
	}
	tops := make([]int, len(drafts))
	for i, draft := range drafts {
		logits := v.CurrentLogits()
		top, err := metalSpeculativeFrontier(logits)
		if err != nil {
			return nil, v.abort(err)
		}
		tops[i] = top
		if err := v.adapter.Eval(v.state, draft, pos+i); err != nil {
			return nil, v.abort(err)
		}
	}
	v.rowEpoch = v.state.runtime.speculativeEpoch
	return tops, nil
}

func metalSpeculativeFrontier(logits []float32) (int, error) {
	if len(logits) != Flash.Vocab {
		return 0, fmt.Errorf("metal speculative verifier: logits shape %d", len(logits))
	}
	if err := checkFiniteLogits("metal speculative frontier", logits); err != nil {
		return 0, err
	}
	return metalSpeculativeArgmax(logits), nil
}

func metalSpeculativeArgmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

func (v *metalSpeculativeVerifier) CommitPrefix(n int) error {
	if err := v.valid(); err != nil {
		return err
	}
	if v.cleanupFailed {
		return dsv4.ErrDesynchronized
	}
	if v.frontier == nil || v.rowEpoch != v.state.runtime.speculativeEpoch || v.state.runtime.pos != v.pos+len(v.drafts) {
		return dsv4.ErrUnimplemented
	}
	if n < 0 || n > len(v.drafts) {
		return fmt.Errorf("Metal speculative verifier: invalid accepted prefix %d", n)
	}
	if err := v.state.Restore(v.frontier); err != nil {
		return v.abort(fmt.Errorf("Metal speculative verifier restore frontier: %w", err))
	}
	for i := 0; i < n; i++ {
		if err := v.adapter.Eval(v.state, v.drafts[i], v.pos+i); err != nil {
			return v.abort(fmt.Errorf("Metal speculative verifier replay draft %d: %w", i, err))
		}
	}
	v.drafts = nil
	v.rowLogits = nil
	return v.releaseFrontier()
}

func (v *metalSpeculativeVerifier) abort(err error) error {
	v.rowLogits = nil
	if v.frontier != nil {
		if restoreErr := v.state.Restore(v.frontier); restoreErr != nil {
			poison := errors.Join(dsv4.ErrDesynchronized, restoreErr)
			v.state.runtime.failed = errors.Join(v.state.runtime.failed, poison)
			err = errors.Join(err, poison)
		}
		err = errors.Join(err, v.releaseFrontier())
	}
	v.drafts = nil
	return err
}
