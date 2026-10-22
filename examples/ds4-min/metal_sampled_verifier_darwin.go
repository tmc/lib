//go:build darwin && arm64

package main

import (
	"fmt"
	"os"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// metalSampledVerifier admits only the full-row grouped readout. It remains
// private until the sampled request constructor selects this capability.
type metalSampledVerifier struct {
	*metalSpeculativeVerifier
}

// metalSampledBackend is an explicit private capability view. The ordinary
// adapter continues to construct the greedy verifier used by native cycles.
type metalSampledBackend struct {
	*metalStateAdapter
}

var _ dsv4.SpeculativeBackend = (*metalSampledBackend)(nil)

func (a *metalSampledBackend) NewSpeculativeVerifier(state dsv4.State) (dsv4.SpeculativeVerifier, error) {
	if a == nil || a.metalStateAdapter == nil {
		return nil, dsv4.ErrClosed
	}
	if os.Getenv("DS4_METAL_SPECULATIVE_GROUPED") != "1" {
		return nil, dsv4.ErrUnimplemented
	}
	v, err := a.metalStateAdapter.NewSpeculativeVerifier(state)
	if err != nil {
		return nil, err
	}
	return &metalSampledVerifier{v.(*metalSpeculativeVerifier)}, nil
}

func (v *metalSampledVerifier) BoundTo(state dsv4.State) bool {
	return v != nil && v.metalSpeculativeVerifier != nil && !v.closed && !v.failed && !v.cleanupFailed && v.state != nil &&
		state == v.state && !v.state.closed && v.state.runtime != nil && v.adapter != nil &&
		!v.adapter.closed && v.state.owner == v.adapter && metalStatePoison(v.state.runtime) == nil
}

func (v *metalSampledVerifier) CurrentLogits() []float32 {
	if v == nil || v.metalSpeculativeVerifier == nil {
		return nil
	}
	return append([]float32(nil), v.metalSpeculativeVerifier.CurrentLogits()...)
}

func (v *metalSampledVerifier) VerifyBatch(seed int, drafts []int, position int) ([]int, error) {
	if v == nil || v.metalSpeculativeVerifier == nil || !v.BoundTo(v.state) {
		return nil, dsv4.ErrClosed
	}
	v.rowLogits = nil
	if os.Getenv("DS4_METAL_SPECULATIVE_GROUPED") != "1" ||
		admitMetalGroup(v.state.runtime, drafts, position) != nil {
		return nil, dsv4.ErrUnimplemented
	}
	return v.metalSpeculativeVerifier.VerifyBatch(seed, drafts, position)
}

func (v *metalSampledVerifier) RowLogits(row int) ([]float32, error) {
	if v == nil || v.metalSpeculativeVerifier == nil || !v.BoundTo(v.state) {
		return nil, dsv4.ErrClosed
	}
	if v.frontier == nil || !v.lastGrouped || v.rowEpoch != v.state.runtime.speculativeEpoch || row < 0 || row >= len(v.rowLogits) {
		return nil, dsv4.ErrUnimplemented
	}
	if len(v.rowLogits[row]) != Flash.Vocab {
		return nil, dsv4.ErrUnimplemented
	}
	return append([]float32(nil), v.rowLogits[row]...), nil
}

func (v *metalSampledVerifier) Restore(snapshot dsv4.Snapshot) error {
	if v == nil || v.metalSpeculativeVerifier == nil {
		return dsv4.ErrClosed
	}
	v.rowLogits = nil
	return v.metalSpeculativeVerifier.Restore(snapshot)
}

func detachMetalSpeculativeRows(rows [][]float32, count int) ([][]float32, error) {
	if count < 1 || len(rows) != count {
		return nil, fmt.Errorf("metal sampled verifier: invalid row count")
	}
	out := make([][]float32, count)
	for i, row := range rows {
		if len(row) != Flash.Vocab {
			return nil, fmt.Errorf("metal sampled verifier: invalid row %d extent", i)
		}
		if err := checkFiniteLogits("metal sampled row", row); err != nil {
			return nil, err
		}
		out[i] = append([]float32(nil), row...)
	}
	return out, nil
}
