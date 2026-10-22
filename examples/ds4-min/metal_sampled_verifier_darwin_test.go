//go:build darwin && arm64

package main

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestMetalSampledRows(t *testing.T) {
	rows := [][]float32{make([]float32, Flash.Vocab), make([]float32, Flash.Vocab)}
	rows[0][3], rows[1][4] = 7, 9
	got, err := detachMetalSpeculativeRows(rows, 2)
	if err != nil {
		t.Fatal(err)
	}
	rows[0][3] = 0
	if got[0][3] != 7 || got[1][4] != 9 {
		t.Fatal("full rows were aliased or shifted")
	}
	for _, invalid := range [][][]float32{nil, {make([]float32, 1)}, {make([]float32, Flash.Vocab)}} {
		if _, err := detachMetalSpeculativeRows(invalid, 2); err == nil {
			t.Fatal("accepted missing or malformed rows")
		}
	}
	rows[1][0] = float32(math.NaN())
	if _, err := detachMetalSpeculativeRows(rows, 2); err == nil {
		t.Fatal("accepted nonfinite final row")
	}
}

func TestMetalSampledVerifierOwnership(t *testing.T) {
	a := &metalStateAdapter{}
	s := &metalRequestState{owner: a, runtime: &metalRuntime{speculativeEpoch: 4}}
	values := make([]float32, Flash.Vocab)
	values[0], values[1] = 2, 3
	raw := &metalSpeculativeVerifier{adapter: a, state: s, frontier: &metalSnapshot{}, lastGrouped: true, rowEpoch: 4, rowLogits: [][]float32{values}}
	v := &metalSampledVerifier{raw}
	if !v.BoundTo(s) || v.BoundTo(&metalRequestState{owner: a}) {
		t.Fatal("lost exact request identity")
	}
	row, err := v.RowLogits(0)
	if err != nil {
		t.Fatal(err)
	}
	row[0] = 9
	if raw.rowLogits[0][0] != 2 {
		t.Fatal("readout aliases cached row")
	}
	s.runtime.speculativeEpoch++
	if _, err := v.RowLogits(0); !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatal("published stale row")
	}
	t.Setenv("DS4_METAL_SPECULATIVE_GROUPED", "0")
	before := s.runtime.speculativeEpoch
	if _, err := v.VerifyBatch(1, []int{2}, 0); !errors.Is(err, dsv4.ErrUnimplemented) || s.runtime.speculativeEpoch != before || raw.rowLogits != nil {
		t.Fatalf("scalar route mutated request or published rows: %v", err)
	}
}
