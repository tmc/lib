//go:build darwin

package main

import (
	"math"
	"testing"
)

// The pinned position-0 capture stores -Inf in every attention score slot
// outside the live window, so an oracle comparison meets far more sentinels
// than finite values: 7168 of 8192 on layer 2 and 65024 of 65536 on layer 3.
// Rejecting them outright leaves those layers unmeasured, and accepting a
// sentinel that only one side produced would hide a real masking bug.
func TestMaskedComparison(t *testing.T) {
	ninf := float32(math.Inf(-1))
	nan := float32(math.NaN())
	for _, test := range []struct {
		name      string
		got, want []float32
		live      int
	}{
		{"live slot behind a mask", []float32{ninf, -0.5}, []float32{ninf, -0.5}, 1},
		{"candidate masked a live slot", []float32{ninf, ninf}, []float32{ninf, -0.5}, 0},
		{"candidate filled a masked slot", []float32{-0.5, -0.5}, []float32{ninf, -0.5}, 0},
		{"everything masked measures nothing", []float32{ninf, ninf}, []float32{ninf, ninf}, 0},
		{"positive infinity is not the sentinel", []float32{ninf, float32(math.Inf(1))}, []float32{ninf, float32(math.Inf(1))}, 0},
		{"nan in a live slot", []float32{ninf, nan}, []float32{ninf, -0.5}, 0},
		{"live slot outside tolerance", []float32{ninf, -0.6}, []float32{ninf, -0.5}, 0},
		{"all live", []float32{-0.5, 0}, []float32{-0.5, 0}, 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			live, err := maskedComparison(test.got, test.want, oracleAtol, oracleRtol)
			if test.live == 0 {
				if err == nil {
					t.Fatalf("maskedComparison = %d, nil, want an error", live)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if live != test.live {
				t.Errorf("maskedComparison = %d live, want %d", live, test.live)
			}
		})
	}
}
