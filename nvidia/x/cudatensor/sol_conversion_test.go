package cudatensor

import (
	"errors"
	"math"
	"math/big"
	"testing"
)

func TestComputeSOLBoundsFrequencyConversion(t *testing.T) {
	tests := []struct {
		name      string
		macs      uint64
		macRate   float64
		frequency float64
		wantError bool
	}{
		{"ordinary", 1000000000000000000, 1, 2.5, false},
		{"frequency conversion overflow", 1000000000000000000, 1, 1e300, false},
		{"cycles divided by frequency overflow", 1000000000, 1, 1e-300, false},
		{"subnormal duration", 1, 1, 1e308, false},
		{"duration overflow", 1000000000000000000, 1, math.SmallestNonzeroFloat64, true},
		{"duration underflow", 0, 1, 1e308, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			elements := uint64(1)
			bytesPerElement := 1.0
			if tt.name == "duration underflow" {
				bytesPerElement = math.SmallestNonzeroFloat64
			}
			g, err := CanonicalizeRanks(&Graph{Nodes: []EinsumNode{
				{ID: "input", OutputRank: 1, OutputTensors: []TensorInfo{{Name: "x", Elements: elements}}},
				{ID: "op", Equation: "A->A", Inputs: []string{"input"}, MACs: tt.macs,
					InputTensors:  []TensorInfo{{Name: "x", Elements: elements}},
					OutputTensors: []TensorInfo{{Name: "y", Elements: elements}}},
			}})
			if err != nil {
				t.Fatal(err)
			}
			report, err := ComputeSOLBounds(g, Ceiling{
				MACsPerCycle: tt.macRate, BytesPerCycle: 1,
				FrequencyGHz: tt.frequency, BytesPerElement: bytesPerElement,
			})
			if tt.wantError {
				if !errors.Is(err, ErrSOLOverflow) || report != (SOLReport{}) {
					t.Fatalf("ComputeSOLBounds = %+v, %v; want zero report and ErrSOLOverflow", report, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			// Exact rational arithmetic on the supplied float64 values avoids
			// sharing the implementation's division or exponent ordering.
			compute := new(big.Rat).SetUint64(tt.macs)
			compute.Quo(compute, new(big.Rat).SetFloat64(tt.macRate))
			memory := new(big.Rat).SetInt64(2)
			if compute.Cmp(memory) < 0 {
				compute = memory
			}
			rate := new(big.Rat).SetFloat64(tt.frequency)
			rate.Mul(rate, new(big.Rat).SetInt64(1000000000))
			want, _ := new(big.Rat).Quo(compute, rate).Float64()
			for name, got := range map[string]float64{
				"Unfused": report.Unfused, "Fused": report.Fused,
				"UnfusedWholeGraph": report.UnfusedWholeGraph,
				"FusedWholeGraph":   report.FusedWholeGraph,
				"FusedPrefetched":   report.FusedPrefetched,
			} {
				// Allow two rounding steps, including at the subnormal boundary.
				ulp := math.Nextafter(want, math.Inf(1)) - want
				if got <= 0 || math.IsInf(got, 0) || math.IsNaN(got) || math.Abs(got-want) > 2*ulp {
					t.Errorf("%s = %.17g, want %.17g (within two ulps)", name, got, want)
				}
			}
			if report.UnfusedBytes != 2 || report.FusedBytes != 2 || report.FusedPrefetchedBytes != 2 {
				t.Errorf("byte counts changed: %+v", report)
			}
		})
	}
}
