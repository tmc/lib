//go:build darwin && arm64

package main

import (
	"fmt"
	"math"
	"testing"
)

// oracleDrift records every element, including the values excluded from the
// historical relativeDrift metric. These are diagnostics, not new tolerances.
type oracleDrift struct {
	count, smallCount int
	maxAbs, rms       float64
	smallMaxAbs       float64
}

func measureOracleDrift(got, want []float32) (oracleDrift, error) {
	var d oracleDrift
	if err := oracleComparisonShape(len(got), len(want)); err != nil {
		return d, err
	}
	var squares float64
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			return oracleDrift{}, fmt.Errorf("element %d: non-finite values %g, %g", i, g, w)
		}
		// Subtract in float64 so opposite finite float32 extremes remain finite.
		delta := math.Abs(g - w)
		d.maxAbs = max(d.maxAbs, delta)
		squares += delta * delta
		if math.Abs(w) <= 0.1 {
			d.smallCount++
			d.smallMaxAbs = max(d.smallMaxAbs, delta)
		}
	}
	d.count = len(got)
	d.rms = math.Sqrt(squares / float64(d.count))
	return d, nil
}

func TestOracleDrift(t *testing.T) {
	for _, tt := range []struct {
		name      string
		got, want []float32
		wantError bool
	}{
		{"empty", nil, nil, true},
		{"truncated", []float32{1}, []float32{1, 2}, true},
		{"candidate nan", []float32{float32(math.NaN())}, []float32{0}, true},
		{"oracle nan", []float32{0}, []float32{float32(math.NaN())}, true},
		{"equal negative infinity", []float32{float32(math.Inf(-1))}, []float32{float32(math.Inf(-1))}, true},
		{"positive infinity", []float32{float32(math.Inf(1))}, []float32{0}, true},
		{"finite extremes", []float32{math.MaxFloat32}, []float32{-math.MaxFloat32}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			d, err := measureOracleDrift(tt.got, tt.want)
			if (err != nil) != tt.wantError {
				t.Fatalf("error = %v, want error %v", err, tt.wantError)
			}
			if err == nil && (math.IsInf(d.rms, 0) || math.IsNaN(d.rms)) {
				t.Fatalf("non-finite diagnostic: %+v", d)
			}
		})
	}
	t.Run("error hidden by relative metric", func(t *testing.T) {
		want := []float32{0, 0.0625, -0.0625, 4}
		got := []float32{1, 0.0625, -0.0625, 4}
		if drift := relativeDrift(t, got, want); drift != 0 {
			t.Fatalf("historical relative drift = %g, want 0", drift)
		}
		d, err := measureOracleDrift(got, want)
		if err != nil {
			t.Fatal(err)
		}
		if d.count != 4 || d.smallCount != 3 || d.maxAbs != 1 || d.smallMaxAbs != 1 || d.rms != 0.5 {
			t.Fatalf("injected error not measured: %+v", d)
		}
	})
}
