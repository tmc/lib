package cudagraph_test

import (
	"fmt"
	"math"
)

func checkPDLNumerical(want, got []float32) error {
	if len(want) != len(got) {
		return fmt.Errorf("length mismatch: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		diff := math.Abs(float64(want[i] - got[i]))
		if math.IsNaN(float64(want[i])) || math.IsNaN(float64(got[i])) || !(diff <= 1e-4) {
			return fmt.Errorf("numerical divergence at %d: want=%f got=%f diff=%e", i, want[i], got[i], diff)
		}
	}
	return nil
}
