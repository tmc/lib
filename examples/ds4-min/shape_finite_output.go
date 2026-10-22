package main

import (
	"fmt"
	"math"
)

func checkShapeFiniteOutput(values []float32, expected int) error {
	if expected <= 0 || len(values) != expected {
		return fmt.Errorf("cuda shape forward: output count %d, want %d", len(values), expected)
	}
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("cuda shape forward: nonfinite output at %d", i)
		}
	}
	return nil
}
