package cudalmvet

import (
	"fmt"
	"math"
)

// LayerDiff stores error statistics between GPU output and CPU reference tensor.
type LayerDiff struct {
	Name       string
	Length     int
	MaxAbsDiff float64
	MeanDiff   float64
	CosineSim  float64
	Passed     bool
}

// VetLayer compares a computed float32 slice against reference output under max allowed tolerance.
func VetLayer(name string, got, want []float32, tol float64) (*LayerDiff, error) {
	if len(got) != len(want) {
		return nil, fmt.Errorf("cudalmvet: length mismatch (%d vs %d) for layer %s", len(got), len(want), name)
	}

	maxAbs := 0.0
	sumDiff := 0.0
	dotProduct := 0.0
	normGot := 0.0
	normWant := 0.0

	for i := range got {
		g := float64(got[i])
		w := float64(want[i])
		diff := math.Abs(g - w)

		if diff > maxAbs {
			maxAbs = diff
		}
		sumDiff += diff
		dotProduct += g * w
		normGot += g * g
		normWant += w * w
	}

	meanDiff := sumDiff / float64(len(got))
	cosineSim := 1.0
	denom := math.Sqrt(normGot) * math.Sqrt(normWant)
	if denom > 0 {
		cosineSim = dotProduct / denom
	}

	passed := maxAbs <= tol

	return &LayerDiff{
		Name:       name,
		Length:     len(got),
		MaxAbsDiff: maxAbs,
		MeanDiff:   meanDiff,
		CosineSim:  cosineSim,
		Passed:     passed,
	}, nil
}

// Report holds the cumulative vetting results across all model layers.
type Report struct {
	Layers       []LayerDiff
	TotalPassed  int
	TotalFailed  int
	WorstMaxDiff float64
}

// NewReport constructs a vetting report aggregating diffs.
func NewReport(diffs ...LayerDiff) *Report {
	passed := 0
	failed := 0
	worst := 0.0

	for _, d := range diffs {
		if d.Passed {
			passed++
		} else {
			failed++
		}
		if d.MaxAbsDiff > worst {
			worst = d.MaxAbsDiff
		}
	}

	return &Report{
		Layers:       diffs,
		TotalPassed:  passed,
		TotalFailed:  failed,
		WorstMaxDiff: worst,
	}
}
