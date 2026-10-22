package dsv4tools

import (
	"fmt"
	"io"
	"math"
	"sort"
)

const maxDiagnosticValues = 1 << 20

// StatValue identifies one of the largest-magnitude entries in a vector.
type StatValue struct {
	Index int
	Value float32
}

// VectorStats is the bounded host representation used by head/first-token
// diagnostics. It does not imply that a backend or model was exercised.
type VectorStats struct {
	Min float32
	Max float32
	RMS float64
	Top []StatValue
}

// SummarizeVector computes deterministic min, max, RMS, and top-magnitude
// entries. Non-finite and oversized vectors are rejected.
func SummarizeVector(values []float32) (VectorStats, error) {
	if len(values) == 0 {
		return VectorStats{}, fmt.Errorf("summarize vector: empty input")
	}
	if len(values) > maxDiagnosticValues {
		return VectorStats{}, fmt.Errorf("summarize vector: %d values exceeds limit", len(values))
	}
	min, max := values[0], values[0]
	var sum float64
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return VectorStats{}, fmt.Errorf("summarize vector: value %d is non-finite", i)
		}
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
		sum += float64(value) * float64(value)
	}
	top := make([]StatValue, len(values))
	for i, value := range values {
		top[i] = StatValue{Index: i, Value: value}
	}
	sort.Slice(top, func(i, j int) bool {
		ai, aj := math.Abs(float64(top[i].Value)), math.Abs(float64(top[j].Value))
		if ai != aj {
			return ai > aj
		}
		return top[i].Index < top[j].Index
	})
	if len(top) > 8 {
		top = top[:8]
	}
	return VectorStats{Min: min, Max: max, RMS: math.Sqrt(sum / float64(len(values))), Top: top}, nil
}

// WriteVectorStats writes the upstream-compatible one-line summary followed
// by up to eight indexed top-magnitude entries.
func WriteVectorStats(w io.Writer, name string, values []float32) error {
	if w == nil {
		return fmt.Errorf("write vector stats: nil writer")
	}
	if name == "" {
		return fmt.Errorf("write vector stats: empty name")
	}
	stats, err := SummarizeVector(values)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "%s: min=%.9g max=%.9g rms=%.9g\n", name, stats.Min, stats.Max, stats.RMS); err != nil {
		return fmt.Errorf("write vector stats: %w", err)
	}
	for _, entry := range stats.Top {
		if _, err := fmt.Fprintf(w, "  top[%d]=%.9g\n", entry.Index, entry.Value); err != nil {
			return fmt.Errorf("write vector stats: %w", err)
		}
	}
	return nil
}
