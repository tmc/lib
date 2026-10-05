package cudahotswap

import (
	"fmt"
	"math"
)

// ReplayMetrics describes one FP32 output comparison. Exact compares bits,
// including the sign of zero. ULP distance treats both zeros as the same value.
// If a candidate contains nonfinite values, NonFinite counts them and the
// numeric metrics are nil. Reference values must be finite.
type ReplayMetrics struct {
	Elements  int      `json:"elements"`
	NonFinite int      `json:"non_finite"`
	Exact     bool     `json:"exact"`
	MaxAbs    *float64 `json:"max_abs,omitempty"`
	RMS       *float64 `json:"rms,omitempty"`
	MaxULP    *uint32  `json:"max_ulp,omitempty"`
}

// MeasureReplay compares equally sized, nonempty FP32 outputs. It reports
// measurements; the caller applies its declared tolerances. A nonfinite
// candidate never has an exact comparison or finite numeric metrics.
func MeasureReplay(got, want []float32) (ReplayMetrics, error) {
	if len(got) != len(want) || len(want) == 0 {
		return ReplayMetrics{}, fmt.Errorf("replay comparison requires equal nonempty outputs")
	}
	m := ReplayMetrics{Elements: len(want), Exact: true}
	var maxAbs, squared float64
	var maxULP uint32
	for i, w := range want {
		if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
			return ReplayMetrics{}, fmt.Errorf("replay reference element %d is not finite", i)
		}
		g := got[i]
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
			m.NonFinite++
			m.Exact = false
			continue
		}
		if math.Float32bits(g) != math.Float32bits(w) {
			m.Exact = false
		}
		abs := math.Abs(float64(g) - float64(w))
		maxAbs = max(maxAbs, abs)
		squared += abs * abs
		// Ordered finite FP32 magnitudes fit in int64 even across zero.
		distance := replayOrdered(g) - replayOrdered(w)
		if distance < 0 {
			distance = -distance
		}
		maxULP = max(maxULP, uint32(distance))
	}
	if m.NonFinite == 0 {
		rms := math.Sqrt(squared / float64(m.Elements))
		m.MaxAbs, m.RMS, m.MaxULP = &maxAbs, &rms, &maxULP
	}
	return m, nil
}

func replayOrdered(v float32) int64 {
	bits := math.Float32bits(v)
	magnitude := int64(bits & 0x7fffffff)
	if bits&0x80000000 != 0 {
		return -magnitude
	}
	return magnitude
}
