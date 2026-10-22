package dsv4cpu

import (
	"errors"
	"math"
)

// Attend evaluates raw or preselected compressed KV rows for each query head.
// KV is shared across heads and used as both key and value, matching DS4 MLA.
// The per-head sink contributes to the denominator but has no value vector.
// dst, query and rows must not overlap. Row selection and cache quantization
// are the caller's responsibility.
func Attend(dst, query, rows, sinks []float32, headDim int) error {
	if headDim <= 0 || len(sinks) == 0 || len(query)/headDim != len(sinks) || len(query)%headDim != 0 || len(dst) != len(query) || len(rows) == 0 || len(rows)%headDim != 0 {
		return errors.New("invalid attention shape")
	}
	n := len(rows) / headDim
	scores := make([]float32, n)
	scale := float32(1) / float32(math.Sqrt(float64(headDim)))
	for h, sink := range sinks {
		q := query[h*headDim : (h+1)*headDim]
		out := dst[h*headDim : (h+1)*headDim]
		max := sink
		for r := 0; r < n; r++ {
			var dot float32
			for d, v := range q {
				dot += float32(v * rows[r*headDim+d])
			}
			scores[r] = dot * scale
			if scores[r] > max {
				max = scores[r]
			}
		}
		clear(out)
		denom := float32(math.Exp(float64(sink - max)))
		for r, score := range scores {
			weight := float32(math.Exp(float64(score - max)))
			denom += weight
			for d := range out {
				out[d] += float32(rows[r*headDim+d] * weight)
			}
		}
		inv := float32(1) / denom
		for d := range out {
			out[d] *= inv
		}
	}
	return nil
}
