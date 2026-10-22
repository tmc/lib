package main

import (
	"math"
	"testing"
)

// Float64 oracle independent of the CUDA block/reduction layout. Its fixture
// represents one query with every raw/closed compressed row visible. Masks are
// additive (0 or -Inf); the sink contributes only to the denominator.
func shapeAttentionReference(q, raw, compressed, sinks, mask []float32, heads, width int) []float64 {
	rawRows, compRows := len(raw)/width, len(compressed)/width
	out := make([]float64, heads*width)
	for h := 0; h < heads; h++ {
		scores := make([]float64, rawRows+compRows)
		maximum := float64(sinks[h])
		for row := range scores {
			values := raw
			at := row
			add := 0.0
			if row >= rawRows {
				values, at = compressed, row-rawRows
				if mask != nil {
					add = float64(mask[at])
				}
			}
			dot := 0.0
			for d := 0; d < width; d++ {
				dot += float64(q[h*width+d]) * float64(values[at*width+d])
			}
			scores[row] = dot/math.Sqrt(float64(width)) + add
			maximum = math.Max(maximum, scores[row])
		}
		denominator := math.Exp(float64(sinks[h]) - maximum)
		for i := range scores {
			scores[i] = math.Exp(scores[i] - maximum)
			denominator += scores[i]
		}
		for d := 0; d < width; d++ {
			value := 0.0
			for row, p := range scores {
				if row < rawRows {
					value += float64(raw[row*width+d]) * p
				} else {
					value += float64(compressed[(row-rawRows)*width+d]) * p
				}
			}
			out[h*width+d] = value / denominator
		}
	}
	return out
}

func shapeAttentionFixture(heads int) (q, raw, compressed, sinks, mask []float32) {
	const width = 512
	q, raw, compressed, sinks, mask = make([]float32, heads*width), make([]float32, 3*width), make([]float32, 2*width), make([]float32, heads), []float32{0, float32(math.Inf(-1))}
	for i := range q {
		q[i] = float32((i*11)%29-14) / 64
	}
	for i := range raw {
		raw[i] = float32((i*7)%23-11) / 32
	}
	for i := range compressed {
		compressed[i] = float32((i*13)%31-15) / 32
	}
	for i := range sinks {
		sinks[i] = float32(i%9-4) / 8
	}
	return
}

func TestShapeAttentionReference(t *testing.T) {
	q, raw, comp, sinks, mask := shapeAttentionFixture(128)
	values := shapeAttentionReference(q, raw, comp, sinks, mask, 128, 512)
	if len(values) != 65536 {
		t.Fatal("reference output truncated")
	}
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatal("nonfinite reference")
		}
	}
	// A masked cache row must not affect a single coordinate in any Pro head.
	for i := 512; i < len(comp); i++ {
		comp[i] *= 16
	}
	other := shapeAttentionReference(q, raw, comp, sinks, mask, 128, 512)
	for i, v := range values {
		if v != other[i] {
			t.Fatalf("masked value affected output%d", i)
		}
	}
	// Include a real sink control: dropping its mass changes the output.
	for i := range sinks {
		sinks[i] = float32(math.Inf(-1))
	}
	other = shapeAttentionReference(q, raw, comp, sinks, mask, 128, 512)
	changed := false
	for i, v := range values {
		changed = changed || v != other[i]
	}
	if !changed {
		t.Fatal("sink control insensitive")
	}
}
