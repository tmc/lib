package main

import (
	"math"
	"testing"
)

func TestShapeFiniteOutput(t *testing.T) {
	for _, width := range []int{4096, 7168} {
		values := make([]float32, width)
		if err := checkShapeFiniteOutput(values, width); err != nil {
			t.Fatal(err)
		}
		for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			for _, index := range []int{0, width / 2, width - 1} {
				values[index] = v
				if err := checkShapeFiniteOutput(values, width); err == nil {
					t.Fatalf("FINITE_OUTPUT_GUARD width=%d index=%d value=%g", width, index, v)
				}
				values[index] = 0
			}
		}
		if err := checkShapeFiniteOutput(values[:width-1], width); err == nil {
			t.Fatal("FINITE_OUTPUT_COUNT accepted truncated output")
		}
	}
	if err := checkShapeFiniteOutput(nil, 0); err == nil {
		t.Fatal("FINITE_OUTPUT_COUNT accepted missing output")
	}
}
