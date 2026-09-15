package metalmxfp4

import (
	"math"
	"testing"
)

func TestMetalMXFP4Roundtrip(t *testing.T) {
	groupSize := 32
	numElements := 128
	src := make([]float32, numElements)
	for i := range src {
		src[i] = float32(math.Cos(float64(i+1)*0.2)) * 4.0
	}

	packed, scales, err := Quantize(src, groupSize)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	dst := make([]float32, numElements)
	if err := Dequantize(packed, scales, groupSize, dst); err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	for i := 0; i < numElements; i++ {
		diff := float32(math.Abs(float64(src[i] - dst[i])))
		if diff > 1.5 {
			t.Errorf("element %d diff %f too large", i, diff)
		}
	}
}
