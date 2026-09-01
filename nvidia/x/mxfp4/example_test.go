package mxfp4_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/mxfp4"
)

func Example() {
	// Original float32 activations
	weights := []float32{
		0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
		-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
		0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0,
		-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
	}

	packed, scales, err := mxfp4.Quantize(weights, 32)
	if err != nil {
		fmt.Printf("quantize error: %v\n", err)
		return
	}

	dst := make([]float32, len(weights))
	if err := mxfp4.Dequantize(packed, scales, 32, dst); err != nil {
		fmt.Printf("dequantize error: %v\n", err)
		return
	}

	fmt.Printf("Original len: %d, Packed bytes: %d, Scales: %d\n", len(weights), len(packed), len(scales))
	fmt.Printf("Dequantized first 4: [%.1f, %.1f, %.1f, %.1f]\n", dst[0], dst[1], dst[2], dst[3])

	// Output:
	// Original len: 32, Packed bytes: 16, Scales: 1
	// Dequantized first 4: [0.0, 0.5, 1.0, 1.5]
}
