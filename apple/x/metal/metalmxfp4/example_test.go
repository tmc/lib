package metalmxfp4_test

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalmxfp4"
)

func Example() {
	src := []float32{
		0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
		-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
		0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0,
		-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
	}

	packed, scales, err := metalmxfp4.Quantize(src, 32)
	if err != nil {
		fmt.Printf("quantize error: %v\n", err)
		return
	}

	dst := make([]float32, len(src))
	if err := metalmxfp4.Dequantize(packed, scales, 32, dst); err != nil {
		fmt.Printf("dequantize error: %v\n", err)
		return
	}

	fmt.Printf("Original: %d, Packed: %d, Scales: %d\n", len(src), len(packed), len(scales))
	fmt.Printf("Dequantized first 4: [%.1f, %.1f, %.1f, %.1f]\n", dst[0], dst[1], dst[2], dst[3])

	// Output:
	// Original: 32, Packed: 16, Scales: 1
	// Dequantized first 4: [0.0, 0.5, 1.0, 1.5]
}
