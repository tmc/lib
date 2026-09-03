// SPDX-License-Identifier: Apache-2.0

package metalquant_test

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalquant"
)

func Example() {
	values := make([]float32, metalquant.BlockSize)
	values[0] = -4
	values[16] = 3.5
	data, err := metalquant.QuantizeQ4_0(values)
	if err != nil {
		panic(err)
	}
	decoded, err := metalquant.DequantizeQ4_0(data, len(values))
	if err != nil {
		panic(err)
	}
	fmt.Printf("bytes=%d values=%d endpoints=%.1f,%.1f\n", len(data), len(decoded), decoded[0], decoded[16])
	// Output: bytes=18 values=32 endpoints=-4.0,3.5
}
