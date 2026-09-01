// SPDX-License-Identifier: Apache-2.0

package quant_test

import (
	"fmt"

	"github.com/tmc/lib/lm/quant"
)

func ExampleFloat16ToFloat32() {
	f := quant.Float16ToFloat32(0x3C00) // 1.0 in FP16
	fmt.Printf("Decoded FP16: %.1f\n", f)
	// Output:
	// Decoded FP16: 1.0
}
