// SPDX-License-Identifier: Apache-2.0

package cudaquant_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaquant"
)

func ExampleQuantizeScaleUE8M0() {
	exp := cudaquant.QuantizeScaleUE8M0(448.0, 448.0)
	scale := cudaquant.DequantizeScaleUE8M0(exp)
	packed, err := cudaquant.PackUE8M0([]uint8{exp, exp + 1})
	if err != nil {
		panic(err)
	}
	unpacked, err := cudaquant.UnpackUE8M0(packed, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("exp=%d scale=%.1f unpacked=%v\n", exp, scale, unpacked)
	// Output:
	// exp=127 scale=1.0 unpacked=[127 128]
}
