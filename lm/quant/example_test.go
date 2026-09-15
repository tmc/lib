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

// Example_dequantizeQ2K shows [DequantizeQ2_K]. The underscore in the function
// name keeps it from being written as ExampleDequantizeQ2_K.
func Example_dequantizeQ2K() {
	// One Q2_K super-block: d = 1.0, dmin = 1.0, every group scaled by 1
	// with a minimum of 2, and element 0 quantized to 3.
	block := make([]byte, 84)
	for i := range 16 {
		block[i] = 0x21
	}
	block[16] = 0x03
	block[80], block[81] = 0x00, 0x3C
	block[82], block[83] = 0x00, 0x3C

	out, err := quant.DequantizeQ2_K(block, quant.SuperBlockSize)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%.1f %.1f\n", out[0], out[1])
	// Output:
	// 1.0 -2.0
}

// Example_dequantizeIQ2XXS shows [DequantizeIQ2_XXS].
func Example_dequantizeIQ2XXS() {
	// One IQ2_XXS super-block with d = 1.0. Every codebook index and sign
	// index is zero, selecting grid entry 0x0808080808080808 unsigned, and
	// every group scale nibble is zero, giving 0.125 * 8.
	block := make([]byte, 66)
	block[0], block[1] = 0x00, 0x3C

	out, err := quant.DequantizeIQ2_XXS(block, quant.SuperBlockSize)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%.1f %.1f\n", out[0], out[255])
	// Output:
	// 1.0 1.0
}
