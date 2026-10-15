package cudaprepack_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprepack"
)

func ExampleDequantizeFP8() {
	p, err := cudaprepack.QuantizeFP8([]float32{1, -2}, 1, 2, cudaprepack.TypeFP8_E4M3)
	if err != nil {
		fmt.Println(err)
		return
	}
	dst := make([]float32, 2)
	if err := cudaprepack.DequantizeFP8(p, dst); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(dst)
	// Output: [1 -2]
}

func ExampleDequantizeMXFP4() {
	src := make([]float32, 32)
	src[0] = 6
	p, err := cudaprepack.PackMXFP4(src, 1, 32, 32)
	if err != nil {
		fmt.Println(err)
		return
	}
	dst := make([]float32, 32)
	if err := cudaprepack.DequantizeMXFP4(p, 32, dst); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(dst[0], len(dst))
	// Output: 6 32
}

func ExamplePrepackedTensor_iq2Blocks() {
	raw := make([]byte, 66)
	raw[2] = 7
	p, err := cudaprepack.RepackIQ2_XXS(raw, 1, 256)
	if err != nil {
		fmt.Println(err)
		return
	}
	decoded, err := cudaprepack.UnpackIQ2_XXS(p)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(decoded), decoded[2])
	// Output: 66 7
}
