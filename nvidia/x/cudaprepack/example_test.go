// SPDX-License-Identifier: Apache-2.0

package cudaprepack_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprepack"
)

func Example() {
	// Pre-pack a 2D weight matrix for Blackwell sm_121a TMA transfer
	transformer, err := cudaprepack.NewLayoutTransformer(
		cudaprepack.ArchSM121a,
		cudaprepack.TypeFP8_E4M3,
		cudaprepack.WithAlignment(16),
	)
	if err != nil {
		panic(err)
	}

	rawWeights := make([]byte, 64*100) // 64 rows, 100 columns (1 byte per FP8)
	tensor, err := transformer.Transform2D(rawWeights, 64, 100)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Arch: %s\n", tensor.Arch)
	fmt.Printf("Rank: %d\n", tensor.BoxDesc.Rank)
	fmt.Printf("RowStride: %d bytes\n", tensor.BoxDesc.GlobalStrides[0])
	fmt.Printf("BoxDims: [%d, %d]\n", tensor.BoxDesc.BoxDims[0], tensor.BoxDesc.BoxDims[1])

	// Output:
	// Arch: sm_121a
	// Rank: 2
	// RowStride: 112 bytes
	// BoxDims: [112, 32]
}

func ExampleCompress2to4() {
	// Compress a matrix row with 2:4 structured sparsity for sm_100a tcgen05
	dense := []float32{1.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 2.0}
	sparse, meta, err := cudaprepack.Compress2to4(dense, 1, 8)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Sparse elements: %d\n", len(sparse))
	fmt.Printf("Metadata bytes: %d\n", len(meta))
	fmt.Printf("Metadata byte 0: 0x%02X\n", meta[0])

	// Output:
	// Sparse elements: 4
	// Metadata bytes: 1
	// Metadata byte 0: 0xD8
}

func ExamplePackMXFP4() {
	// Quantize float32 weights into 128-byte hardware-aligned MXFP4 tiles
	src := make([]float32, 32*64)
	for i := range src {
		src[i] = float32((i % 7) - 3)
	}

	tensor, err := cudaprepack.PackMXFP4(src, 32, 64, 32)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Type: %s\n", tensor.Type)
	fmt.Printf("Payload bytes: %d\n", len(tensor.Payload))
	fmt.Printf("Scale factors: %d\n", len(tensor.Scales)/2)

	// Output:
	// Type: MXFP4
	// Payload bytes: 1024
	// Scale factors: 64
}
