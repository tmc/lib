// SPDX-License-Identifier: Apache-2.0

package cudaspec_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaspec"
)

func ExampleComputeSeqlensQByConfidenceCPU() {
	prob := [][]float32{
		{0.9, 0.8, 0.5},
		{0.8, 0.5, 0.5},
		{0.7, 0.2, 0.1},
	}
	seqlensQ, cuSeqlensQ, err := cudaspec.ComputeSeqlensQByConfidenceCPU(prob, nil, 6, 3)
	if err != nil {
		panic(err)
	}
	fmt.Printf("seqlensQ=%v cuSeqlensQ=%v\n", seqlensQ, cuSeqlensQ)
	// Output:
	// seqlensQ=[3 2 1] cuSeqlensQ=[0 3 5 6]
}
