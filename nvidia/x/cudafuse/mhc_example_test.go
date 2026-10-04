package cudafuse_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudafuse"
)

func ExampleMHCSource() {
	c := cudafuse.MHCConfig{Batch: 1, Width: 33, Repeat: 10, Epsilon: 1e-6}
	kernels, err := cudafuse.MHCSource(c)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, k := range kernels {
		fmt.Println(k.Name, len(k.Inputs), len(k.Outputs))
	}
	// Output:
	// mhc_sinkhorn_fp32 1 1
	// mhc_post_fp32 4 1
}

func ExampleExecuteMHC() {
	c := cudafuse.MHCConfig{Batch: 1, Width: 33, Repeat: 10, Epsilon: 1e-6}
	_, err := cudafuse.ExecuteMHC(context.Background(), nil, c, cudafuse.MHCInputs{})
	fmt.Println(err)
	// Output: cudafuse: nil mhc stream or device
}

func ExampleCompileMHC() {
	// Invalid dimensions are refused before accessing the compiler library.
	_, err := cudafuse.CompileMHC(cudafuse.MHCConfig{}, 12, 1)
	fmt.Println(err)
	// Output: cudafuse: invalid mhc dimensions, repeat or epsilon
}
