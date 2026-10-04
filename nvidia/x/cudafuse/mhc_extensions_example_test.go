package cudafuse_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudafuse"
)

func ExampleMHCBF16Source() {
	s, err := cudafuse.MHCBF16Source(cudafuse.MHCConfig{Batch: 1, Width: 33, Repeat: 10, Epsilon: 1e-6})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, k := range s {
		fmt.Println(k.Name)
	}
	// Output:
	// mhc_sinkhorn_fp32
	// mhc_post_bf16
}

func ExampleMHCFusedSource() {
	s, err := cudafuse.MHCFusedSource(cudafuse.MHCConfig{Batch: 1, Width: 33, Repeat: 10, Epsilon: 1e-6})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(s.Name)
	// Output: mhc_fused_fp32
}

func ExampleExecuteMHCBF16() {
	c := cudafuse.MHCConfig{Batch: 1, Width: 33, Repeat: 10, Epsilon: 1e-6}
	_, err := cudafuse.ExecuteMHCBF16(context.Background(), nil, c, cudafuse.MHCBF16Inputs{})
	fmt.Println(err)
	// Output: cudafuse: nil mhc stream or device
}

func ExampleExecuteMHCFused() {
	c := cudafuse.MHCConfig{Batch: 1, Width: 33, Repeat: 10, Epsilon: 1e-6}
	_, err := cudafuse.ExecuteMHCFused(context.Background(), nil, c, cudafuse.MHCInputs{})
	fmt.Println(err)
	// Output: cudafuse: nil mhc stream or device
}
