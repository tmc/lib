package metalx_test

import (
	"fmt"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

// decodeKernels names the pipeline states one decode step dispatches. The
// "metal" tag gives the kernel's name in the compiled library; an untagged
// field is looked up under its own name.
type decodeKernels struct {
	Embed apple.MTLComputePipelineState `metal:"EmbeddingLookup"`
	Norm  apple.MTLComputePipelineState `metal:"ComputeRMSNorm"`
	GEMV  apple.MTLComputePipelineState `metal:"FastGEMV"`
}

// ExampleBind resolves a library's kernels in one step, so a renamed kernel
// fails at startup with the field that named it rather than at dispatch.
func ExampleBind() {
	dev, err := metal.NewDevice()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer dev.Close()

	lib, err := dev.CompileLibrary(decodeMSL)
	if err != nil {
		fmt.Println(err)
		return
	}

	var k decodeKernels
	if err := metalx.Bind(dev, lib, &k); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(k.Embed != nil, k.Norm != nil, k.GEMV != nil)
}

const decodeMSL = `
#include <metal_stdlib>
using namespace metal;

kernel void EmbeddingLookup(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) { out[id] = 0; }
kernel void ComputeRMSNorm(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) { out[id] = 0; }
kernel void FastGEMV(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) { out[id] = 0; }
`
