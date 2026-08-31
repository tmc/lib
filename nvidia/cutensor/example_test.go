package cutensor_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cutensor"
)

// Example creates a cuTENSOR handle and describes a rank-2 tensor.
// It returns early when the cuTENSOR library is not available.
// Building an operation from descriptors (CreateContraction, CreatePlan,
// Contract) additionally needs a ComputeDescriptor for the desired
// precision.
func Example() {
	h, err := cutensor.Create()
	if err != nil {
		fmt.Println("cuTENSOR unavailable:", err)
		return
	}
	defer h.Destroy()
	fmt.Println("cuTENSOR version:", cutensor.GetVersion())

	// A 64x32 float32 tensor with dense (nil) strides, 128-byte aligned.
	extent := []int64{64, 32}
	// DataType 0 is CUDA_R_32F (library_types.h).
	desc, err := h.CreateTensorDescriptor(2, &extent[0], nil,
		cutensor.DataType(0), 128)
	if err != nil {
		fmt.Println("tensor descriptor:", err)
		return
	}
	defer desc.Destroy()
	fmt.Println("tensor descriptor ready")
}
