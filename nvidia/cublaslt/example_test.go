package cublaslt_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
)

// Example runs a single-precision matrix multiply through cublasLtMatmul.
// It returns early when the cuBLASLt library is not available. The device
// buffers dA..dD are CUDA device pointers, e.g. from the driver package's
// MemAlloc.
func Example() {
	h, err := cublaslt.Create()
	if err != nil {
		fmt.Println("cuBLASLt unavailable:", err)
		return
	}
	defer h.Destroy()

	const m, n, k = 128, 128, 128
	// DataType 0 is CUDA_R_32F (library_types.h).
	r32f := cublaslt.DataType(0)

	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, r32f)
	if err != nil {
		fmt.Println("matmul desc:", err)
		return
	}
	defer desc.Destroy()

	aLayout, _ := cublaslt.NewMatrixLayout(r32f, m, k, m)
	bLayout, _ := cublaslt.NewMatrixLayout(r32f, k, n, k)
	cLayout, _ := cublaslt.NewMatrixLayout(r32f, m, n, m)
	defer aLayout.Destroy()
	defer bLayout.Destroy()
	defer cLayout.Destroy()

	var dA, dB, dC uintptr // device allocations, m*k, k*n, m*n floats
	alpha, beta := float32(1), float32(0)
	err = h.Matmul(desc,
		uintptr(unsafe.Pointer(&alpha)), dA, aLayout, dB, bLayout,
		uintptr(unsafe.Pointer(&beta)), dC, cLayout, dC, cLayout,
		nil, 0, 0, cublaslt.Stream(0))
	if err != nil {
		fmt.Println("matmul:", err)
		return
	}
	fmt.Println("matmul enqueued")
}
