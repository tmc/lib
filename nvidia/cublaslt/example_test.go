package cublaslt_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/driver"
)

// Example runs a single-precision matrix multiply through cublasLtMatmul.
// It returns early when the cuBLASLt library is not available. The device
// buffers dA..dD are CUDA device pointers, e.g. from the driver package's
// MemAlloc. It has no Output comment, so it is compiled but not run: what it
// prints depends on the host's cuBLASLt.
func Example() {
	h, err := cublaslt.Create()
	if err != nil {
		fmt.Println("cuBLASLt unavailable:", err)
		return
	}
	defer h.Close()

	const m, n, k = 128, 128, 128

	desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
	if err != nil {
		fmt.Println("matmul desc:", err)
		return
	}
	defer desc.Close()

	aLayout, _ := cublaslt.NewMatrixLayout(cublaslt.R32F, m, k, m)
	bLayout, _ := cublaslt.NewMatrixLayout(cublaslt.R32F, k, n, k)
	cLayout, _ := cublaslt.NewMatrixLayout(cublaslt.R32F, m, n, m)
	defer aLayout.Close()
	defer bLayout.Close()
	defer cLayout.Close()

	var dA, dB, dC driver.DevicePtr // device allocations, m*k, k*n, m*n floats
	err = h.Matmul(desc,
		cublaslt.Float32Scalar(1), dA, aLayout, dB, bLayout,
		cublaslt.Float32Scalar(0), dC, cLayout, dC, cLayout,
		nil, 0, 0, 0)
	if err != nil {
		fmt.Println("matmul:", err)
		return
	}
	fmt.Println("matmul enqueued")
}
