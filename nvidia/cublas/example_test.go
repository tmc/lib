package cublas_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
)

// Example creates a cuBLAS handle and queries the library version.
// It returns early when the cuBLAS library is not available.
func Example() {
	h, err := cublas.Create()
	if err != nil {
		fmt.Println("cuBLAS unavailable:", err)
		return
	}
	defer h.Close()

	major, err := cublas.GetProperty(cublas.MajorVersion)
	if err != nil {
		fmt.Println("property:", err)
		return
	}
	fmt.Println("cuBLAS major version:", major)
}

// ExampleHandle_DgetriBatched inverts a batch of n×n matrices. DgetrfBatched
// factors each matrix in place; DgetriBatched takes the same arguments and
// writes each inverse to a separate matrix. A nonzero info entry marks a
// singular matrix. It needs a current CUDA context.
func ExampleHandle_DgetriBatched() {
	h, err := cublas.Create()
	if err != nil {
		fmt.Println("cuBLAS unavailable:", err)
		return
	}
	defer h.Close()

	const n, batch = 4, 16
	// a and c are device arrays of batch pointers to column-major n×n
	// matrices; pivots holds n*batch int32s and info holds batch int32s.
	var a, c, pivots, info driver.DevicePtr

	if err := h.DgetrfBatched(n, a, n, pivots, info, batch); err != nil {
		fmt.Println("getrf:", err)
		return
	}
	if err := h.DgetriBatched(n, a, n, pivots, c, n, info, batch); err != nil {
		fmt.Println("getri:", err)
		return
	}
	status := make([]int32, batch)
	if err := info.CopyToHost(status); err != nil {
		fmt.Println("copy info:", err)
		return
	}
	for i, s := range status {
		if s != 0 {
			fmt.Printf("matrix %d is singular: U(%d,%d) = 0\n", i, s, s)
		}
	}
}
