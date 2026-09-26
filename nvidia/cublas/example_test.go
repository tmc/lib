package cublas_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cublas"
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
