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
	defer h.Destroy()

	// LibraryPropertyType 0 is MAJOR_VERSION (library_types.h).
	major, err := cublas.GetProperty(cublas.LibraryPropertyType(0))
	if err != nil {
		fmt.Println("property:", err)
		return
	}
	fmt.Println("cuBLAS major version:", major)
}
