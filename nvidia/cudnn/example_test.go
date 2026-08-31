package cudnn_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cudnn"
)

// Example creates a cuDNN handle and queries the library version.
// It returns early when the cuDNN library is not available.
func Example() {
	h, err := cudnn.Create()
	if err != nil {
		fmt.Println("cuDNN unavailable:", err)
		return
	}
	defer h.Destroy()

	// LibraryPropertyType 0 is MAJOR_VERSION (library_types.h).
	major, err := cudnn.GetProperty(cudnn.LibraryPropertyType(0))
	if err != nil {
		fmt.Println("property:", err)
		return
	}
	fmt.Println("cuDNN major version:", major)
}
