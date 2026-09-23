package cudnn_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cudnn"
)

// Example creates a cuDNN handle, queries the library version, and creates a
// tensor descriptor. It returns early when the cuDNN library is not
// available. It has no Output comment, so it is compiled but not run: what it
// prints depends on the host's cuDNN.
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

	desc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		fmt.Println("tensor descriptor:", err)
		return
	}
	defer desc.Destroy()
	fmt.Println("tensor descriptor ready")
}
