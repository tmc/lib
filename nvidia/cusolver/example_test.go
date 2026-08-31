package cusolver_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cusolver"
)

// Example creates a cuSOLVER dense handle and queries the library version.
// It returns early when the cuSOLVER library is not available.
func Example() {
	version, err := cusolver.GetVersion()
	if err != nil {
		fmt.Println("cuSOLVER unavailable:", err)
		return
	}
	fmt.Println("cuSOLVER version:", version)

	h, err := cusolver.Create()
	if err != nil {
		fmt.Println("create:", err)
		return
	}
	defer h.Destroy()
	fmt.Println("dense handle ready")
}
