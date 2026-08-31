package cufft_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/cufft"
)

// Example creates a 1D complex-to-complex FFT plan. It returns early when
// the cuFFT library is not available. Executing the plan (ExecC2C) takes
// CUDA device buffers, e.g. from the driver package's MemAlloc.
func Example() {
	version, err := cufft.GetVersion()
	if err != nil {
		fmt.Println("cuFFT unavailable:", err)
		return
	}
	fmt.Println("cuFFT version:", version)

	plan, err := cufft.Plan1d(1024, cufft.C2C, 1)
	if err != nil {
		fmt.Println("plan:", err)
		return
	}
	defer plan.Destroy()
	fmt.Println("1024-point C2C plan ready")
}
