package cufft_test

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cufft"
	"github.com/tmc/lib/nvidia/driver"
)

// Example transforms a four-point complex impulse in place on the first CUDA
// device. The transform of an impulse is 1+0i at every point. It returns
// early when the CUDA driver or cuFFT is not available. It has no Output
// comment, so it is compiled but not run: it needs a CUDA device.
func Example() {
	version, err := cufft.GetVersion()
	if err != nil {
		fmt.Println("cuFFT unavailable:", err)
		return
	}
	fmt.Println("cuFFT version:", version)

	// A CUDA context is current on an OS thread, not on a goroutine.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := driver.Init(0); err != nil {
		fmt.Println("CUDA driver unavailable:", err)
		return
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		fmt.Println("device:", err)
		return
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		fmt.Println("context:", err)
		return
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		fmt.Println("set context:", err)
		return
	}

	values := []cufft.Complex{{X: 1}, {}, {}, {}}
	size := uint(len(values)) * uint(unsafe.Sizeof(values[0]))
	p, err := driver.MemAlloc(size)
	if err != nil {
		fmt.Println("alloc:", err)
		return
	}
	defer p.Free()
	if err := p.CopyFromHost(unsafe.Pointer(&values[0]), size); err != nil {
		fmt.Println("copy in:", err)
		return
	}

	plan, err := cufft.Plan1d(int32(len(values)), cufft.C2C, 1)
	if err != nil {
		fmt.Println("plan:", err)
		return
	}
	defer plan.Destroy()
	if err := plan.ExecC2C(uintptr(p), uintptr(p), cufft.Forward); err != nil {
		fmt.Println("exec:", err)
		return
	}

	if err := p.CopyToHost(unsafe.Pointer(&values[0]), size); err != nil {
		fmt.Println("copy out:", err)
		return
	}
	fmt.Println("FFT(impulse) =", values)
}
