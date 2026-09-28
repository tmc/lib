package driver_test

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Example initializes the driver, takes a context on the first device, and
// round-trips a slice through device memory. It returns early when the CUDA
// driver is unavailable. Successful execution also requires a CUDA device.
func Example() {
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
	name, err := dev.Name()
	if err != nil {
		fmt.Println("name:", err)
		return
	}
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		fmt.Println("context:", err)
		return
	}
	defer ctx.Close()

	host := []float32{1, 2, 3, 4}
	size := uint(len(host)) * uint(unsafe.Sizeof(host[0]))
	dptr, err := driver.MemAllocBytes(size)
	if err != nil {
		fmt.Println("alloc:", err)
		return
	}
	defer dptr.Free()

	if err := dptr.CopyFromHost(host); err != nil {
		fmt.Println("copy in:", err)
		return
	}
	back := make([]float32, len(host))
	if err := dptr.CopyToHost(back); err != nil {
		fmt.Println("copy out:", err)
		return
	}
	fmt.Println(name, back)
}

// ExampleDeviceGetCount prints the driver version and the name and compute
// capability of each device. It returns early when no CUDA driver is
// available. It has no Output comment, so it is compiled but not run: what
// it prints depends on the host's driver and devices.
func ExampleDeviceGetCount() {
	if err := driver.Init(0); err != nil {
		fmt.Println("CUDA driver unavailable:", err)
		return
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		fmt.Println("version:", err)
		return
	}
	count, err := driver.DeviceGetCount()
	if err != nil {
		fmt.Println("device count:", err)
		return
	}
	for i := range count {
		dev, err := driver.DeviceGet(i)
		if err != nil {
			fmt.Println("device:", err)
			return
		}
		name, err := dev.Name()
		if err != nil {
			fmt.Println("name:", err)
			return
		}
		major, err := dev.Attribute(driver.DevAttrComputeCapabilityMajor)
		if err != nil {
			fmt.Println("attribute:", err)
			return
		}
		minor, err := dev.Attribute(driver.DevAttrComputeCapabilityMinor)
		if err != nil {
			fmt.Println("attribute:", err)
			return
		}
		fmt.Printf("CUDA %d: device %d: %s (sm_%d%d)\n", version, i, name, major, minor)
	}
}

// ExampleThreadExchangeStreamCaptureMode relaxes the calling thread's stream
// capture mode around a call that capture would otherwise prohibit, such as
// MemAllocBytes, and restores the previous mode on return. The mode belongs
// to the OS thread, so the goroutine stays locked to it throughout.
func ExampleThreadExchangeStreamCaptureMode() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	mode := driver.StreamCaptureModeRelaxed
	if err := driver.ThreadExchangeStreamCaptureMode(&mode); err != nil {
		fmt.Println("capture mode:", err)
		return
	}
	// mode now holds the previous mode; exchanging again restores it.
	defer driver.ThreadExchangeStreamCaptureMode(&mode)

	p, err := driver.MemAllocBytes(1 << 20)
	if err != nil {
		fmt.Println("alloc:", err)
		return
	}
	defer p.Free()
}

// ExampleFunction_LaunchKernel loads a module and launches
//
//	extern "C" __global__ void scale(float *x, float a, int n)
//
// over n elements. Each kernel parameter is a pointer to the argument's
// value, in declaration order. It needs a current CUDA context.
func ExampleFunction_LaunchKernel() {
	var image []byte // PTX or cubin, e.g. compiled with the nvrtc package

	mod, err := driver.ModuleLoadData(image)
	if err != nil {
		fmt.Println("load module:", err)
		return
	}
	defer mod.Close()
	scale, err := mod.Function("scale")
	if err != nil {
		fmt.Println("function:", err)
		return
	}

	const n, block = 1 << 20, 256
	x, err := driver.MemAllocBytes(n * 4)
	if err != nil {
		fmt.Println("alloc:", err)
		return
	}
	defer x.Free()

	a, count := float32(2), int32(n)
	params := []unsafe.Pointer{unsafe.Pointer(&x), unsafe.Pointer(&a), unsafe.Pointer(&count)}
	err = scale.LaunchKernel(driver.Dims((n+block-1)/block, 1, 1), driver.Dims(block, 1, 1), 0, 0, params)
	if err != nil {
		fmt.Println("launch:", err)
		return
	}
	if err := driver.CtxSynchronize(); err != nil {
		fmt.Println("synchronize:", err)
	}
}
