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
	ctx, err := driver.NewContext(nil, 0, dev)
	if err != nil {
		fmt.Println("context:", err)
		return
	}
	defer ctx.Destroy()

	host := []float32{1, 2, 3, 4}
	size := uint(len(host)) * uint(unsafe.Sizeof(host[0]))
	dptr, err := driver.MemAlloc(size)
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
