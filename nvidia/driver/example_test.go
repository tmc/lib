package driver_test

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Example initializes the driver, takes a context on the first device, and
// round-trips a slice through device memory. It returns early when no CUDA
// driver is unavailable. Successful execution also requires a CUDA device.
func Example() {
	// A CUDA context is current on an OS thread, not on a goroutine.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := driver.Init(0); err != nil {
		fmt.Println("CUDA driver unavailable:", err)
		return
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		fmt.Println("device:", err)
		return
	}
	name, err := dev.GetName()
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
	defer dptr.MemFree()

	if err := dptr.MemcpyHtoD(unsafe.Pointer(&host[0]), size); err != nil {
		fmt.Println("copy in:", err)
		return
	}
	back := make([]float32, len(host))
	if err := dptr.MemcpyDtoH(unsafe.Pointer(&back[0]), size); err != nil {
		fmt.Println("copy out:", err)
		return
	}
	fmt.Println(name, back)
}
