package nccl_test

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// Example queries the NCCL version and creates a single-process
// communicator spanning the local GPUs. It returns early when the NCCL
// library is not available.
func Example() {
	version, err := nccl.GetVersion()
	if err != nil {
		fmt.Println("NCCL unavailable:", err)
		return
	}
	fmt.Println("NCCL version:", version)

	devs := []int32{0}
	comms := make([]nccl.Comm, len(devs))
	err = nccl.CommInitAll(&comms[0], int32(len(devs)), &devs[0])
	if err != nil {
		fmt.Println("comm init:", err)
		return
	}
	defer comms[0].Destroy()

	n, _ := comms[0].Count()
	fmt.Println("communicator ranks:", n)
}

// Example_allReduce sums a float32 buffer across a one-rank communicator on
// the first CUDA device, in place. With one rank the sum is the input. It
// returns early when the CUDA driver or NCCL is not available. It has no
// Output comment, so it is compiled but not run: it needs a CUDA device.
func Example_allReduce() {
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

	devs := []int32{0}
	comms := make([]nccl.Comm, len(devs))
	if err := nccl.CommInitAll(&comms[0], int32(len(devs)), &devs[0]); err != nil {
		fmt.Println("comm init:", err)
		return
	}
	defer comms[0].Destroy()

	values := []float32{3, 5, 7, 11}
	size := uint(len(values)) * uint(unsafe.Sizeof(values[0]))
	p, err := driver.MemAlloc(size)
	if err != nil {
		fmt.Println("alloc:", err)
		return
	}
	defer p.MemFree()
	if err := p.MemcpyHtoD(unsafe.Pointer(&values[0]), size); err != nil {
		fmt.Println("copy in:", err)
		return
	}
	err = comms[0].AllReduce(nccl.Buffer(p), nccl.Buffer(p), uint(len(values)), nccl.Float32, nccl.Sum, 0)
	if err != nil {
		fmt.Println("all-reduce:", err)
		return
	}
	if err := p.MemcpyDtoH(unsafe.Pointer(&values[0]), size); err != nil {
		fmt.Println("copy out:", err)
		return
	}
	fmt.Println("all-reduce =", values)
}
