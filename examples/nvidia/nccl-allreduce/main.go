// Nccl-allreduce runs a one-rank FP32 all-reduce on the first CUDA device.
package main

import (
	"fmt"
	"log"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		return err
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		return err
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return err
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		return err
	}

	device := int32(0)
	comm, err := nccl.NewCommInitAll(1, &device)
	if err != nil {
		return err
	}
	defer comm.Destroy()
	values := []float32{3, 5, 7, 11}
	p, err := driver.MemAlloc(uint(len(values)) * 4)
	if err != nil {
		return err
	}
	defer p.MemFree()
	if err := p.MemcpyHtoD(unsafe.Pointer(&values[0]), uint(len(values))*4); err != nil {
		return err
	}
	if err := comm.AllReduce(nccl.Buffer(p), nccl.Buffer(p), uint(len(values)), nccl.Float32, nccl.Sum, 0); err != nil {
		return err
	}
	if err := p.MemcpyDtoH(unsafe.Pointer(&values[0]), uint(len(values))*4); err != nil {
		return err
	}
	fmt.Printf("all-reduce = %v\n", values)
	return nil
}
