// Cublas-sgemm multiplies two 2x2 FP32 matrices on the GPU.
package main

import (
	"fmt"
	"log"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
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

	a := []float32{1, 3, 2, 4}
	b := []float32{5, 7, 6, 8}
	c := make([]float32, 4)
	ptrs := make([]driver.DevicePtr, 3)
	for i, values := range [][]float32{a, b, c} {
		p, err := driver.MemAlloc(uint(len(values)) * 4)
		if err != nil {
			return err
		}
		defer p.MemFree()
		if err := p.MemcpyHtoD(unsafe.Pointer(&values[0]), uint(len(values))*4); err != nil {
			return err
		}
		ptrs[i] = p
	}

	h, err := cublas.Create()
	if err != nil {
		return err
	}
	defer h.Destroy()
	alpha, beta := float32(1), float32(0)
	if err := h.SgemmEx(cublas.OpN, cublas.OpN, 2, 2, 2, &alpha,
		uintptr(ptrs[0]), cublas.R32F, 2,
		uintptr(ptrs[1]), cublas.R32F, 2, &beta,
		uintptr(ptrs[2]), cublas.R32F, 2); err != nil {
		return err
	}
	runtime.KeepAlive(ptrs)
	if err := ptrs[2].MemcpyDtoH(unsafe.Pointer(&c[0]), uint(len(c))*4); err != nil {
		return err
	}
	want := []float32{19, 43, 22, 50}
	for i := range want {
		if c[i] != want[i] {
			return fmt.Errorf("C[%d] = %g, want %g", i, c[i], want[i])
		}
	}
	fmt.Printf("C = %v\n", c)
	return nil
}
