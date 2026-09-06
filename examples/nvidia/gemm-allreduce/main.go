// Gemm-allreduce multiplies two matrices with cuBLAS and reduces the result
// with a one-rank NCCL communicator, without copying between those operations.
package main

import (
	"fmt"
	"log"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
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

	a := []float32{1, 3, 2, 4}
	b := []float32{5, 7, 6, 8}
	c := make([]float32, 4)
	ptrs, err := copyToDevice(a, b, c)
	if err != nil {
		return err
	}
	defer free(ptrs)

	blas, err := cublas.Create()
	if err != nil {
		return err
	}
	defer blas.Destroy()
	alpha, beta := float32(1), float32(0)
	if err := blas.SgemmEx(cublas.OpN, cublas.OpN, 2, 2, 2, &alpha,
		uintptr(ptrs[0]), cublas.R32F, 2,
		uintptr(ptrs[1]), cublas.R32F, 2, &beta,
		uintptr(ptrs[2]), cublas.R32F, 2); err != nil {
		return err
	}
	runtime.KeepAlive(ptrs)

	devices := []int32{0}
	comms := make([]nccl.Comm, len(devices))
	if err := nccl.CommInitAll(&comms[0], int32(len(devices)), &devices[0]); err != nil {
		return err
	}
	comm := comms[0]
	defer comm.Destroy()
	if err := comm.AllReduce(nccl.Buffer(ptrs[2]), nccl.Buffer(ptrs[2]), uint(len(c)), nccl.Float32, nccl.Sum, 0); err != nil {
		return err
	}
	if err := ptrs[2].MemcpyDtoH(unsafe.Pointer(&c[0]), uint(len(c))*4); err != nil {
		return err
	}
	want := []float32{19, 43, 22, 50}
	for i := range want {
		if c[i] != want[i] {
			return fmt.Errorf("result[%d] = %g, want %g", i, c[i], want[i])
		}
	}
	fmt.Printf("GEMM followed by all-reduce = %v\n", c)
	return nil
}

func copyToDevice(values ...[]float32) ([]driver.DevicePtr, error) {
	ptrs := make([]driver.DevicePtr, len(values))
	for i, value := range values {
		p, err := driver.MemAlloc(uint(len(value)) * 4)
		if err != nil {
			free(ptrs[:i])
			return nil, err
		}
		if err := p.MemcpyHtoD(unsafe.Pointer(&value[0]), uint(len(value))*4); err != nil {
			p.MemFree()
			free(ptrs[:i])
			return nil, err
		}
		ptrs[i] = p
	}
	return ptrs, nil
}

func free(ptrs []driver.DevicePtr) {
	for _, p := range ptrs {
		if p != 0 {
			p.MemFree()
		}
	}
}
