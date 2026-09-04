// Cufft-impulse transforms a four-point complex impulse in place on the GPU.
package main

import (
	"fmt"
	"log"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cufft"
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
		return fmt.Errorf("initialize CUDA: %w", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		return fmt.Errorf("get CUDA device: %w", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return fmt.Errorf("retain primary context: %w", err)
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		return fmt.Errorf("set current context: %w", err)
	}

	values := []cufft.Complex{{X: 1}, {}, {}, {}}
	size := uint(len(values)) * uint(unsafe.Sizeof(values[0]))
	p, err := driver.MemAlloc(size)
	if err != nil {
		return fmt.Errorf("allocate device input: %w", err)
	}
	defer p.MemFree()
	if err := p.MemcpyHtoD(unsafe.Pointer(&values[0]), size); err != nil {
		return fmt.Errorf("copy input to device: %w", err)
	}
	plan, err := cufft.Plan1d(int32(len(values)), cufft.C2C, 1)
	if err != nil {
		return fmt.Errorf("create cuFFT plan: %w", err)
	}
	defer plan.Destroy()
	if err := plan.ExecC2C(uintptr(p), uintptr(p), cufft.Forward); err != nil {
		return fmt.Errorf("execute cuFFT plan: %w", err)
	}
	if err := p.MemcpyDtoH(unsafe.Pointer(&values[0]), size); err != nil {
		return fmt.Errorf("copy output to host: %w", err)
	}
	for i, v := range values {
		if v.X != 1 || v.Y != 0 {
			return fmt.Errorf("result[%d] = %+v, want 1+0i", i, v)
		}
	}
	fmt.Println("FFT(impulse) = [1+0i 1+0i 1+0i 1+0i]")
	return nil
}
