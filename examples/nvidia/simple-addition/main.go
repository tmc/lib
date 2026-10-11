// Simple-addition adds two vectors of a million floats on the GPU.
//
// It is a port of the canonical CUDA vector-add sample. The kernel source is
// compiled at run time with NVRTC, so no CUDA toolchain is needed to build
// this program; only the driver and NVRTC shared libraries at run time.
package main

import (
	"fmt"
	"log"
	"math/rand/v2"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
)

const kernelSource = `
extern "C" __global__ void VectorAdd(const float *A, const float *B, float *C, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
		C[idx] = A[idx] + B[idx];
}
`

const (
	vectorLen       = 1024 * 1024
	threadsPerBlock = 128
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
		return fmt.Errorf("cuInit: %w", err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		return fmt.Errorf("cuDeviceGet: %w", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return fmt.Errorf("cuDevicePrimaryCtxRetain: %w", err)
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		return fmt.Errorf("cuCtxSetCurrent: %w", err)
	}

	ptx, err := compile(dev)
	if err != nil {
		return err
	}
	mod, err := driver.ModuleLoadData(ptx)
	if err != nil {
		return fmt.Errorf("cuModuleLoadData: %w", err)
	}
	defer mod.Close()
	fn, err := mod.Function("VectorAdd")
	if err != nil {
		return fmt.Errorf("cuModuleGetFunction: %w", err)
	}

	// Host vectors.
	a := make([]float32, vectorLen)
	b := make([]float32, vectorLen)
	c := make([]float32, vectorLen)
	for i := range a {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}

	size := uint(vectorLen * unsafe.Sizeof(float32(0)))
	var ptrs [3]driver.DevicePtr
	for i := range ptrs {
		p, err := driver.MemAlloc[float32](vectorLen)
		if err != nil {
			return fmt.Errorf("cuMemAlloc: %w", err)
		}
		defer p.Free()
		ptrs[i] = p
	}
	dA, dB, dC := ptrs[0], ptrs[1], ptrs[2]

	if err := dA.CopyFromHostPtr(unsafe.Pointer(&a[0]), size); err != nil {
		return fmt.Errorf("cuMemcpyHtoD A: %w", err)
	}
	if err := dB.CopyFromHostPtr(unsafe.Pointer(&b[0]), size); err != nil {
		return fmt.Errorf("cuMemcpyHtoD B: %w", err)
	}

	// Kernel parameters are pointers to the argument values. The slice keeps
	// them reachable across the launch.
	n := int32(vectorLen)
	args := []unsafe.Pointer{
		unsafe.Pointer(&dA), unsafe.Pointer(&dB), unsafe.Pointer(&dC), unsafe.Pointer(&n),
	}
	blocksPerGrid := uint32((vectorLen + threadsPerBlock - 1) / threadsPerBlock)
	if err := fn.LaunchKernel(driver.Dim3{X: blocksPerGrid, Y: 1, Z: 1}, driver.Dim3{X: threadsPerBlock, Y: 1, Z: 1}, 0, 0, args); err != nil {
		return fmt.Errorf("cuLaunchKernel: %w", err)
	}
	if err := dC.CopyToHostPtr(unsafe.Pointer(&c[0]), size); err != nil {
		return fmt.Errorf("cuMemcpyDtoH C: %w", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		return fmt.Errorf("cuCtxSynchronize: %w", err)
	}

	for i := range c {
		if want := a[i] + b[i]; c[i] != want {
			return fmt.Errorf("C[%d] = %v, want %v", i, c[i], want)
		}
	}
	name, err := dev.Name()
	if err != nil {
		return fmt.Errorf("cuDeviceGetName: %w", err)
	}
	fmt.Printf("%s: added %d floats\n", name, vectorLen)
	return nil
}

// compile compiles kernelSource to PTX for dev's compute capability. The
// returned PTX is NUL-terminated, as cuModuleLoadData requires.
func compile(dev driver.Device) ([]byte, error) {
	major, err := dev.Attribute(driver.DevAttrComputeCapabilityMajor)
	if err != nil {
		return nil, fmt.Errorf("cuDeviceGetAttribute major: %w", err)
	}
	minor, err := dev.Attribute(driver.DevAttrComputeCapabilityMinor)
	if err != nil {
		return nil, fmt.Errorf("cuDeviceGetAttribute minor: %w", err)
	}
	prog, err := nvrtc.CreateProgram(kernelSource, "vectoradd.cu")
	if err != nil {
		return nil, fmt.Errorf("nvrtcCreateProgram: %w", err)
	}
	defer prog.Destroy()

	arch := fmt.Sprintf("--gpu-architecture=compute_%d%d", major, minor)
	if err := prog.Compile(arch); err != nil {
		if logText, logErr := prog.Log(); logErr == nil && logText != "" {
			return nil, fmt.Errorf("nvrtcCompileProgram: %w\n%s", err, logText)
		}
		return nil, fmt.Errorf("nvrtcCompileProgram: %w", err)
	}
	ptx, err := prog.PTX()
	if err != nil {
		return nil, fmt.Errorf("nvrtcGetPTX: %w", err)
	}
	return ptx, nil
}
