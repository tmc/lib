// Simple-addition-gorgoniacu adds two vectors of a million floats on the GPU
// using gorgonia.org/cu, a cgo wrapper of the CUDA Driver API.
//
// It is the same program as ../simple-addition, which uses the purego-based
// bindings in github.com/tmc/lib/nvidia; the two side by side show what the
// cgo dependency buys and costs. Building this one requires the CUDA toolkit
// (headers and libcuda/libnvrtc), so it lives in its own module and is not
// part of the github.com/tmc/lib build.
package main

import (
	"fmt"
	"log"
	"math/rand/v2"
	"unsafe"

	"gorgonia.org/cu"
	"gorgonia.org/cu/nvrtc"
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
	count, err := cu.NumDevices()
	if err != nil {
		return fmt.Errorf("device count: %w", err)
	}
	if count == 0 {
		return fmt.Errorf("no CUDA devices")
	}
	dev := cu.Device(0)
	ctx, err := dev.MakeContext(cu.SchedAuto)
	if err != nil {
		return fmt.Errorf("make context: %w", err)
	}
	defer ctx.Destroy()
	// cu's driver calls are thread-affine: pin the goroutine to its OS thread
	// with the context current for the duration.
	if err := ctx.Lock(); err != nil {
		return fmt.Errorf("lock context: %w", err)
	}
	defer ctx.Unlock()

	ptx, err := compile(dev)
	if err != nil {
		return err
	}
	mod, err := cu.LoadData(ptx)
	if err != nil {
		return fmt.Errorf("load module: %w", err)
	}
	defer mod.Unload()
	fn, err := mod.Function("VectorAdd")
	if err != nil {
		return fmt.Errorf("get function: %w", err)
	}

	// Host vectors.
	a := make([]float32, vectorLen)
	b := make([]float32, vectorLen)
	c := make([]float32, vectorLen)
	for i := range a {
		a[i] = rand.Float32()
		b[i] = rand.Float32()
	}

	size := int64(vectorLen) * int64(unsafe.Sizeof(float32(0)))
	var ptrs [3]cu.DevicePtr
	for i := range ptrs {
		p, err := cu.MemAlloc(size)
		if err != nil {
			return fmt.Errorf("mem alloc: %w", err)
		}
		defer cu.MemFree(p)
		ptrs[i] = p
	}
	dA, dB, dC := ptrs[0], ptrs[1], ptrs[2]

	if err := cu.MemcpyHtoD(dA, unsafe.Pointer(&a[0]), size); err != nil {
		return fmt.Errorf("memcpy A: %w", err)
	}
	if err := cu.MemcpyHtoD(dB, unsafe.Pointer(&b[0]), size); err != nil {
		return fmt.Errorf("memcpy B: %w", err)
	}

	// Kernel parameters are pointers to the argument values, which must stay
	// reachable across the launch.
	n := int32(vectorLen)
	args := []unsafe.Pointer{
		unsafe.Pointer(&dA), unsafe.Pointer(&dB), unsafe.Pointer(&dC), unsafe.Pointer(&n),
	}
	blocksPerGrid := (vectorLen + threadsPerBlock - 1) / threadsPerBlock
	if err := fn.Launch(blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, cu.Stream{}, args); err != nil {
		return fmt.Errorf("launch: %w", err)
	}
	if err := cu.MemcpyDtoH(unsafe.Pointer(&c[0]), dC, size); err != nil {
		return fmt.Errorf("memcpy C: %w", err)
	}
	if err := cu.Synchronize(); err != nil {
		return fmt.Errorf("synchronize: %w", err)
	}

	for i := range c {
		if want := a[i] + b[i]; c[i] != want {
			return fmt.Errorf("C[%d] = %v, want %v", i, c[i], want)
		}
	}
	name, err := dev.Name()
	if err != nil {
		return fmt.Errorf("device name: %w", err)
	}
	fmt.Printf("%s: added %d floats\n", name, vectorLen)
	return nil
}

// compile compiles kernelSource to PTX for dev's compute capability.
func compile(dev cu.Device) (string, error) {
	major, err := dev.Attribute(cu.ComputeCapabilityMajor)
	if err != nil {
		return "", fmt.Errorf("compute capability major: %w", err)
	}
	minor, err := dev.Attribute(cu.ComputeCapabilityMinor)
	if err != nil {
		return "", fmt.Errorf("compute capability minor: %w", err)
	}
	prog, err := nvrtc.CreateProgram(kernelSource, "vectoradd.cu")
	if err != nil {
		return "", fmt.Errorf("create program: %w", err)
	}
	defer prog.Destroy()

	arch := fmt.Sprintf("--gpu-architecture=compute_%d%d", major, minor)
	if err := prog.Compile(arch); err != nil {
		if logText, logErr := prog.GetLog(); logErr == nil && logText != "" {
			return "", fmt.Errorf("compile: %w\n%s", err, logText)
		}
		return "", fmt.Errorf("compile: %w", err)
	}
	ptx, err := prog.GetPTX()
	if err != nil {
		return "", fmt.Errorf("get ptx: %w", err)
	}
	return ptx, nil
}
