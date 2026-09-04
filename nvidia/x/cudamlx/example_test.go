package cudamlx_test

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamlx"
)

// Example JIT-compiles an MLX-style arange kernel and runs it through
// the command encoder: allocate from the caching allocator, add a kernel
// node with tracked dependencies, commit the graph, and wait for a
// completion handler. It returns early when no CUDA device is available.
func Example() {
	devs, err := cudamlx.Devices()
	if err != nil || len(devs) == 0 {
		fmt.Println("CUDA unavailable")
		return
	}
	dev := devs[0]

	mod, err := cudamlx.GetJitModule(dev, "example_arange", func() (cudamlx.KernelSource, error) {
		return cudamlx.KernelSource{
			Source: `
extern "C" __global__ void arange(float* out, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = float(i);
}`,
			KernelNames: []string{"arange"},
		}, nil
	})
	if err != nil {
		fmt.Println("jit:", err)
		return
	}
	fn, blockDim, err := mod.GetKernel("arange", nil)
	if err != nil {
		fmt.Println("kernel:", err)
		return
	}

	enc, err := cudamlx.NewCommandEncoder(dev)
	if err != nil {
		fmt.Println("encoder:", err)
		return
	}
	defer enc.Close()

	alloc, err := cudamlx.GetAllocator()
	if err != nil {
		fmt.Println("allocator:", err)
		return
	}
	const n = 1024
	out, err := alloc.MallocAsync(n*4, dev.Ordinal(), enc.Stream())
	if err != nil {
		fmt.Println("malloc:", err)
		return
	}
	defer alloc.Free(out)

	var args cudamlx.KernelArgs
	args.AppendBuffer(out)
	args.AppendInt32(n)
	enc.SetOutputBuffer(out)
	if blockDim > n {
		blockDim = n
	}
	err = enc.AddKernelNode(fn, cudamlx.Dim3{X: (n + blockDim - 1) / blockDim},
		cudamlx.Dim3{X: blockDim}, 0, &args)
	if err != nil {
		fmt.Println("kernel node:", err)
		return
	}
	if err := enc.Synchronize(); err != nil {
		fmt.Println("synchronize:", err)
		return
	}

	host := make([]float32, n)
	err = dev.WithCurrent(func() error {
		return driver.DevicePtr(out.Data).MemcpyDtoH(unsafe.Pointer(&host[0]), n*4)
	})
	if err != nil {
		fmt.Println("copy:", err)
		return
	}
	fmt.Println("first values:", host[:4])
}
