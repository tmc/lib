package nvrtc_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvrtc"
)

// Example compiles a CUDA kernel to PTX at runtime. It returns early when
// the NVRTC library is not available. The resulting PTX can be loaded and
// launched with the driver package.
func Example() {
	if _, _, err := nvrtc.Version(); err != nil {
		fmt.Println("NVRTC unavailable:", err)
		return
	}

	const src = `
extern "C" __global__ void saxpy(float a, float *x, float *y, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a*x[i] + y[i];
}`
	prog, err := nvrtc.CreateProgram(src, "saxpy.cu", nil, nil)
	if err != nil {
		fmt.Println("create:", err)
		return
	}
	defer nvrtc.DestroyProgram(&prog)

	if err := prog.Compile([]string{"--gpu-architecture=compute_70"}); err != nil {
		log, _ := prog.GetProgramLog()
		fmt.Printf("compile: %v\n%s", err, log)
		return
	}
	ptx, err := prog.GetPTX()
	if err != nil {
		fmt.Println("ptx:", err)
		return
	}
	fmt.Println("compiled", len(ptx), "bytes of PTX")
}
