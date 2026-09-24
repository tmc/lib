package nvrtc_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/nvrtc"
)

// Example compiles a CUDA kernel to PTX at runtime. It returns early when
// the NVRTC library is not available. The resulting PTX can be loaded and
// launched with the driver package. It has no Output comment, so it is
// compiled but not run: what it prints depends on the host's NVRTC.
func Example() {
	major, minor, err := nvrtc.Version()
	if err != nil {
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
	defer prog.Destroy()

	if err := prog.Compile([]string{"--gpu-architecture=compute_70"}); err != nil {
		log, _ := prog.Log()
		fmt.Printf("compile: %v\n%s", err, log)
		return
	}
	ptx, err := prog.PTX()
	if err != nil {
		fmt.Println("ptx:", err)
		return
	}
	fmt.Printf("NVRTC %d.%d compiled %d bytes of PTX\n", major, minor, len(ptx))
}
