// Nvrtc-compile compiles a CUDA source string to PTX without nvcc.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/nvrtc"
)

const source = `extern "C" __global__ void scale(float *x) { x[threadIdx.x] *= 2; }`

func main() {
	major, minor, err := nvrtc.Version()
	if err != nil {
		log.Fatal(err)
	}
	p, err := nvrtc.CreateProgram(source, "scale.cu", nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer nvrtc.DestroyProgram(&p)
	if err := p.Compile(nil); err != nil {
		log.Fatal(err)
	}
	ptx, err := p.GetPTX()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("NVRTC %d.%d compiled %d bytes of PTX\n", major, minor, len(ptx))
}
