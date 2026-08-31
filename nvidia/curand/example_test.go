package curand_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/curand"
)

// Example generates uniform random numbers on the host with a cuRAND host
// generator, writing directly into Go memory — no GPU required. It returns
// early when the cuRAND library is not available.
func Example() {
	gen, err := curand.CreateGeneratorHost(curand.CURAND_RNG_PseudoDefault)
	if err != nil {
		fmt.Println("cuRAND unavailable:", err)
		return
	}
	defer gen.Destroy()

	if err := gen.SetPseudoRandomGeneratorSeed(42); err != nil {
		fmt.Println("seed:", err)
		return
	}
	out := make([]float32, 1024)
	if err := gen.GenerateUniform(&out[0], uint(len(out))); err != nil {
		fmt.Println("generate:", err)
		return
	}
	fmt.Println("generated", len(out), "uniforms")
}
