package curand_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/curand"
)

// Example generates uniform random numbers on the host with a cuRAND host
// generator, writing directly into Go memory — no GPU required. A fixed seed
// gives the same values on every run. It returns early when the cuRAND
// library is not available. It has no Output comment, so it is compiled but
// not run: a host without cuRAND prints an error instead.
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
	out := make([]float32, 4)
	if err := gen.GenerateUniform(&out[0], uint(len(out))); err != nil {
		fmt.Println("generate:", err)
		return
	}
	fmt.Println(out)
}
