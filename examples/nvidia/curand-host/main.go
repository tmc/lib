// Curand-host generates reproducible random values without a CUDA context.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/curand"
)

func main() {
	g, err := curand.CreateGeneratorHost(curand.CURAND_RNG_PseudoDefault)
	if err != nil {
		log.Fatal(err)
	}
	defer g.Destroy()
	if err := g.SetPseudoRandomGeneratorSeed(1); err != nil {
		log.Fatal(err)
	}
	values := make([]float32, 4)
	if err := g.GenerateUniform(&values[0], uint(len(values))); err != nil {
		log.Fatal(err)
	}
	fmt.Println(values)
}
