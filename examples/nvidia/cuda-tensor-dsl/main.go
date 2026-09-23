// Cuda-tensor-dsl builds a fluent tensor expression and emits its fused CUDA kernel.
package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	g := cudair.NewGraph()
	a := cudatensor.NewInput(g, "a", []int{1024}, cudair.Float32)
	b := cudatensor.NewInput(g, "b", []int{1024}, cudair.Float32)
	scale := cudatensor.NewInput(g, "scale", []int{1024}, cudair.Float32)
	out := a.Add(b).Mul(scale).Relu().Exp()

	source, err := out.Compile("fusedActivation")
	if err != nil {
		return err
	}
	if !strings.Contains(source, "fusedActivation") {
		return fmt.Errorf("generated source has no kernel name")
	}
	fmt.Printf("fused %d IR nodes into %d bytes of CUDA source\n", len(g.Nodes()), len(source))
	return nil
}
