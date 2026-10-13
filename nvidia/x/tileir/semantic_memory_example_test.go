package tileir_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/tileir"
	"github.com/tmc/lib/nvidia/x/tileiras"
)

func exampleMemory() *tileir.LoweringResult {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	y := g.AddInput("y", []int{1}, cudair.Float32)
	_ = g.SetOutputs(g.AddBinary("add", x, y))
	snapshot, err := g.Snapshot()
	if err != nil {
		return nil
	}
	r, err := tileir.LowerSemanticMemory(snapshot, tileir.LowerOptions{})
	if err != nil {
		return nil
	}
	return r
}

func ExampleLowerSemanticMemory() {
	r := exampleMemory()
	fmt.Println(r.Module.Functions[0].Name, r.Grid, r.Schema.TotalParamBytes)
	// Output: tile_singleton_add [1 1 1] 24
}

func ExampleSemanticMemoryBinding_References() {
	r := exampleMemory()
	for _, source := range r.Memory.References() {
		fmt.Println(source.Role, source.ABIIndex)
	}
	// Output:
	// input 0
	// input 1
	// add -1
	// output 2
}

func ExampleSemanticMemoryBinding_BytecodeDigest() {
	r := exampleMemory()
	fmt.Println(len(r.Memory.BytecodeDigest()))
	// Output: 64
}

func ExampleSemanticMemoryBinding_Verify() {
	r := exampleMemory()
	fmt.Println(r.Memory.Verify(r))
	r.Grid[0] = 2
	fmt.Println(r.Memory.Verify(r) != nil)
	// Output:
	// <nil>
	// true
}

func ExampleAssembleSemanticMemory() {
	// Suffix rejection is a pure preflight; no assembler is invoked.
	_, err := tileir.AssembleSemanticMemory(context.Background(), nil, "sm_90a")
	fmt.Println(errors.Is(err, tileiras.ErrArchSuffix))
	// Output: true
}
