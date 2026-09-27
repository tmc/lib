package cudatensor_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

func ExampleTensor_MatMul() {
	g := cudair.NewGraph()
	x := cudatensor.NewInput(g, "X", []int{2, 3}, cudair.Float32)
	w := cudatensor.NewInput(g, "W", []int{3, 4}, cudair.Float32)

	y := x.MatMul(w).Relu()
	if err := y.Err(); err != nil {
		panic(err)
	}

	fmt.Printf("shape: %v, op: %s\n", y.Shape(), y.Node().Op)
	// Output:
	// shape: [2 4], op: relu
}

func ExampleCanonicalizeRanks() {
	graph := &cudatensor.Graph{Nodes: []cudatensor.EinsumNode{
		{ID: "input", OutputRank: 1},
		{ID: "relu", Equation: "x->x", Inputs: []string{"input"}},
	}}
	canonical, err := cudatensor.CanonicalizeRanks(graph)
	if err != nil {
		panic(err)
	}
	fmt.Println(canonical.Nodes[1].Equation)
	fmt.Println(cudatensor.CanonicalKey(graph) != "")
	// Output:
	// A->A
	// true
}

func ExampleComputeSOLBounds() {
	graph := &cudatensor.Graph{Nodes: []cudatensor.EinsumNode{
		{ID: "input", OutputRank: 1, OutputTensors: []cudatensor.TensorInfo{{Name: "x", Elements: 4}}},
		{ID: "copy", Equation: "i->i", Inputs: []string{"input"}, MACs: 4,
			InputTensors:  []cudatensor.TensorInfo{{Name: "x", Elements: 4}},
			OutputTensors: []cudatensor.TensorInfo{{Name: "y", Elements: 4}}},
	}}
	canonical, err := cudatensor.CanonicalizeRanks(graph)
	if err != nil {
		panic(err)
	}
	report, err := cudatensor.ComputeSOLBounds(canonical, cudatensor.Ceiling{
		MACsPerCycle: 1, BytesPerCycle: 4, FrequencyGHz: 1, BytesPerElement: 2,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("fused bound: %.0f ns\n", report.Fused*1e9)
	// Output:
	// fused bound: 4 ns
}
