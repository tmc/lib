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
