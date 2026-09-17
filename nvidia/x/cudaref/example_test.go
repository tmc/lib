package cudaref_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func ExampleEvaluate() {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2, 2}, cudair.Float32)
	w := g.AddInput("W", []int{2, 2}, cudair.Float32)
	y := g.AddMatMul(x, w)
	out := g.AddUnary("relu", y)
	_ = g.SetOutputs(out)

	inputs := map[string][]float32{
		"X": {1, -2, 3, 4},
		"W": {1, 0, 0, 1},
	}

	res, err := cudaref.Evaluate(g, inputs)
	if err != nil {
		panic(err)
	}

	fmt.Println(res[out.Name])
	// Output:
	// [1 0 3 4]
}
