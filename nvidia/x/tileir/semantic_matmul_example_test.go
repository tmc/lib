package tileir_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/tileir"
)

func ExampleLowerSemanticMatMul() {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 3}, cudair.Float32)
	b := g.AddInput("b", []int{3, 4}, cudair.Float32)
	product := g.AddMatMul(a, b)
	product.Name = "product"
	if err := g.SetOutputs(product); err != nil {
		fmt.Println(err)
		return
	}
	snap, err := g.Snapshot()
	if err != nil {
		fmt.Println(err)
		return
	}
	r, err := tileir.LowerSemanticMatMul(snap, tileir.LowerOptions{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Grid, r.Schema.TotalParamBytes, r.Memory.Verify(r))
	// Output: [1 1 1] 24 <nil>
}
