package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleFusionTensor() {
	x := &egraph.Expr{Op: "x", Data: egraph.FusionTensor{Shape: []int{2, 3}}}
	transpose := &egraph.Expr{Op: "transpose", Data: egraph.FusionTensor{Shape: []int{3, 2}, Permutation: []int{1, 0}}, Children: []*egraph.Expr{x}}
	plan, err := egraph.ExtractFusionRegions(transpose)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(plan.Regions), plan.Regions[0].OpNames, plan.Regions[0].Kernel != nil)
	// Output: 1 [transpose] true
}
