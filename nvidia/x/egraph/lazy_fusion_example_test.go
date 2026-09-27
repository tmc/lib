package egraph_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleExtractFusionRegions() {
	// Build an AST expression with chained elementwise operations bounded by sum:
	// (sum (mul (add a b) c))
	a := &egraph.Expr{Op: "a"}
	b := &egraph.Expr{Op: "b"}
	c := &egraph.Expr{Op: "c"}

	add := &egraph.Expr{Op: "add", Children: []*egraph.Expr{a, b}}
	mul := &egraph.Expr{Op: "mul", Children: []*egraph.Expr{add, c}}
	expr := &egraph.Expr{Op: "sum", Children: []*egraph.Expr{mul}}

	plan, err := egraph.ExtractFusionRegions(expr)
	if err != nil {
		fmt.Printf("extraction failed: %v\n", err)
		return
	}

	fmt.Printf("root boundary: %s\n", plan.RootBoundary)
	fmt.Printf("extracted regions: %d\n", len(plan.Regions))
	if len(plan.Regions) > 0 {
		fmt.Printf("region 1 ops: %v\n", plan.Regions[0].OpNames)
		fmt.Printf("region 1 kernel compiled: %t\n", plan.Regions[0].Kernel != nil)
	}

	// Output:
	// root boundary: sum
	// extracted regions: 1
	// region 1 ops: [add mul]
	// region 1 kernel compiled: true
}
