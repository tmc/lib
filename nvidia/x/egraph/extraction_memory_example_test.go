package egraph

import (
	"context"
	"fmt"

	"github.com/tmc/lib/egraphcore"
)

type exampleMemoryCost struct{}

func (exampleMemoryCost) NodeCost(_ ENode, costs []float64) float64 {
	return egraphcore.ASTSizeCost(costs)
}
func (exampleMemoryCost) NodeCostWithMemory(_ ENode, costs []float64, memory *egraphcore.MemoryScope) (float64, error) {
	if err := memory.Reserve(8); err != nil {
		return 0, err
	}
	value := new(float64)
	*value = egraphcore.ASTSizeCost(costs)
	return *value, nil
}

func ExampleMemoryCostModel() {
	g := New()
	root := g.Add(NewNode("leaf"))
	_ = g.RebuildChecked()
	var model MemoryCostModel = exampleMemoryCost{}
	memory := new(allocationFixture)
	owner, err := ExtractBoundedWithMemory(context.Background(), g, root, model, DefaultExtractionLimits(), memory)
	if err != nil {
		owner.Close()
		fmt.Println(err)
		return
	}
	fmt.Println(owner.Cost, memory.live > 0)
	owner.Close()
	fmt.Println(memory.live)
	// Output:
	// 1 true
	// 0
}

func ExampleExtractBoundedWithMemory() {
	g := New()
	root := g.Add(NewNode("leaf"))
	_ = g.RebuildChecked()
	memory := new(allocationFixture)
	owner, err := ExtractBoundedWithMemory(context.Background(), g, root, nil, DefaultExtractionLimits(), memory)
	if err != nil {
		owner.Close()
		fmt.Println(err)
		return
	}
	fmt.Println(owner.Expr.Op, owner.Cost, memory.live > 0, memory.live < memory.peak)
	owner.Close()
	owner.Close()
	fmt.Println(memory.live)
	// Output:
	// leaf 1 true true
	// 0
}
