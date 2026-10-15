package egraph

import (
	"context"
	"fmt"
	"github.com/tmc/lib/egraphcore"
)

func ExampleEGraph_CloneWithBudget() {
	g := New()
	g.Add(NewNode("leaf"))
	scope, _ := egraphcore.NewMemoryScope(new(allocationFixture))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	copy, err := g.CloneWithBudget(context.Background(), budget)
	classes, nodes := copy.Size()
	fmt.Println(classes, nodes, err)
	// Output: 1 1 <nil>
}

func ExampleSearchState_CloneWithBudget() {
	g := New()
	id := g.Add(NewNode("leaf"))
	s := NewSearchState(g, map[string]Id{"root": id})
	scope, _ := egraphcore.NewMemoryScope(new(allocationFixture))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	copy, err := s.CloneWithBudget(context.Background(), budget)
	fmt.Println(copy.Roots["root"], err)
	// Output: 0 <nil>
}

func ExampleSearchState_ApplyActionWithCloneBudget() {
	g := New()
	id := g.Add(NewNode("base"))
	s := NewSearchState(g, map[string]Id{"root": id})
	actions := EnumerateActionsBounded(s.Graph, []*Rewrite{MustRewrite("change", "base", "alternative")}, 1)
	scope, _ := egraphcore.NewMemoryScope(new(allocationFixture))
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	result, err := s.ApplyActionWithCloneBudget(context.Background(), actions[0], budget)
	fmt.Println(result.State.Depth, s.Depth, err)
	// Output: 1 0 <nil>
}
