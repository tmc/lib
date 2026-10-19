package egraph_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type transitionExampleMemory struct{ bytes int64 }

func (m *transitionExampleMemory) Reserve(n int64) error { m.bytes += n; return nil }
func (m *transitionExampleMemory) Release(n int64)       { m.bytes -= n }

func ExampleSearchState_ApplyActionWithCloneBudget_transition() {
	g := egraph.New()
	root := g.Add(egraph.NewNode("a"))
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	rule := egraph.MustRewrite("replace", "a", "b")
	actions := egraph.EnumerateActions(state.Graph, []*egraph.Rewrite{rule})
	memory := new(transitionExampleMemory)
	scope, _ := egraphcore.NewMemoryScope(memory)
	defer scope.Close()
	budget, _ := egraphcore.NewWorkBudgetWithMemory(0, scope)
	result, err := state.ApplyActionWithCloneBudget(context.Background(), actions[0], budget)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Merged, len(result.State.Derivation), memory.bytes > 0)
	// Output: true 1 true
}
