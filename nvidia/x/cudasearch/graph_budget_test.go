package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestGraphBudgetBeforeEvaluation(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, kind := range []string{"initial_nodes", "initial_classes", "expanded_nodes", "expanded_classes", "rollout"} {
			t.Run(strategy+"/"+kind, func(t *testing.T) {
				s, rules := beamContractState()
				budget := DefaultBudget()
				budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
				wantCalls := 1
				switch kind {
				case "initial_nodes":
					s.Graph.Add(egraph.ENode{Op: "extra"})
					budget.MaxNodesPerState = 1
					wantCalls = 0
				case "initial_classes":
					s.Graph.Add(egraph.ENode{Op: "extra"})
					budget.MaxClassesPerState = 1
					wantCalls = 0
				case "expanded_nodes":
					budget.MaxNodesPerState = 1
				case "expanded_classes":
					budget.MaxClassesPerState = 1
					rules = []*egraph.Rewrite{egraph.MustRewrite("grow", "base", "(pair extra extra)")}
				case "rollout":
					budget.MaxNodesPerState = 2
					rules = append(rules, egraph.MustRewrite("deeper", "alternative", "deeper"))
					wantCalls = 2
				}
				calls := 0
				eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
					calls++
					return &CandidatePlan{Validated: true, Cost: 10}, nil
				})
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				}
				if calls != wantCalls {
					t.Fatalf("calls=%d want=%d result=%+v err=%v", calls, wantCalls, result, err)
				}
				if wantCalls == 0 {
					if result != nil || !errors.Is(err, ErrBudgetExhausted) {
						t.Fatalf("initial admission: %+v %v", result, err)
					}
					return
				}
				if err != nil || result == nil || result.Winner == nil || !errors.Is(result.StopReason, ErrBudgetExhausted) {
					t.Fatalf("lost best at graph bound: %+v %v", result, err)
				}
			})
		}
	}
}
