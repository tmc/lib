package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestTransitionAllocationPublicSearch(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			state, _ := beamContractState()
			children := make([]*egraph.Pattern, 128)
			for i := range children {
				children[i] = egraph.Pat("base")
			}
			rule := &egraph.Rewrite{Name: "wide", LHS: egraph.Pat("base"), RHS: egraph.Pat("wide", children...)}
			calls := 0
			baseline := &CandidatePlan{ID: "incumbent", Cost: 10, Validated: true}
			eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				calls++
				if calls != 1 {
					t.Fatal("evaluated a refused transition")
				}
				return baseline, nil
			})
			budget := SearchBudget{MaxMemoryBytes: 64 << 10, CallbackMemoryBytes: cloneCallbackBytes(), MaxStates: 3, MaxDepth: 1, MaxActions: 5, MaxEvaluations: 3}
			budget, admittedEval := transitionSetupBudget(t, budget, eval)
			result, err := memoryStrategy(strategy, budget, state, []*egraph.Rewrite{rule}, admittedEval)
			if result != nil {
				defer result.Close()
			}
			// StatesVisited counts the attempted branch before its admission.
			if !errors.Is(err, ErrMemoryBudgetExceeded) || result == nil || result.Winner != baseline || calls != 1 || result.ActionsAttempted != 1 || result.StatesVisited != 2 {
				t.Fatalf("transition refusal lost incumbent: result=%+v calls=%d err=%v", result, calls, err)
			}
		})
	}
}
