package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestNoOpTransitionStateBudget(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			s, _ := beamContractState()
			rules := []*egraph.Rewrite{egraph.MustRewrite("noop-a", "base", "base"), egraph.MustRewrite("noop-b", "base", "base")}
			calls := 0
			eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
				calls++
				return &CandidatePlan{Cost: 1, Validated: true}, nil
			})
			budget := SearchBudget{MaxStates: 2, MaxTotalActions: 20}
			var r *SearchResult
			var err error
			if strategy == "beam" {
				r, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
			} else {
				r, err = NewMCTSSearch(MCTSConfig{Budget: budget, Iterations: 10, RolloutDepth: 4}).Search(context.Background(), s, rules, eval)
			}
			if err != nil {
				t.Fatal(err)
			}
			if r.ActionsAttempted != 1 || r.StatesVisited != 2 || calls != 1 || r.Winner == nil || !errors.Is(r.StopReason, ErrBudgetExhausted) {
				t.Fatalf("calls=%d result=%+v", calls, r)
			}
		})
	}
}

func TestMCTSRolloutNoOpStateBudget(t *testing.T) {
	s, _ := beamContractState()
	// Only one productive expansion; all actions in its rollout are no-ops.
	rules := []*egraph.Rewrite{egraph.MustRewrite("expand", "base", "child"), egraph.MustRewrite("noop", "child", "child")}
	calls := 0
	eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
		calls++
		return &CandidatePlan{Cost: 1, Validated: true}, nil
	})
	r, err := NewMCTSSearch(MCTSConfig{Iterations: 1, RolloutDepth: 4, Budget: SearchBudget{MaxStates: 3, MaxTotalActions: 20, MaxDepth: 4}}).Search(context.Background(), s, rules, eval)
	if err != nil {
		t.Fatal(err)
	}
	if r.ActionsAttempted != 2 || r.StatesVisited != 3 || calls != 2 || !errors.Is(r.StopReason, ErrBudgetExhausted) {
		t.Fatalf("calls=%d result=%+v", calls, r)
	}
}
