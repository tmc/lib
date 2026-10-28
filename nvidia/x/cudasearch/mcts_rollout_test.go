package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestMCTSEvaluatesRollout(t *testing.T) {
	for _, limit := range []string{"none", "states", "evaluations", "depth", "rejected", "memory", "canceled"} {
		t.Run(limit, func(t *testing.T) {
			s, rules := beamContractState()
			rules = append(rules, egraph.MustRewrite("deeper", "alternative", "deeper"))
			budget := DefaultBudget()
			budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
			budget.MaxDepth = 2
			switch limit {
			case "states":
				budget.MaxStates = 2
			case "evaluations":
				budget.MaxEvaluations = 2
			case "depth":
				budget.MaxDepth = 1
			case "memory":
				budget.CallbackMemoryBytes = cloneCallbackBytes()
				budget.MaxMemoryBytes = transitionIncumbentBudget(t, budget, 2, 9, func(budget SearchBudget, eval Evaluator) (*SearchResult, error) {
					return NewMCTSSearch(MCTSConfig{Iterations: 1, RolloutDepth: 4, Budget: budget}).Search(context.Background(), s, rules, eval)
				})
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			var eval Evaluator = beamEvaluatorFunc(func(_ context.Context, state *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				calls++
				depth := len(state.Derivation)
				if depth == 2 && limit == "rejected" {
					return nil, ErrNoAdmittedPlan
				}
				if depth == 1 && limit == "canceled" {
					cancel()
				}
				return &CandidatePlan{Cost: float64(10 - depth), Validated: true}, nil
			})
			if limit == "memory" {
				budget, eval = transitionSetupBudget(t, budget, eval)
			}
			result, err := NewMCTSSearch(MCTSConfig{Iterations: 1, RolloutDepth: 4, Budget: budget}).Search(ctx, s, rules, eval)
			if limit == "memory" && result != nil {
				defer result.Close()
			}
			wantCalls, wantCost := 2, float64(9)
			switch limit {
			case "none":
				wantCalls, wantCost = 3, 8
			case "rejected":
				wantCalls = 3
			}
			if result == nil || calls != wantCalls || result.Evaluations != calls || result.Winner.Cost != wantCost {
				t.Fatalf("calls=%d result=%+v err=%v", calls, result, err)
			}
			switch limit {
			case "memory":
				if !errors.Is(err, ErrMemoryBudgetExceeded) {
					t.Fatal(err)
				}
			case "canceled":
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			default:
				if err != nil {
					t.Fatal(err)
				}
			}
			if limit == "rejected" && (len(result.Rejected) != 1 || len(result.Rejected[0].Derivation) != 2) {
				t.Fatalf("missing rollout rejection: %+v", result.Rejected)
			}
		})
	}
}

func TestMCTSFailedReward(t *testing.T) {
	if got := mctsReward(nil, 10); got != 0 {
		t.Fatalf("failed reward=%v", got)
	}
	engine := NewMCTSSearch(MCTSConfig{})
	if got := engine.selectChild(&mctsNode{children: []*mctsNode{{isTerminal: true}}}); got != nil {
		t.Fatal("selected exhausted child")
	}
}
