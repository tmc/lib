package cudasearch

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// A narrower frontier may discard a temporary regression that enables a better
// later rewrite. Beam pruning is a heuristic, not an optimality guarantee.
func TestBeamTemporaryCostIncrease(t *testing.T) {
	for _, tc := range []struct {
		width int
		want  float64
	}{{1, 1}, {2, 0.5}} {
		s, _ := beamContractState()
		rules := []*egraph.Rewrite{
			egraph.MustRewrite("tempting", "base", "tempting"),
			egraph.MustRewrite("detour", "base", "detour"),
			egraph.MustRewrite("goal", "detour", "goal"),
		}
		eval := beamEvaluatorFunc(func(_ context.Context, state *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
			cost := float64(3)
			for _, action := range state.Derivation {
				switch action.RuleName {
				case "tempting":
					cost = 1
				case "detour":
					cost = 4
				case "goal":
					cost = 0.5
				}
			}
			return &CandidatePlan{Validated: true, Cost: cost, CostSource: CostEstimated, Derivation: append([]egraph.Action(nil), state.Derivation...)}, nil
		})
		budget := DefaultBudget()
		budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
		budget.MaxDepth = 2
		result, err := NewBeamSearch(BeamConfig{BeamWidth: tc.width, Budget: budget}).Search(context.Background(), s, rules, eval)
		if err != nil {
			t.Fatal(err)
		}
		if result.Winner.Cost != tc.want {
			t.Fatalf("width=%d cost=%v want=%v", tc.width, result.Winner.Cost, tc.want)
		}
		if tc.width == 2 && (len(result.Winner.Derivation) != 2 || result.Winner.Derivation[0].RuleName != "detour" || result.Winner.Derivation[1].RuleName != "goal") {
			t.Fatalf("missing enabling detour: %+v", result.Winner.Derivation)
		}
	}
}
