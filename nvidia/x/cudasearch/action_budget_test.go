package cudasearch

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSearchTotalActionBudget(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, noOp := range []bool{false, true} {
			t.Run(strategy+map[bool]string{false: "/productive", true: "/noop"}[noOp], func(t *testing.T) {
				s, rules := beamContractState()
				if noOp {
					rules = []*egraph.Rewrite{egraph.MustRewrite("identity", "base", "base")}
				} else {
					rules = append(rules, egraph.MustRewrite("deeper", "alternative", "deeper"))
				}
				budget := DefaultBudget()
				budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
				budget.MaxTotalActions = 1
				calls := 0
				eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
					calls++
					return &CandidatePlan{Cost: float64(11 - calls), Validated: true}, nil
				})
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				}
				wantCalls := 2
				if noOp {
					wantCalls = 1
				}
				if err != nil || result == nil || result.ActionsAttempted != 1 || calls != wantCalls || result.Winner.Cost != float64(11-wantCalls) {
					t.Fatalf("calls=%d result=%+v err=%v", calls, result, err)
				}
				if !noOp && !errors.Is(result.StopReason, ErrBudgetExhausted) {
					t.Fatalf("missing budget stop: %v", result.StopReason)
				}
			})
		}
	}
}
