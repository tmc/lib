package cudasearch

import (
	"context"
	"errors"
	"testing"
)

func TestSearchSharesExtractionWork(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, limit := range []int{3, 4} {
			t.Run(strategy+string(rune('0'+limit)), func(t *testing.T) {
				s, rules := beamContractState()
				budget := DefaultBudget()
				budget.MaxExtractionWork = limit
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, &MockEvaluator{})
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, &MockEvaluator{})
				}
				if err != nil || result == nil || result.Winner == nil || !errors.Is(result.StopReason, ErrBudgetExhausted) || result.ExtractionWork != limit || len(result.AllCandidates) != 1 {
					t.Fatalf("result=%+v err=%v", result, err)
				}
			})
		}
	}
}
