package cudasearch

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSearchFreezesScoring(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			s, rules := beamContractState()
			evidence := &ProfileEvidence{Epoch: "old", Occupancy: 1, StallMemoryThrottle: 0.25}
			model := DefaultScoringModel("old")
			eval := &MockEvaluator{Scoring: model, Evidence: map[string]*ProfileEvidence{}}
			for _, op := range []string{"base", "alternative"} {
				eval.Evidence[(&egraph.Expr{Op: op}).Fingerprint()] = evidence
			}
			eval.Workload.ResetState = func() error {
				eval.Scoring = DefaultScoringModel("new")
				eval.Scoring.BaseLatencyWeight = 2
				evidence.StallMemoryThrottle = 1
				clear(eval.Evidence)
				return nil
			}
			budget := DefaultBudget()
			// Reset mutates caller-owned captures and returns nil.
			budget.CallbackMemoryBytes = 1
			budget.MaxStates, budget.MaxEvaluations = 2, 2
			search := func() (*SearchResult, error) {
				if strategy == "beam" {
					return NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				}
				return NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
			}
			result, err := search()
			if err != nil {
				t.Fatal(err)
			}
			if len(result.AllCandidates) != 2 || eval.ResetCount != 2 {
				t.Fatalf("result=%+v resets=%d", result, eval.ResetCount)
			}
			for _, candidate := range result.AllCandidates {
				if candidate.Profile == nil || candidate.Profile.StallMemoryThrottle != 0.25 || candidate.Profile == evidence {
					t.Fatalf("lost detached observation: %+v", candidate.Profile)
				}
				want, err := model.Score(candidate.Receipt, &ProfileEvidence{Epoch: "old", Occupancy: 1, StallMemoryThrottle: 0.25})
				if err != nil || candidate.Cost != want || candidate.ScoringEpoch != "old" {
					t.Fatalf("candidate=%+v want=%v err=%v", candidate, want, err)
				}
			}
			result.AllCandidates[0].Profile.StallMemoryThrottle = 0.9
			if result.AllCandidates[1].Profile.StallMemoryThrottle != 0.25 {
				t.Fatal("candidate observations alias")
			}
			next, err := search()
			if err != nil {
				t.Fatal(err)
			}
			if next.Winner.ScoringEpoch != "new" || next.Winner.Cost == result.Winner.Cost {
				t.Fatal("next search did not refresh scoring")
			}
		})
	}
}
