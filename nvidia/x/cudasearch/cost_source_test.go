package cudasearch

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSearchRejectsIncomparableCosts(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		for _, mismatch := range []string{"none", "source", "epoch", "timing", "missing_receipt"} {
			t.Run(strategy+"/"+mismatch, func(t *testing.T) {
				base := &CandidatePlan{ID: "base", Cost: 100, CostSource: CostMeasured, Validated: true, ScoringEpoch: "v1", Receipt: &cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingSummedKernel, SummedKernel: 100 * time.Nanosecond}}
				candidate := *base
				candidate.ID, candidate.Cost = "fast", 1
				switch mismatch {
				case "source":
					candidate.CostSource = CostEstimated
				case "epoch":
					candidate.ScoringEpoch = "v2"
				case "timing":
					candidate.Receipt = &cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingGPUBusy, GPUBusy: time.Nanosecond}
				case "missing_receipt":
					candidate.Receipt = nil
				}
				s, rules := beamContractState()
				calls := 0
				eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
					calls++
					if calls == 1 {
						return base, nil
					}
					return &candidate, nil
				})
				budget := DefaultBudget()
				budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
				budget.MaxStates, budget.MaxEvaluations = 2, 2
				var result *SearchResult
				var err error
				if strategy == "beam" {
					result, err = NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				} else {
					result, err = NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, eval)
				}
				if err != nil {
					t.Fatal(err)
				}
				if mismatch == "none" {
					if result.Winner != &candidate {
						t.Fatal("comparable faster candidate rejected")
					}
					return
				}
				if result.Winner != base || len(result.Rejected) != 1 {
					t.Fatalf("incomparable candidate admitted: %+v", result)
				}
			})
		}
	}
}

func TestBuiltInCostSources(t *testing.T) {
	s := executableState()
	mock, err := (&MockEvaluator{}).Evaluate(context.Background(), s, s.Roots["root"])
	if err != nil || mock.CostSource != CostEstimated {
		t.Fatalf("mock=%+v err=%v", mock, err)
	}
	e, _, _, _ := executableFixture(t)
	measured, err := e.Evaluate(context.Background(), s, s.Roots["root"])
	if err != nil || measured.CostSource != CostMeasured {
		t.Fatalf("measured=%+v err=%v", measured, err)
	}
}
