package cudasearch

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestMCTSReuseReplaysSeed(t *testing.T) {
	for _, seed := range []int64{0, 19, -3} {
		t.Run(fmt.Sprint(seed), func(t *testing.T) {
			s, _ := beamContractState()
			var rules []*egraph.Rewrite
			for i := 0; i < 16; i++ {
				name := fmt.Sprintf("alternative%d", i)
				rules = append(rules, egraph.MustRewrite(name, "base", name))
			}
			budget := DefaultBudget()
			budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
			// Leave room for no-op attempts while testing five evaluations.
			budget.MaxStates = 16
			budget.MaxEvaluations = 5
			engine := NewMCTSSearch(MCTSConfig{Seed: seed, Budget: budget})
			var want []string
			for run := 0; run < 3; run++ {
				var trace []string
				eval := beamEvaluatorFunc(func(_ context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
					for _, a := range s.Derivation {
						trace = append(trace, a.Key())
					}
					return &CandidatePlan{Validated: true, Cost: float64(100 - len(s.Derivation))}, nil
				})
				result, err := engine.Search(context.Background(), s, rules, eval)
				if err != nil {
					t.Fatal(err)
				}
				wantSeed := seed
				if seed == 0 {
					wantSeed = 42
				}
				if result.Seed != wantSeed || result.Evaluations != 5 {
					t.Fatalf("seed=%d evaluations=%d", result.Seed, result.Evaluations)
				}
				if run == 0 {
					want = trace
				} else if !reflect.DeepEqual(trace, want) {
					t.Fatalf("run %d trace differs: %v vs %v", run, trace, want)
				}
			}
		})
	}
}
