package cudasearch

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func ExampleSearchResult_Close_cloneStorage() {
	g := egraph.New()
	id := g.Add(egraph.NewNode("base"))
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": id})
	eval := beamEvaluatorFunc(func(_ context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
		return &CandidatePlan{Cost: 1, Validated: true, Derivation: s.Derivation}, nil
	})
	result, err := NewBeamSearch(BeamConfig{Budget: SearchBudget{MaxMemoryBytes: 64 << 10, CallbackMemoryBytes: 4096, MaxStates: 1}}).Search(context.Background(), state, nil, eval)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Winner.Cost, result.AccountedBytes > 0)
	result.Close() // Clone-backed aliases must not be used after this point.
	fmt.Println(state.Depth)
	// Output:
	// 1 true
	// 0
}
