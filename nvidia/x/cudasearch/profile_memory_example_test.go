package cudasearch

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// A positive MaxMemoryBytes bounds the profile copy and its identity encoding
// as well as the search. The same profile is refused before any evaluation
// when the limit cannot hold it, and admitted unchanged when it can.
func ExampleProfileActions_memoryLimit() {
	profile := actionProfile()
	profile.Snapshot.PCStalls = make(map[uint64]map[string]int64)
	for pc := uint64(0); pc < 4000; pc++ {
		profile.Snapshot.PCStalls[pc] = map[string]int64{"reason": int64(pc)}
	}
	state, rules := beamContractState()
	for _, limit := range []int64{1 << 20, 1 << 28} {
		budget := SearchBudget{MaxMemoryBytes: limit, CallbackMemoryBytes: 4096, MaxStates: 2, MaxDepth: 2, MaxEvaluations: 2}
		eval := beamEvaluatorFunc(func(_ context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
			return &CandidatePlan{Cost: 1, Validated: true, Derivation: s.Derivation}, nil
		})
		result, err := NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: budget, Profile: profile}).Search(context.Background(), state, rules, eval)
		if err != nil {
			fmt.Println(limit, errors.Is(err, ErrMemoryBudgetExceeded))
			continue
		}
		fmt.Println(limit, len(result.ProfileGuidance.PCStalls), result.ProfileGuidance.HotPCs["reason"][:2])
		result.Close()
	}
	// Output:
	// 1048576 true
	// 268435456 4000 [3999 3998]
}
