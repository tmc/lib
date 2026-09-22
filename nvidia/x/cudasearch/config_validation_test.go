package cudasearch

import (
	"context"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSearchRejectsNegativeBudgets(t *testing.T) {
	for _, budget := range []SearchBudget{{MaxNodesPerState: -1}, {MaxClassesPerState: -1}, {MaxStates: -1}, {MaxDepth: -1}, {MaxActions: -1}, {MaxTotalActions: -1}, {MaxMemoryBytes: -1}, {MaxEvaluations: -1}, {MaxExtractions: -1}, {MaxCompilations: -1}, {Timeout: -1}} {
		s, rules := beamContractState()
		eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
			t.Fatal("evaluated invalid budget")
			return nil, nil
		})
		if r, err := NewBeamSearch(BeamConfig{Budget: budget}).Search(context.Background(), s, rules, eval); err == nil || r != nil {
			t.Fatalf("beam accepted %+v", budget)
		}
		if r, err := NewMCTSSearch(MCTSConfig{Budget: budget}).Search(context.Background(), s, rules, eval); err == nil || r != nil {
			t.Fatalf("MCTS accepted %+v", budget)
		}
	}
}

func TestSearchRejectsInvalidStrategyOptions(t *testing.T) {
	s, rules := beamContractState()
	eval := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
		t.Fatal("evaluated invalid strategy")
		return nil, nil
	})
	if _, err := NewBeamSearch(BeamConfig{BeamWidth: -1}).Search(context.Background(), s, rules, eval); err == nil {
		t.Fatal("accepted negative width")
	}
	for _, cfg := range []MCTSConfig{{Iterations: -1}, {RolloutDepth: -1}, {ExplorationC: -1}, {ExplorationC: math.NaN()}, {ExplorationC: math.Inf(1)}} {
		if _, err := NewMCTSSearch(cfg).Search(context.Background(), s, rules, eval); err == nil {
			t.Fatalf("accepted %+v", cfg)
		}
	}
}

func TestSearchZeroDefaultsAndSignedSeed(t *testing.T) {
	s, rules := beamContractState()
	if _, err := NewBeamSearch(BeamConfig{}).Search(context.Background(), s, rules, &MockEvaluator{}); err != nil {
		t.Fatal(err)
	}
	for _, seed := range []int64{0, -1, 1} {
		if _, err := NewMCTSSearch(MCTSConfig{Seed: seed}).Search(context.Background(), s, rules, &MockEvaluator{}); err != nil {
			t.Fatal(err)
		}
	}
}
