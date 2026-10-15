package cudasearch

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func actionProfile() *ProfileActions {
	now := time.Unix(100, 0)
	target := cudaprof.ProfileTarget{CodeDigest: strings.Repeat("a", 64), Region: "kernel", Workload: "work", Device: "device", CollectorMode: "synthetic"}
	return &ProfileActions{Target: target, Time: now, MaxAge: time.Minute, MinCoverage: 0.5, MaxUncertaintyNs: 10, Snapshot: &cudaprof.ProfileSnapshot{Version: 1, CodeDigest: target.CodeDigest, Region: target.Region, Workload: target.Workload, Device: target.Device, CollectorMode: target.CollectorMode, Timestamp: now, Coverage: 1, Metrics: map[string]int64{"reason": 20}, MetricUnits: map[string]string{"reason": "samples"}}, Priors: []MetricRulePrior{{Metric: "reason", Unit: "samples", Minimum: 10, Rule: "z"}}}
}

func TestProfileActionsReachBothStrategies(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			s, _ := beamContractState()
			rules := []*egraph.Rewrite{egraph.MustRewrite("a", "base", "first"), egraph.MustRewrite("z", "base", "preferred")}
			profile := actionProfile()
			budget := DefaultBudget()
			budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
			budget.MaxStates = 2
			budget.MaxEvaluations = 2
			eval := beamEvaluatorFunc(func(_ context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
				// Mutating the source after search start must not alter its decision.
				profile.Snapshot.Metrics["reason"] = 0
				profile.Priors[0].Rule = "a"
				return &CandidatePlan{Validated: true, Cost: float64(10 - len(s.Derivation)), Derivation: append([]egraph.Action(nil), s.Derivation...)}, nil
			})
			var result *SearchResult
			var err error
			if strategy == "beam" {
				result, err = NewBeamSearch(BeamConfig{Budget: budget, Profile: profile}).Search(context.Background(), s, rules, eval)
			} else {
				result, err = NewMCTSSearch(MCTSConfig{Budget: budget, Profile: profile}).Search(context.Background(), s, rules, eval)
			}
			if err != nil {
				t.Fatal(err)
			}
			if result.ProfileDecision == nil || result.ProfileDecision.ID == "" || result.ProfileDecision.Priors[0].Rule != "z" {
				t.Fatalf("missing frozen policy receipt: %+v", result.ProfileDecision)
			}
			if len(result.Winner.Derivation) != 1 || result.Winner.Derivation[0].RuleName != "z" || result.Winner.Cost != 9 || result.ProfileFallback != "" || result.ProfileGuidance.StallCounts["reason"] != 20 {
				t.Fatalf("profile not consumed safely: %+v", result)
			}
		})
	}
}

func TestProfileActionsFallback(t *testing.T) {
	for _, kind := range []string{"missing", "stale", "identity", "unit", "coverage", "losses", "clock"} {
		t.Run(kind, func(t *testing.T) {
			p := actionProfile()
			switch kind {
			case "missing":
				p.Snapshot = nil
			case "stale":
				p.Snapshot.Timestamp = p.Time.Add(-time.Hour)
			case "identity":
				p.Target.CodeDigest = strings.Repeat("b", 64)
			case "unit":
				p.Priors[0].Unit = "cycles"
			case "coverage":
				p.Snapshot.Coverage = 0.1
			case "losses":
				p.Snapshot.Losses = 1
			case "clock":
				p.Snapshot.ClockUncertaintyNs = -1
			}
			frozen := freezeProfileActions(p)
			if frozen.fallback == "" || len(frozen.ranks) != 0 || frozen.guidance != nil {
				t.Fatalf("bad fallback: %+v", frozen)
			}
			s, rules := beamContractState()
			budget := DefaultBudget()
			budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
			budget.MaxStates, budget.MaxEvaluations = 2, 2
			result, err := NewBeamSearch(BeamConfig{Budget: budget, Profile: p}).Search(context.Background(), s, rules, &MockEvaluator{})
			if err != nil || result.ProfileFallback == "" || result.ProfileGuidance != nil || result.Evaluations != 2 {
				t.Fatalf("fallback prevented static search: %+v %v", result, err)
			}
		})
	}
}

func ExampleProfileActions() {
	// This fixture supplies synthetic, fully attributed sample counts and an
	// explicit samples threshold. Physical collection is a separate operation.
	profile := actionProfile()
	state, rules := beamContractState()
	profile.Priors[0].Rule = "alternative"
	budget := DefaultBudget()
	budget.CallbackMemoryBytes = fixtureCallbackMemoryBytes(budget.MaxDepth)
	budget.MaxStates, budget.MaxEvaluations = 2, 2
	result, err := NewBeamSearch(BeamConfig{Budget: budget, Profile: profile}).Search(context.Background(), state, rules, &MockEvaluator{})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Evaluations, result.ProfileGuidance.StallCounts["reason"])
	// Output: 2 20
}
