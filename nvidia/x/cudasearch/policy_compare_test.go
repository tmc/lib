package cudasearch_test

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// comparisonEvaluator is a deterministic host oracle, not a timer or compiler.
// One invocation costs one declared comparison unit. It deliberately provides
// no timing receipt and leaves executable qualification absent.
type comparisonEvaluator struct {
	paths []string
	tie   bool
}

func (e *comparisonEvaluator) Evaluate(ctx context.Context, s *egraph.SearchState, _ egraph.Id) (*cudasearch.CandidatePlan, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	name := "base"
	if len(s.Derivation) != 0 {
		name = s.Derivation[len(s.Derivation)-1].RuleName
	}
	e.paths = append(e.paths, name)
	cost := map[string]float64{"base": 100, "a": 30, "b": 20, "z": 10}[name]
	if e.tie && name != "base" {
		cost = 10
	}
	return &cudasearch.CandidatePlan{ID: name, Validated: true, Cost: cost, CostSource: cudasearch.CostEstimated, Derivation: append([]egraph.Action(nil), s.Derivation...)}, nil
}

func comparisonProfile() *cudasearch.ProfileActions {
	now := time.Unix(100, 0).UTC()
	target := cudaprof.ProfileTarget{CodeDigest: strings.Repeat("a", 64), Region: "kernel", Workload: "fixture", Device: "host-fixture", CollectorMode: "synthetic"}
	return &cudasearch.ProfileActions{
		Target: target, Time: now, MaxAge: time.Minute, MinCoverage: 1, MaxUncertaintyNs: 0,
		Snapshot: &cudaprof.ProfileSnapshot{Version: 1, CodeDigest: target.CodeDigest, Region: target.Region, Workload: target.Workload, Device: target.Device, CollectorMode: target.CollectorMode, Timestamp: now, Coverage: 1, Metrics: map[string]int64{"reason": 20}, MetricUnits: map[string]string{"reason": "samples"}},
		Priors:   []cudasearch.MetricRulePrior{{Metric: "reason", Unit: "samples", Minimum: 10, Rule: "z"}},
	}
}

func comparePolicy(ctx context.Context, units int, profile *cudasearch.ProfileActions, tie bool) (*cudasearch.SearchResult, *comparisonEvaluator, int, error) {
	// Admission of an offline profile costs one unit even when it is refused.
	// Timing-only spends that same unit on one additional candidate evaluation.
	profileUnits := 0
	if profile != nil {
		profileUnits = 1
	}
	if units <= profileUnits {
		return nil, nil, 0, fmt.Errorf("comparison: no candidate evaluation budget")
	}
	g := egraph.New()
	id := g.Add(egraph.ENode{Op: "base"})
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": id})
	rules := []*egraph.Rewrite{
		egraph.MustRewrite("a", "base", "first"),
		egraph.MustRewrite("b", "base", "second"),
		egraph.MustRewrite("z", "base", "preferred"),
	}
	budget := cudasearch.DefaultBudget()
	budget.MaxDepth, budget.MaxActions = 1, len(rules)
	budget.MaxStates, budget.MaxEvaluations = units-profileUnits, units-profileUnits
	budget.Timeout = 0
	eval := &comparisonEvaluator{tie: tie}
	result, err := cudasearch.NewBeamSearch(cudasearch.BeamConfig{Budget: budget, Profile: profile, EvaluatorID: "deterministic-host-objective-v1"}).Search(ctx, state, rules, eval)
	return result, eval, profileUnits + len(eval.paths), err
}

func TestPolicyComparisonEqualCostAndAblation(t *testing.T) {
	tests := []struct {
		name     string
		edit     func(*cudasearch.ProfileActions)
		paths    []string
		winner   string
		fallback bool
	}{
		{"timing-only", nil, []string{"base", "a", "b"}, "b", false},
		{"guided", func(*cudasearch.ProfileActions) {}, []string{"base", "z"}, "z", false},
		{"ablated", func(p *cudasearch.ProfileActions) { p.Priors = nil }, []string{"base", "a"}, "a", false},
		{"unmatched-rule", func(p *cudasearch.ProfileActions) { p.Priors[0].Rule = "absent" }, []string{"base", "a"}, "a", false},
		{"wrong-target", func(p *cudasearch.ProfileActions) { p.Target.Workload = "other" }, []string{"base", "a"}, "a", true},
		{"invalid-evidence", func(p *cudasearch.ProfileActions) { p.Snapshot.Coverage = 0 }, []string{"base", "a"}, "a", true},
		{"refused-prior", func(p *cudasearch.ProfileActions) { p.Priors[0].Unit = "cycles" }, []string{"base", "a"}, "a", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for repeat := 0; repeat < 2; repeat++ {
				var profile *cudasearch.ProfileActions
				if tt.edit != nil {
					profile = comparisonProfile()
					tt.edit(profile)
				}
				before, err := json.Marshal(profile)
				if err != nil {
					t.Fatal(err)
				}
				result, eval, spent, err := comparePolicy(context.Background(), 3, profile, false)
				if err != nil {
					t.Fatal(err)
				}
				if spent != 3 || result.Evaluations != len(eval.paths) || result.ActionsAttempted != len(eval.paths)-1 || !reflect.DeepEqual(eval.paths, tt.paths) {
					t.Fatalf("unequal or changed work: spent=%d paths=%v result=%+v", spent, eval.paths, result)
				}
				if result.Winner.ID != tt.winner || (result.ProfileFallback != "") != tt.fallback || result.WorkAccounted {
					t.Fatalf("wrong policy disposition: %+v", result)
				}
				for _, candidate := range result.AllCandidates {
					if candidate.Receipt != nil || candidate.CostSource != cudasearch.CostEstimated {
						t.Fatal("host objective masquerades as measurement")
					}
					want := map[string]float64{"base": 100, "a": 30, "b": 20, "z": 10}[candidate.ID]
					if candidate.Cost != want {
						t.Fatalf("policy changed objective for %s: got %v want %v", candidate.ID, candidate.Cost, want)
					}
				}
				after, err := json.Marshal(profile)
				if err != nil || string(after) != string(before) {
					t.Fatalf("search mutated evidence input: %s -> %s, %v", before, after, err)
				}
			}
		})
	}
}

func TestPolicyComparisonDeterministicTieAndBudget(t *testing.T) {
	// Give both policies enough evaluation allowance to reach all alternatives.
	// Their order differs, but the public deterministic tie contract must agree.
	for _, profile := range []*cudasearch.ProfileActions{nil, comparisonProfile()} {
		units := 4
		if profile != nil {
			units++
		}
		result, _, _, err := comparePolicy(context.Background(), units, profile, true)
		if err != nil || result.Winner.ID != "a" {
			t.Fatalf("tie result=%+v err=%v", result, err)
		}
	}
	if _, _, _, err := comparePolicy(context.Background(), 1, comparisonProfile(), false); err == nil {
		t.Fatal("admitted search without evaluation budget")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, eval, spent, err := comparePolicy(ctx, 3, nil, false); err == nil || len(eval.paths) != 0 || spent != 0 {
		t.Fatalf("canceled search performed work: %v %d %v", eval.paths, spent, err)
	}
}

func ExampleBeamSearch_policyComparison() {
	for _, arm := range []struct {
		name    string
		profile *cudasearch.ProfileActions
	}{
		{"timing-only", nil}, {"guided", comparisonProfile()},
	} {
		result, _, spent, err := comparePolicy(context.Background(), 3, arm.profile, false)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(arm.name, spent, result.Winner.ID, result.Winner.CostSource)
	}
	// Output:
	// timing-only 3 b estimated
	// guided 3 z estimated
}
