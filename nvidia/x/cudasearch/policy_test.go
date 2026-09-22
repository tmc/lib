package cudasearch

import (
	"context"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSearchPolicyPreservesStaleEvidenceFallback(t *testing.T) {
	s, rules := beamContractState()
	eval := &MockEvaluator{Evidence: map[string]*ProfileEvidence{
		"stale": {Epoch: "old", Occupancy: math.NaN()},
	}}
	r, err := NewBeamSearch(BeamConfig{}).Search(context.Background(), s, rules, eval)
	if err != nil {
		t.Fatal(err)
	}
	if r.PolicyComplete || r.PolicyID.Options["scoring_evidence_error"] == "" || r.Winner == nil {
		t.Fatal("unencodable evidence must preserve search with incomplete policy identity")
	}
}

func TestSearchPolicyIdentity(t *testing.T) {
	s, rules := beamContractState()
	run := func(cfg BeamConfig, rs []*egraph.Rewrite, eval Evaluator) *SearchResult {
		r, err := NewBeamSearch(cfg).Search(context.Background(), s, rs, eval)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	base := run(BeamConfig{}, rules, &MockEvaluator{})
	if !base.PolicyComplete || base.PolicyID.Strategy != "beam" || base.PolicyID.RuleSet == "" {
		t.Fatalf("missing policy: %+v", base.PolicyID)
	}
	if again := run(BeamConfig{}, rules, &MockEvaluator{}); again.PolicyID.String() != base.PolicyID.String() {
		t.Fatal("stable normalized policy changed")
	}
	for _, kind := range []string{"width", "budget", "rule", "scoring"} {
		t.Run(kind, func(t *testing.T) {
			cfg := BeamConfig{}
			rs := rules
			eval := &MockEvaluator{}
			switch kind {
			case "width":
				cfg.BeamWidth = 8
			case "budget":
				cfg.Budget.MaxEvaluations = 1
			case "rule":
				rs = []*egraph.Rewrite{egraph.MustRewrite("alternative", "base", "different")}
			case "scoring":
				eval.Scoring = DefaultScoringModel("different")
			}
			if got := run(cfg, rs, eval); got.PolicyID.String() == base.PolicyID.String() {
				t.Fatalf("policy omitted %s", kind)
			}
		})
	}
	custom := beamEvaluatorFunc(func(context.Context, *egraph.SearchState, egraph.Id) (*CandidatePlan, error) {
		return &CandidatePlan{ID: "same-artifact", Cost: 1, Validated: true}, nil
	})
	unknown := run(BeamConfig{}, rules, custom)
	known := run(BeamConfig{EvaluatorID: "qualified-evaluator-v1"}, rules, custom)
	if unknown.PolicyComplete || !known.PolicyComplete || unknown.PolicyID.String() == known.PolicyID.String() || unknown.Winner.ID != known.Winner.ID {
		t.Fatal("evaluator policy mixed with artifact identity")
	}
}

func TestMCTSPolicyRecordsEffectiveSeed(t *testing.T) {
	s, rules := beamContractState()
	a, err := NewMCTSSearch(MCTSConfig{}).Search(context.Background(), s, rules, &MockEvaluator{})
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewMCTSSearch(MCTSConfig{Seed: 43}).Search(context.Background(), s, rules, &MockEvaluator{})
	if err != nil {
		t.Fatal(err)
	}
	if a.PolicyID.Seed != 42 || b.PolicyID.Seed != 43 || a.PolicyID.String() == b.PolicyID.String() {
		t.Fatal("policy omitted normalized seed")
	}
}
