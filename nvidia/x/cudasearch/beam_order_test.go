package cudasearch

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestBeamFullPathPruning(t *testing.T) {
	s, _ := beamContractState()
	rules := []*egraph.Rewrite{
		egraph.MustRewrite("a", "base", "A"), egraph.MustRewrite("b", "base", "B"),
		egraph.MustRewrite("y", "A", "AY"), egraph.MustRewrite("z", "A", "AZ"),
		egraph.MustRewrite("x", "B", "BX"), egraph.MustRewrite("goal", "AZ", "GOAL"),
	}
	eval := beamEvaluatorFunc(func(_ context.Context, s *egraph.SearchState, _ egraph.Id) (*CandidatePlan, error) {
		names := make([]string, len(s.Derivation))
		for i, action := range s.Derivation {
			names[i] = action.RuleName
		}
		path := strings.Join(names, "/")
		cost, ok := map[string]float64{"": 9, "a": 5, "b": 5, "a/y": 2, "a/z": 2, "b/x": 2, "a/z/goal": 1}[path]
		if !ok {
			return nil, fmt.Errorf("fixture rejects path %s", path)
		}
		return &CandidatePlan{ID: path, Cost: cost, Validated: true, Derivation: s.Derivation}, nil
	})
	r, err := NewBeamSearch(BeamConfig{BeamWidth: 2, Budget: SearchBudget{MaxDepth: 3}}).Search(context.Background(), s, rules, eval)
	if err != nil {
		t.Fatal(err)
	}
	// Last-action ordering retains b/x and a/y, losing a/z and its goal.
	if r.Winner.ID != "a/z/goal" {
		t.Fatalf("winner=%s", r.Winner.ID)
	}
	if r.PolicyID.Options["tie_order"] != "full-replay-path-v1" {
		t.Fatal("tie policy not recorded")
	}
}

func TestReplayPathOrder(t *testing.T) {
	a := []egraph.Action{{RuleName: "a", Subst: egraph.Subst{"z": 2, "x": 1}}, {RuleName: "z"}}
	b := []egraph.Action{{RuleName: "b"}, {RuleName: "a"}}
	if compareDerivations(a, b) >= 0 || compareDerivations(b, a) <= 0 {
		t.Fatal("last action overrode prefix")
	}
	c := append([]egraph.Action(nil), a...)
	c[0].Subst = egraph.Subst{"x": 1, "z": 2}
	if compareDerivations(a, c) != 0 {
		t.Fatal("map order affected tie")
	}
	if compareDerivations(a[:1], a) >= 0 {
		t.Fatal("path prefix must sort first")
	}
	for _, field := range []string{"rule", "source", "rhs", "class", "step", "subst"} {
		t.Run(field, func(t *testing.T) {
			c := append([]egraph.Action(nil), a...)
			switch field {
			case "rule":
				c[0].RuleID = "different"
			case "source":
				c[0].Source = "different"
			case "rhs":
				c[0].RHSNode = "different"
			case "class":
				c[0].ClassID++
			case "step":
				c[0].StepIndex++
			case "subst":
				c[0].Subst = egraph.Subst{"x": 9, "z": 2}
			}
			if compareDerivations(a, c) == 0 {
				t.Fatal("replay field omitted")
			}
		})
	}
}
