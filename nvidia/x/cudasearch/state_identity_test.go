package cudasearch

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func identityState() *egraph.SearchState {
	g := egraph.New()
	a := g.Add(egraph.ENode{Op: "a"})
	b := g.Add(egraph.ENode{Op: "b"})
	s := egraph.NewSearchState(g, map[string]egraph.Id{"a": a, "b": b})
	s.Derivation = []egraph.Action{{RuleName: "rule", RuleID: "identity", Source: "source", RHSNode: "rhs", ClassID: a, StepIndex: 1, Subst: egraph.Subst{"x": a, "y": b}}}
	return s
}

func TestStateIdentityStableMaps(t *testing.T) {
	s := identityState()
	want := stateFingerprint(s)
	for i := 0; i < 100; i++ {
		other := s.Clone()
		other.Roots = map[string]egraph.Id{"b": s.Roots["b"], "a": s.Roots["a"]}
		other.Derivation[0].Subst = egraph.Subst{"y": s.Roots["b"], "x": s.Roots["a"]}
		if got := stateFingerprint(other); got != want {
			t.Fatal("map order changed branch identity")
		}
	}
}

func TestStateIdentityIncludesReplayFields(t *testing.T) {
	for _, field := range []string{"name", "rule", "source", "rhs", "class", "step", "subst", "path", "root"} {
		t.Run(field, func(t *testing.T) {
			s := identityState()
			want := stateFingerprint(s)
			a := &s.Derivation[0]
			switch field {
			case "name":
				a.RuleName += "other"
			case "rule":
				a.RuleID += "other"
			case "source":
				a.Source += "other"
			case "rhs":
				a.RHSNode += "other"
			case "class":
				a.ClassID = s.Roots["b"]
			case "step":
				a.StepIndex++
			case "subst":
				a.Subst["x"] = s.Roots["b"]
			case "path":
				s.Derivation = append(s.Derivation, *a)
			case "root":
				s.Roots["a"] = s.Roots["b"]
			}
			if stateFingerprint(s) == want {
				t.Fatalf("identity omitted %s", field)
			}
		})
	}
}

func TestStateIdentityFramesStrings(t *testing.T) {
	a := identityState()
	b := identityState()
	a.Derivation[0].RuleName = "name;identity"
	a.Derivation[0].RuleID = ""
	b.Derivation[0].RuleName = "name"
	b.Derivation[0].RuleID = "identity;"
	if stateFingerprint(a) == stateFingerprint(b) {
		t.Fatal("delimiter collision")
	}
	a.Roots = map[string]egraph.Id{"a:1;b": a.Roots["b"]}
	if stateFingerprint(a) == stateFingerprint(identityState()) {
		t.Fatal("root delimiter collision")
	}
}
