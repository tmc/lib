package egraph

import "testing"

func TestBoundedSearchNestedSingleMatch(t *testing.T) {
	g := New()
	x := g.Add(NewNode("x"))
	y := g.Add(NewNode("y"))
	inner := g.Add(NewNode("inner", x))
	root := g.Add(NewNode("outer", inner, y))
	rule := MustRewrite("nested", "(outer (inner ?x) ?y)", "(fused ?x ?y)")
	matches, limited := rule.LHS.SearchBounded(g, 1)
	if len(matches) != 1 || !limited || matches[0].ClassID != root || matches[0].Subst["?x"] != x || matches[0].Subst["?y"] != y {
		t.Fatalf("matches=%+v limited=%v", matches, limited)
	}
	s := NewSearchState(g, map[string]Id{"root": root})
	actions := EnumerateActionsBounded(g, []*Rewrite{rule}, 1)
	if len(actions) != 1 {
		t.Fatalf("actions=%d", len(actions))
	}
	next, err := s.ApplyAction(actions[0])
	if err != nil || next.NoOp {
		t.Fatalf("transition=%v err=%v", next, err)
	}
	if _, err := ReplaySearchState(s, []*Rewrite{rule}, next.State.Derivation); err != nil {
		t.Fatal(err)
	}
}

func TestBoundedSearchMatchAllowance(t *testing.T) {
	g := New()
	for i := 0; i < 4; i++ {
		g.Add(NewLeaf(i))
	}
	p := Var("?x")
	for _, limit := range []int{0, 1, 2, 4, 5} {
		matches, limited := p.SearchBounded(g, limit)
		want := min(limit, 4)
		if len(matches) != want || limited != (limit > 0 && limit <= 4) {
			t.Fatalf("limit=%d matches=%d limited=%v", limit, len(matches), limited)
		}
	}
	missing, _ := ParsePattern("absent")
	if matches, limited := missing.SearchBounded(g, 1); len(matches) != 0 || limited {
		t.Fatalf("absent matches=%v limited=%v", matches, limited)
	}
}

func TestEnumerationStopsAtActionAllowance(t *testing.T) {
	g := New()
	g.Add(NewNode("x"))
	rule := MustRewrite("first", "x", "y")
	// A later rule is intentionally unavailable: the full allowance must stop
	// traversal before dereferencing it.
	actions := EnumerateActionsBounded(g, []*Rewrite{rule, nil}, 1)
	if len(actions) != 1 || actions[0].Rule != rule {
		t.Fatalf("actions=%v", actions)
	}
}
