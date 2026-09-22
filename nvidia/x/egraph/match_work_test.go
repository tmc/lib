package egraph

import (
	"errors"
	"testing"
)

func TestMatcherChargesAlternativeScans(t *testing.T) {
	g := New()
	root := g.Add(ENode{Op: "value", Data: 0})
	for i := 1; i < 128; i++ {
		g.Union(root, g.Add(ENode{Op: "value", Data: i}))
	}
	g.Rebuild()
	for _, op := range []string{"value", "absent"} {
		p := &Pattern{Op: op}
		if matches, err := p.MatchEClassBounded(g, root, make(Subst), 16); !errors.Is(err, ErrMatchWorkLimit) || matches != nil {
			t.Fatalf("op=%s matches=%d error=%v", op, len(matches), err)
		}
		matches, err := p.MatchEClassBounded(g, root, make(Subst), 129)
		if err != nil {
			t.Fatal(err)
		}
		want := 0
		if op == "value" {
			want = 128
		}
		if len(matches) != want {
			t.Fatalf("op=%s matches=%d want=%d", op, len(matches), want)
		}
	}
}

func TestMatcherStopsPartialExpansion(t *testing.T) {
	g := New()
	x := g.Add(NewNode("x"))
	root := g.Add(NewNode("pair", x, x))
	p, _ := ParsePattern("(pair ?a ?b)")
	work := 0
	matches := p.matchEClassBudget(g, root, make(Subst), &work, 3)
	if len(matches) != 0 || work != 4 {
		t.Fatalf("partial match leaked or work continued: matches=%v work=%d", matches, work)
	}
	matches, err := p.MatchEClassBounded(g, root, make(Subst), 4)
	if err != nil || len(matches) != 1 || matches[0]["?a"] != x || matches[0]["?b"] != x {
		t.Fatalf("matches=%v err=%v", matches, err)
	}
}
