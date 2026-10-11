package egraph

import (
	"context"
	"testing"
)

type provenanceCost struct{}

func (provenanceCost) NodeCost(n ENode, children []float64) float64 {
	cost := 1.0
	if n.Op == "new" {
		cost = 0.5
	}
	for _, c := range children {
		cost += c
	}
	return cost
}

func TestExtractedProvenanceEveryExtractor(t *testing.T) {
	g := New()
	x := g.Add(NewLeaf("x"))
	root := g.Add(NewNode("old", x))
	r := MustRewrite("introduce-new", "(old ?x)", "(new ?x)")
	if r.Apply(g) != 1 {
		t.Fatal("rule did not instantiate")
	}
	check := func(e *Expr, cost float64, err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		if e.Fingerprint() != "(new x{string:x})" || cost != 1.5 {
			t.Fatal(e.Fingerprint(), cost)
		}
		if e.Class != g.Find(root) || e.Origin != "introduce-new" || len(e.Children) != 1 || e.Children[0].Class != g.Find(x) || e.Children[0].Origin != "input" {
			t.Fatalf("provenance root=%+v child=%+v", e, e.Children[0])
		}
	}
	check(ExtractBest(g, root, provenanceCost{}))
	check(ExtractBounded(context.Background(), g, root, provenanceCost{}, DefaultExtractionLimits()))
	e, c, _, err := ExtractBoundedWithStats(context.Background(), g, root, provenanceCost{}, DefaultExtractionLimits())
	check(e, c, err)
	top, err := ExtractTopK(g, root, 2, provenanceCost{})
	if err != nil {
		t.Fatal(err)
	}
	if len(top) != 2 {
		t.Fatal(len(top))
	}
	for _, candidate := range top {
		if candidate.Expr.Op == "new" && candidate.Expr.Origin != "introduce-new" || candidate.Expr.Op == "old" && candidate.Expr.Origin != "input" {
			t.Fatal(candidate.Expr)
		}
	}
	clone := g.Clone()
	check(ExtractBest(clone, root, provenanceCost{}))
	// Reusing an RHS already introduced as input must retain its first origin.
	g2 := New()
	leaf := g2.Add(NewLeaf("x"))
	input := g2.Add(NewNode("new", leaf))
	old := g2.Add(NewNode("old", leaf))
	r.Apply(g2)
	e2, _, err := ExtractBest(g2, old, provenanceCost{})
	if err != nil || e2.Origin != "input" || e2.Class != g2.Find(input) {
		t.Fatal(e2, err)
	}
}

func TestProvenanceSurvivesChildCanonicalization(t *testing.T) {
	g := New()
	a := g.Add(NewLeaf("a"))
	b := g.Add(NewLeaf("b"))
	parent := g.addFrom(NewNode("new", b), "parent-rule")
	g.Union(a, b)
	g.Union(parent, g.Add(NewNode("costly", a, a, a)))
	g.Rebuild()
	e, _, err := ExtractBest(g, parent, AstSizeCost{})
	if err != nil || e.Origin != "parent-rule" || e.Children[0].Class != g.Find(a) {
		t.Fatal(e, err)
	}
}

func FuzzProvenanceMetadataDoesNotChangeExtraction(f *testing.F) {
	f.Add([]byte{0, 1, 2, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 32 {
			return
		}
		g := New()
		a := g.Add(NewLeaf("a"))
		b := g.Add(NewLeaf("b"))
		root := a
		for _, v := range data {
			root = g.addFrom(NewNode([]string{"add", "mul", "silu"}[v%3], root, b), "fuzz-rule")
		}
		before, cost, err := ExtractBest(g, root, AstSizeCost{})
		if err != nil {
			t.Fatal(err)
		}
		noMetadata := g.Clone()
		noMetadata.origins = nil
		after, afterCost, err := ExtractBest(noMetadata, root, AstSizeCost{})
		if err != nil || before.Fingerprint() != after.Fingerprint() || cost != afterCost {
			t.Fatal("provenance changed extraction", err)
		}
	})
}

func TestBlankRewriteNameProvenanceEveryExtractor(t *testing.T) {
	g := New()
	x := g.Add(NewLeaf("x"))
	root := g.Add(NewNode("old", x))
	r, err := NewRewrite("", "(old ?x)", "(new ?x)")
	if err != nil {
		t.Fatal(err)
	}
	if r.Apply(g) != 1 {
		t.Fatal("blank named rewrite admission changed")
	}
	check := func(expr *Expr, cost float64, err error) {
		t.Helper()
		if err != nil || expr.Op != "new" || expr.Origin != "" || expr.Class != g.Find(root) || expr.Children[0].Origin != "input" || cost != 1.5 {
			t.Fatalf("blank rule origin replaced: expr=%+v cost=%v error=%v", expr, cost, err)
		}
	}
	check(ExtractBest(g, root, provenanceCost{}))
	check(ExtractBounded(context.Background(), g, root, provenanceCost{}, DefaultExtractionLimits()))
	expr, cost, _, err := ExtractBoundedWithStats(context.Background(), g, root, provenanceCost{}, DefaultExtractionLimits())
	check(expr, cost, err)
	top, err := ExtractTopK(g, root, 2, provenanceCost{})
	if err != nil {
		t.Fatal(err)
	}
	if len(top) != 2 {
		t.Fatal(len(top))
	}
	for _, candidate := range top {
		if candidate.Expr.Op == "new" {
			check(candidate.Expr, candidate.Cost, nil)
		}
	}
	check(ExtractBest(g.Clone(), root, provenanceCost{}))
}
