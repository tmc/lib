package egraph

import (
	"fmt"
	"reflect"
	"testing"
)

func TestNoOpLookupMatchesInstantiation(t *testing.T) {
	g := New()
	a := g.Add(NewNode("a"))
	b := g.Add(NewNode("b"))
	f := g.Add(NewNode("f", a))
	g.Add(NewLeaf("?missing"))
	g.Union(a, b)
	// Test both a dirty memo and the rebuilt graph against old semantics.
	for _, rebuild := range []bool{false, true} {
		if rebuild {
			g.Rebuild()
		}
		for _, text := range []string{"a", "b", "missing", "?x", "?missing", "(f ?x)", "(f missing)", "(f (f ?x))"} {
			p, err := ParsePattern(text)
			if err != nil {
				t.Fatal(err)
			}
			for _, target := range []Id{a, b, f} {
				subst := Subst{"?x": b}
				probe := g.Clone()
				want := probe.Find(p.Instantiate(probe, subst)) == probe.Find(target)
				before := g.Clone()
				got := patternIsExistingClass(g, p, subst, target)
				if got != want {
					t.Fatalf("rebuilt=%v pattern=%s target=%d got=%v want=%v", rebuild, text, target, got, want)
				}
				if !reflect.DeepEqual(g.Clone(), before) {
					t.Fatal("lookup mutated graph")
				}
			}
		}
	}
}

func TestNoOpLookupBounds(t *testing.T) {
	g := New()
	id := g.Add(NewNode("a"))
	p := &Pattern{Op: "cycle"}
	p.Children = []*Pattern{p}
	if patternIsExistingClass(g, p, nil, id) {
		t.Fatal("cyclic pattern recognized")
	}
	if patternIsExistingClass(g, &Pattern{Op: "a"}, nil, Id(g.storage().Len())) {
		t.Fatal("invalid class ID accepted")
	}
}

func TestNoOpLookupDoesNotCompressParents(t *testing.T) {
	g := New()
	a := g.Add(NewNode("a"))
	b := g.Add(NewNode("b"))
	c := g.Add(NewNode("c"))
	// Build a valid uncompressed chain through public union operations.
	g.Union(b, a)
	g.Union(c, b)
	roots := func() []Id {
		result := make([]Id, g.storage().Len())
		for id := range result {
			root, _, ok := g.storage().FindWithoutCompression(Id(id), g.storage().Len())
			if !ok {
				t.Fatalf("invalid union-find at ID %d", id)
			}
			result[id] = root
		}
		return result
	}
	before := roots()
	if !patternIsExistingClass(g, Var("?x"), Subst{"?x": a}, c) {
		t.Fatal("missed equivalent parent chain")
	}
	after := roots()
	for i := range before {
		if before[i] != after[i] {
			t.Fatal("lookup compressed parent chain")
		}
	}
}

func TestNoOpLookupDoesNotCopyUnrelatedGraph(t *testing.T) {
	g := New()
	id := g.Add(NewNode("a"))
	p := &Pattern{Op: "a"}
	action := ActionCandidate{Rule: &Rewrite{RHS: p}, Match: MatchMatch{ClassID: id}}
	run := func() {
		if !actionIsNoOp(g, action) {
			panic("fixture lookup failed")
		}
	}
	small := testing.AllocsPerRun(10, run)
	for i := 0; i < 1000; i++ {
		g.Add(NewNode(fmt.Sprintf("unrelated%d", i)))
	}
	large := testing.AllocsPerRun(10, run)
	if large > small+2 {
		t.Fatalf("lookup allocations scale with unrelated graph: small=%g large=%g", small, large)
	}
}
