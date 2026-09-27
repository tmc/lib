package cudatensor

import (
	"errors"
	"testing"
)

func TestCanonicalizeRanks_DeterministicAndAlphaEquivalent(t *testing.T) {
	a := &Graph{Nodes: []EinsumNode{
		{ID: "left", OutputRank: 2},
		{ID: "right", OutputRank: 2},
		{ID: "mm", Equation: "xy,yz->xz", Inputs: []string{"left", "right"}},
	}}
	b := &Graph{Nodes: []EinsumNode{
		{ID: "left", OutputRank: 2},
		{ID: "right", OutputRank: 2},
		{ID: "mm", Equation: "AB,BC->AC", Inputs: []string{"left", "right"}},
	}}
	ca, err := CanonicalizeRanks(a)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := CanonicalizeRanks(b)
	if err != nil {
		t.Fatal(err)
	}
	if ca.Nodes[2].Equation != "AB,BD->AD" || CanonicalKey(a) != CanonicalKey(b) {
		t.Fatalf("canonical forms differ: %q, %q", ca.Nodes[2].Equation, cb.Nodes[2].Equation)
	}
	if ca.Nodes[0].Start == false || ca.Nodes[1].Start == false {
		t.Fatal("start nodes were not retained as external inputs")
	}
}

func TestCanonicalizeRanks_DefersUntilPredecessorsReady(t *testing.T) {
	g := &Graph{Nodes: []EinsumNode{
		{ID: "short", OutputRank: 1},
		{ID: "long", OutputRank: 1},
		{ID: "middle", Equation: "C->C", Inputs: []string{"long"}},
		{ID: "join", Equation: "X,Y->X", Inputs: []string{"short", "middle"}},
	}}
	got, err := CanonicalizeRanks(g)
	if err != nil {
		t.Fatal(err)
	}
	if got.Nodes[3].Equation != "A,B->A" {
		t.Fatalf("join equation = %q", got.Nodes[3].Equation)
	}
}

func TestCanonicalizeRanks_CycleFails(t *testing.T) {
	g := &Graph{Nodes: []EinsumNode{
		{ID: "a", Equation: "A->A", Inputs: []string{"b"}},
		{ID: "b", Equation: "A->A", Inputs: []string{"a"}},
	}}
	if _, err := CanonicalizeRanks(g); !errors.Is(err, ErrCyclicGraph) {
		t.Fatalf("error = %v", err)
	}
	if CanonicalKey(g) == CanonicalKey(&Graph{}) {
		t.Fatal("invalid graph key collides with the empty graph")
	}
}

func TestCanonicalizeRanks_ReachableNodeWithCyclicPredecessorFails(t *testing.T) {
	g := &Graph{Nodes: []EinsumNode{
		{ID: "input", OutputRank: 1},
		{ID: "join", Equation: "A,A->A", Inputs: []string{"input", "a"}},
		{ID: "a", Equation: "A->A", Inputs: []string{"b"}},
		{ID: "b", Equation: "A->A", Inputs: []string{"a"}},
	}}
	if _, err := CanonicalizeRanks(g); !errors.Is(err, ErrCyclicGraph) {
		t.Fatalf("error = %v", err)
	}
}

func TestCanonicalKeyIncludesStructureAndDimensionBounds(t *testing.T) {
	base := &Graph{Nodes: []EinsumNode{{ID: "input", OutputRank: 1}}, Shape: map[string]string{"A": "N"}}
	changedBound := &Graph{Nodes: []EinsumNode{{ID: "input", OutputRank: 1}}, Shape: map[string]string{"A": "M"}}
	changedRank := &Graph{Nodes: []EinsumNode{{ID: "input", OutputRank: 2}}, Shape: map[string]string{"A": "N"}}
	if CanonicalKey(base) == CanonicalKey(changedBound) || CanonicalKey(base) == CanonicalKey(changedRank) {
		t.Fatal("different graph bounds or rank structure shared a canonical key")
	}
}

func TestLabelTokenSequence(t *testing.T) {
	for i, want := range []string{"A", "Z", "A0", "Z0", "A1"} {
		index := []int{0, 25, 26, 51, 52}[i]
		if got := string(labelToken(index)); got != want {
			t.Errorf("labelToken(%d) = %q, want %q", index, got, want)
		}
	}
}
