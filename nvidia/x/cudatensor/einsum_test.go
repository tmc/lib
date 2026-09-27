package cudatensor

import (
	"errors"
	"strings"
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

func TestCanonicalizeRanksFreshLabelsHaveStableOrder(t *testing.T) {
	graphs := []*Graph{
		{Nodes: []EinsumNode{{ID: "input", OutputRank: 1}, {ID: "broadcast", Equation: "x->xyz", Inputs: []string{"input"}}}},
		{Nodes: []EinsumNode{{ID: "input", OutputRank: 1}, {ID: "broadcast", Equation: "q->qrs", Inputs: []string{"input"}}}},
	}
	var key string
	for run := 0; run < 50; run++ {
		for i, graph := range graphs {
			canonical, err := CanonicalizeRanks(graph)
			if err != nil {
				t.Fatal(err)
			}
			if got, want := canonical.Nodes[1].Equation, "A->ABC"; got != want {
				t.Fatalf("canonical equation = %q, want %q", got, want)
			}
			got := CanonicalKey(graph)
			if i == 0 && run == 0 {
				key = got
			} else if got != key {
				t.Fatalf("canonical key changed across alpha-renamed graph or run %d", run)
			}
		}
	}
}

func TestCanonicalizeRanksMintsOutputRanksBeforeOperandOnlyRanks(t *testing.T) {
	g := &Graph{Nodes: []EinsumNode{{ID: "op", Equation: "ij->ji", Inputs: []string{""}}}}
	canonical, err := CanonicalizeRanks(g)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := canonical.Nodes[0].Equation, "BA->AB"; got != want {
		t.Fatalf("canonical equation = %q, want %q", got, want)
	}
}

func TestCanonicalizeRanksRejectsConflictingShapeRank(t *testing.T) {
	g := &Graph{Nodes: []EinsumNode{
		{ID: "p", OutputRank: 2},
		{ID: "t1", Equation: "ij->ij", Inputs: []string{"p"}},
		{ID: "t2", Equation: "ji->ij", Inputs: []string{"t1"}},
	}, Shape: map[string]string{"I": "I", "J": "J"}}
	_, err := CanonicalizeRanks(g)
	if err == nil || !strings.Contains(err.Error(), `shape rank "J"`) {
		t.Fatalf("error = %v, want conflict naming rank J", err)
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
	keyGraph := func(equation string, rank int) *Graph {
		return &Graph{Nodes: []EinsumNode{{ID: "left", OutputRank: rank}, {ID: "right", OutputRank: rank}, {ID: "out", Equation: equation, Inputs: []string{"left", "right"}}}}
	}
	for _, pair := range [][2]*Graph{
		{keyGraph("A,A->A", 1), keyGraph("A,B->A", 1)},
		{keyGraph("AB,BC->AC", 2), keyGraph("AB,AB->AB", 2)},
	} {
		if CanonicalKey(pair[0]) == CanonicalKey(pair[1]) {
			t.Fatalf("different einsum structures share key: %q vs %q", pair[0].Nodes[2].Equation, pair[1].Nodes[2].Equation)
		}
	}
	withMetrics := func(macs, elements uint64) *Graph {
		return &Graph{Nodes: []EinsumNode{
			{ID: "input", OutputRank: 1, OutputTensors: []TensorInfo{{Name: "x", Elements: 4}}},
			{ID: "op", Equation: "A->A", Inputs: []string{"input"}, MACs: macs,
				InputTensors: []TensorInfo{{Name: "x", Elements: elements}}, OutputTensors: []TensorInfo{{Name: "y", Elements: 4}}},
		}}
	}
	if CanonicalKey(withMetrics(10, 4)) == CanonicalKey(withMetrics(11, 4)) {
		t.Fatal("MAC count was omitted from canonical key")
	}
	if CanonicalKey(withMetrics(10, 4)) == CanonicalKey(withMetrics(10, 5)) {
		t.Fatal("input element count was omitted from canonical key")
	}
}

func TestCanonicalizeRanksRenamesShapeNamespace(t *testing.T) {
	a := &Graph{Nodes: []EinsumNode{{ID: "op", Equation: "i->i", Inputs: []string{""}}}, Shape: map[string]string{"I": "64"}}
	b := &Graph{Nodes: []EinsumNode{{ID: "op", Equation: "j->j", Inputs: []string{""}}}, Shape: map[string]string{"J": "64"}}
	ca, err := CanonicalizeRanks(a)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := CanonicalizeRanks(b)
	if err != nil {
		t.Fatal(err)
	}
	if CanonicalKey(a) != CanonicalKey(b) || ca.Shape["A"] != "64" || cb.Shape["A"] != "64" {
		t.Fatalf("alpha-renamed shape mismatch: keyA=%q keyB=%q shapeA=%v shapeB=%v", CanonicalKey(a), CanonicalKey(b), ca.Shape, cb.Shape)
	}
}

func TestCanonicalGraphRejectsMutation(t *testing.T) {
	g, err := CanonicalizeRanks(&Graph{Nodes: []EinsumNode{{ID: "input", OutputRank: 1}, {ID: "op", Equation: "i->i", Inputs: []string{"input"}}}})
	if err != nil {
		t.Fatal(err)
	}
	g.Nodes[1].Equation = "q->q"
	if err := ExportTimeloopWorkload(g, &strings.Builder{}); !errors.Is(err, ErrNotCanonicalized) {
		t.Fatalf("mutated export error = %v", err)
	}
	if _, err := ComputeSOLBounds(g, Ceiling{MACsPerCycle: 1, BytesPerCycle: 1, FrequencyGHz: 1, BytesPerElement: 1}); !errors.Is(err, ErrNotCanonicalized) {
		t.Fatalf("mutated SOL error = %v", err)
	}
}

func TestCanonicalGraphRejectsShapeMutation(t *testing.T) {
	g, err := CanonicalizeRanks(&Graph{Nodes: []EinsumNode{
		{ID: "input", OutputRank: 1},
		{ID: "op", Equation: "i->i", Inputs: []string{"input"}},
	}, Shape: map[string]string{"I": "64"}})
	if err != nil {
		t.Fatal(err)
	}
	g.Shape["A"] = "128"
	if err := ExportTimeloopWorkload(g, &strings.Builder{}); !errors.Is(err, ErrNotCanonicalized) {
		t.Fatalf("mutated shape export error = %v", err)
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
