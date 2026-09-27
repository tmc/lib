package egraph

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatensor"
)

func TestSOLCostRanksByFusedNotPrefetched(t *testing.T) {
	makeGraph := func(nodes []cudatensor.EinsumNode) *cudatensor.CanonicalGraph {
		g, err := cudatensor.CanonicalizeRanks(&cudatensor.Graph{Nodes: nodes})
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	phaseSeparated := makeGraph([]cudatensor.EinsumNode{
		{ID: "input", OutputRank: 1, OutputTensors: []cudatensor.TensorInfo{{Name: "x", Elements: 0}}},
		{ID: "compute", Equation: "A->A", Inputs: []string{"input"}, MACs: 100,
			InputTensors: []cudatensor.TensorInfo{{Name: "x", Elements: 0}}, OutputTensors: []cudatensor.TensorInfo{{Name: "y", Elements: 0}}},
		{ID: "memory", Equation: "A->A", Inputs: []string{"compute"},
			InputTensors: []cudatensor.TensorInfo{{Name: "y", Elements: 0}}, OutputTensors: []cudatensor.TensorInfo{{Name: "z", Elements: 100}}},
	})
	balanced := makeGraph([]cudatensor.EinsumNode{
		{ID: "input", OutputRank: 1, OutputTensors: []cudatensor.TensorInfo{{Name: "x", Elements: 60}}},
		{ID: "balanced", Equation: "A->A", Inputs: []string{"input"}, MACs: 120,
			InputTensors: []cudatensor.TensorInfo{{Name: "x", Elements: 60}}, OutputTensors: []cudatensor.TensorInfo{{Name: "y", Elements: 60}}},
	})
	ceiling := cudatensor.Ceiling{MACsPerCycle: 1, BytesPerCycle: 1, FrequencyGHz: 1, BytesPerElement: 1}
	a, err := cudatensor.ComputeSOLBounds(phaseSeparated, ceiling)
	if err != nil {
		t.Fatal(err)
	}
	b, err := cudatensor.ComputeSOLBounds(balanced, ceiling)
	if err != nil {
		t.Fatal(err)
	}
	if !(a.FusedPrefetched < b.FusedPrefetched && b.Fused < a.Fused) {
		t.Fatalf("fixture does not separate ranking objectives: A=%+v B=%+v", a, b)
	}
	ranked, err := RankSOLCandidates([]SOLCandidate{{Name: "phase-separated", Graph: phaseSeparated}, {Name: "balanced", Graph: balanced}}, SOLCost(ceiling))
	if err != nil {
		t.Fatal(err)
	}
	if len(ranked) != 2 || ranked[0].Candidate.Name != "balanced" {
		t.Fatalf("Fused ranking = %+v", ranked)
	}
}

func TestSOLCostRequiresCanonicalGraphForExpression(t *testing.T) {
	_, err := SOLCost(cudatensor.Ceiling{MACsPerCycle: 1, BytesPerCycle: 1, FrequencyGHz: 1, BytesPerElement: 1})(SOLCandidate{Name: "rewrite", Expr: &Expr{Op: "add"}})
	if !errors.Is(err, ErrMissingSOLGraph) || !strings.Contains(err.Error(), "rewrite") || !strings.Contains(err.Error(), "tensor metrics") {
		t.Fatalf("error = %v", err)
	}
}
