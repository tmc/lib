package cudaautograd_test

import (
	"errors"
	"maps"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
)

type checkFixture struct {
	graph  *cudair.Graph
	loss   *cudair.Node
	target *cudair.Node
	result *cudaautograd.BackwardResult
	inputs map[string][]float32
}

func newCheckFixture(t *testing.T, targets ...func(*cudair.Graph, *cudair.Node, *cudair.Node) *cudair.Node) checkFixture {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	y := g.AddBinary("mul", x, x)
	loss := g.AddReduce("sum", y, 0)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	targetNodes := []*cudair.Node{x}
	if len(targets) > 0 {
		targetNodes = []*cudair.Node{targets[0](g, x, loss)}
	}
	res, err := cudaautograd.Differentiate(snap, loss, targetNodes)
	if err != nil {
		t.Fatal(err)
	}
	return checkFixture{
		graph:  g,
		loss:   loss,
		target: targetNodes[0],
		result: res,
		inputs: map[string][]float32{"x": {2, -3}},
	}
}

func TestCheckGradientsValidatesArgumentsAndCoverage(t *testing.T) {
	f := newCheckFixture(t)
	tests := []struct {
		name string
		call func() error
	}{
		{"nil graph", func() error { return cudaautograd.CheckGradients(nil, f.loss, f.result, f.inputs, 1e-3, 1e-2) }},
		{"nil loss", func() error { return cudaautograd.CheckGradients(f.graph, nil, f.result, f.inputs, 1e-3, 1e-2) }},
		{"nil result", func() error { return cudaautograd.CheckGradients(f.graph, f.loss, nil, f.inputs, 1e-3, 1e-2) }},
		{"missing seed", func() error {
			broken := *f.result
			broken.LossGradInput = nil
			return cudaautograd.CheckGradients(f.graph, f.loss, &broken, f.inputs, 1e-3, 1e-2)
		}},
		{"missing target mapping", func() error {
			broken := *f.result
			broken.ForwardNodes = maps.Clone(f.result.ForwardNodes)
			delete(broken.ForwardNodes, f.target.ID)
			return cudaautograd.CheckGradients(f.graph, f.loss, &broken, f.inputs, 1e-3, 1e-2)
		}},
		{"missing target name coverage", func() error {
			broken := *f.result
			broken.TargetNames = maps.Clone(f.result.TargetNames)
			delete(broken.TargetNames, f.target.ID)
			return cudaautograd.CheckGradients(f.graph, f.loss, &broken, f.inputs, 1e-3, 1e-2)
		}},
		{"nil gradient", func() error {
			broken := *f.result
			broken.Gradients = maps.Clone(f.result.Gradients)
			broken.Gradients[f.target.ID] = nil
			return cudaautograd.CheckGradients(f.graph, f.loss, &broken, f.inputs, 1e-3, 1e-2)
		}},
		{"no comparable target", func() error {
			broken := *f.result
			broken.Gradients = map[int]*cudair.Node{f.loss.ID: f.result.Gradients[f.target.ID]}
			broken.ForwardNodes = maps.Clone(f.result.ForwardNodes)
			broken.TargetNames = map[int]string{f.loss.ID: f.loss.Name}
			broken.ForwardNodes[f.loss.ID] = f.loss
			return cudaautograd.CheckGradients(f.graph, f.loss, &broken, f.inputs, 1e-3, 1e-2)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); err == nil {
				t.Fatal("CheckGradients() succeeded with invalid input")
			}
		})
	}
}

func TestCheckGradientsUsesForwardAndBackward(t *testing.T) {
	f := newCheckFixture(t)
	before := slices.Clone(f.inputs["x"])
	if err := cudaautograd.CheckGradients(f.graph, f.loss, f.result, f.inputs, 1e-3, 1e-3); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(f.inputs["x"], before) {
		t.Fatalf("CheckGradients changed caller input: got %v, want %v", f.inputs["x"], before)
	}
}

func TestCheckGradientsRejectsForeignGraph(t *testing.T) {
	f := newCheckFixture(t)
	other := cudair.NewGraph()
	x := other.AddInput("x", []int{2}, cudair.Float32)
	loss := other.AddReduce("sum", other.AddBinary("mul", x, x), 0)
	if err := other.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	if err := cudaautograd.CheckGradients(other, loss, f.result, f.inputs, 1e-3, 1e-2); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("CheckGradients() error = %v, want ErrActivationsMismatch", err)
	}
}

func TestCheckGradientsRejectsChangedGraph(t *testing.T) {
	f := newCheckFixture(t)
	f.graph.AddInput("unused", []int{1}, cudair.Float32)
	if err := cudaautograd.CheckGradients(f.graph, f.loss, f.result, f.inputs, 1e-3, 1e-2); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("CheckGradients() error = %v, want ErrActivationsMismatch", err)
	}
}

func TestFiniteDifferenceGradPreservesInputsOnEvaluationError(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	y := g.AddInput("y", []int{2}, cudair.Float32)
	loss := g.AddReduce("sum", g.AddBinary("add", x, y), 0)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	inputs := map[string][]float32{"x": {1, 2}}
	before := slices.Clone(inputs["x"])
	if _, err := cudaautograd.FiniteDifferenceGrad(g, loss, "x", inputs, 1e-3); err == nil {
		t.Fatal("FiniteDifferenceGrad() succeeded without required input y")
	}
	if !slices.Equal(inputs["x"], before) {
		t.Fatalf("FiniteDifferenceGrad changed caller input: got %v, want %v", inputs["x"], before)
	}
}
