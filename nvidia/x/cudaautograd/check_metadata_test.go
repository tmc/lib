package cudaautograd_test

import (
	"errors"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
)

type metadataFixture struct {
	graph   *cudair.Graph
	loss    *cudair.Node
	result  *cudaautograd.BackwardResult
	inputs  map[string][]float32
	x, y, z *cudair.Node
}

func newMetadataFixture(t *testing.T) metadataFixture {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	y := g.AddInput("y", []int{2}, cudair.Float32)
	z := g.AddInput("z", []int{2}, cudair.Float32)
	loss := g.AddReduce("sum", g.AddBinary("add", g.AddBinary("mul", x, y), g.AddUnary("exp", x)), 0)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	result, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x, z})
	if err != nil {
		t.Fatal(err)
	}
	if result.SavedActivations[y.ID] == nil || result.Gradients[y.ID] != nil || result.SavedActivations[z.ID] != nil || result.Gradients[z.ID] == nil {
		t.Fatal("fixture must have saved-only y and target-only z")
	}
	return metadataFixture{g, loss, result, map[string][]float32{"x": {.4, -.7}, "y": {2, 3}, "z": {1, 2}}, x, y, z}
}

func metadataNode(t *testing.T, f *metadataFixture, role string) (*cudair.Node, func(*cudair.Node)) {
	t.Helper()
	r := f.result
	forward, err := f.graph.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	backward, err := r.BackwardGraph.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	switch role {
	case "saved forward":
		return forward.NodeByID(f.y.ID), func(n *cudair.Node) { r.ForwardNodes[f.y.ID] = n }
	case "saved input":
		return backward.NodeByID(r.SavedActivations[f.y.ID].ID), func(n *cudair.Node) { r.SavedActivations[f.y.ID] = n }
	case "target":
		return forward.NodeByID(f.z.ID), func(n *cudair.Node) { r.ForwardNodes[f.z.ID] = n }
	case "gradient":
		s, err := r.BackwardGraph.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		return s.NodeByID(r.Gradients[f.z.ID].ID), func(n *cudair.Node) { r.Gradients[f.z.ID] = n }
	case "loss":
		s, err := f.graph.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		return s.NodeByID(f.loss.ID), func(n *cudair.Node) { f.loss = n }
	case "seed":
		return backward.NodeByID(r.LossGradInput.ID), func(n *cudair.Node) { r.LossGradInput = n }
	}
	t.Fatalf("unknown role %q", role)
	return nil, nil
}

func TestCheckMetadataPositive(t *testing.T) {
	f := newMetadataFixture(t)
	before := maps.Clone(f.inputs)
	for k, v := range before {
		before[k] = slices.Clone(v)
	}
	for _, role := range []string{"saved forward", "saved input", "target", "gradient", "loss", "seed"} {
		n, set := metadataNode(t, &f, role)
		copy := *n
		copy.Shape = slices.Clone(n.Shape)
		set(&copy)
	}
	if err := cudaautograd.CheckGradients(f.graph, f.loss, f.result, f.inputs, 1e-3, 1e-2); err != nil {
		t.Fatal(err)
	}
	for k, v := range before {
		if !slices.Equal(v, f.inputs[k]) {
			t.Fatalf("input %s changed", k)
		}
	}
	_, acts, err := cudaautograd.Forward(f.result, f.graph, f.inputs)
	if err != nil {
		t.Fatal(err)
	}
	gradients, err := cudaautograd.Backward(f.result, acts)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(gradients[f.z.ID], []float32{0, 0}) {
		t.Fatalf("disconnected gradient %v", gradients[f.z.ID])
	}
	if len(gradients[f.x.ID]) != 2 || gradients[f.x.ID][0] == 0 || gradients[f.x.ID][1] == 0 {
		t.Fatalf("connected gradient %v", gradients[f.x.ID])
	}
}

func TestCheckMetadataRefusesBeforeEvaluation(t *testing.T) {
	mutations := []struct {
		name   string
		change func(*cudair.Node)
	}{
		{"int32", func(n *cudair.Node) { n.Type = cudair.Int32 }},
		{"float16", func(n *cudair.Node) { n.Type = cudair.Float16 }},
		{"bfloat16", func(n *cudair.Node) { n.Type = cudair.BFloat16 }},
		{"shape", func(n *cudair.Node) { n.Shape = []int{1, n.NumElements()} }},
		{"name", func(n *cudair.Node) { n.Name += "_changed" }},
		{"kind", func(n *cudair.Node) {
			if n.Kind == cudair.KindConstant {
				n.Kind = cudair.KindInput
			} else {
				n.Kind = cudair.KindConstant
			}
		}},
	}
	for _, role := range []string{"saved forward", "saved input", "target", "gradient", "loss", "seed"} {
		for _, mutation := range mutations {
			t.Run(role+"/"+mutation.name, func(t *testing.T) {
				f := newMetadataFixture(t)
				n, set := metadataNode(t, &f, role)
				copy := *n
				copy.Shape = slices.Clone(n.Shape)
				mutation.change(&copy)
				set(&copy)
				// Invalid inputs would fail evaluation if metadata admission were bypassed.
				for _, inputs := range []map[string][]float32{f.inputs, nil} {
					err := cudaautograd.CheckGradients(f.graph, f.loss, f.result, inputs, 1e-3, 1e-2)
					if !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
						t.Fatalf("metadata admitted or evaluated: %v", err)
					}
				}
			})
		}
	}
}

func TestCheckMetadataNilSaved(t *testing.T) {
	for _, role := range []string{"saved forward", "saved input"} {
		t.Run(role, func(t *testing.T) {
			f := newMetadataFixture(t)
			_, set := metadataNode(t, &f, role)
			set(nil)
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("CheckGradients panicked: %v", p)
				}
			}()
			for _, inputs := range []map[string][]float32{f.inputs, nil} {
				err := cudaautograd.CheckGradients(f.graph, f.loss, f.result, inputs, 1e-3, 1e-2)
				if !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
					t.Fatalf("nil saved metadata admitted or evaluated: %v", err)
				}
			}
		})
	}
}

func TestCheckMetadataErrorOrder(t *testing.T) {
	f := newMetadataFixture(t)
	if err := cudaautograd.CheckGradients(nil, nil, nil, nil, 1e-3, 1e-2); err == nil || err.Error() != "cudaautograd: nil forward graph" {
		t.Fatalf("graph-first error %v", err)
	}
	if err := cudaautograd.CheckGradients(f.graph, nil, nil, nil, 1e-3, 1e-2); err == nil || err.Error() != "cudaautograd: nil loss node" {
		t.Fatalf("loss-first error %v", err)
	}
	f.result.LossGradInput = nil
	f.result.BackwardGraph = nil
	err := cudaautograd.CheckGradients(f.graph, f.loss, f.result, nil, 1e-3, 1e-2)
	if !errors.Is(err, cudaautograd.ErrActivationsMismatch) || !strings.Contains(err.Error(), "incomplete backward result") {
		t.Fatalf("incomplete precedence %v", err)
	}
}

func TestCheckMetadataRoleBinding(t *testing.T) {
	for _, which := range []string{"saved role", "gradient shape", "missing saved"} {
		t.Run(which, func(t *testing.T) {
			f := newMetadataFixture(t)
			switch which {
			case "saved role":
				f.result.SavedActivations[f.y.ID] = f.result.LossGradInput
			case "gradient shape":
				n := *f.result.Gradients[f.z.ID]
				n.Shape = []int{1, 2}
				f.result.Gradients[f.z.ID] = &n
			case "missing saved":
				delete(f.result.SavedActivations, f.y.ID)
			}
			err := cudaautograd.CheckGradients(f.graph, f.loss, f.result, nil, 1e-3, 1e-2)
			if !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
				t.Fatalf("%s admitted or evaluated: %v", which, err)
			}
		})
	}
}

func TestCheckMetadataScalar(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", nil, cudair.Float32)
	loss := g.AddBinary("mul", x, x)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	result, err := cudaautograd.Differentiate(snapshot, loss, []*cudair.Node{x})
	if err != nil {
		t.Fatal(err)
	}
	if err := cudaautograd.CheckGradients(g, loss, result, map[string][]float32{"x": {2}}, 1e-3, 1e-2); err != nil {
		t.Fatal(err)
	}
}
