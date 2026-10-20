package cudaautograd_test

import (
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func trainingLossGraph(t *testing.T) (*cudair.Graph, *cudair.Snapshot, *cudair.Node, *cudair.Node, *cudaautograd.BackwardResult) {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	loss := g.AddBinary("mul", x, x)
	other := g.AddUnary("neg", x)
	for _, n := range []*cudair.Node{loss, other} {
		if err := g.AddOutput(n); err != nil {
			t.Fatal(err)
		}
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	res, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
	if err != nil {
		t.Fatal(err)
	}
	return g, snap, loss, other, res
}

func TestTrainingLossIdentity(t *testing.T) {
	for _, detached := range []bool{false, true} {
		t.Run(fmt.Sprintf("detached=%v", detached), func(t *testing.T) {
			g, snap, loss, _, res := trainingLossGraph(t)
			if detached {
				loss = snap.NodeByID(loss.ID)
			}
			opt, err := cudaautograd.NewSGDOptimizer(.1, 0)
			if err != nil {
				t.Fatal(err)
			}
			params := map[string][]float32{"x": {2}}
			value, updated, err := cudaautograd.TrainingStep(g, loss, res, opt, params, nil)
			if err != nil || value != 4 || !slices.Equal(updated["x"], []float32{1.6}) {
				t.Fatalf("TrainingStep = %v, %v, %v; want 4, x=1.6, nil", value, updated, err)
			}
		})
	}
}

func TestTrainingLossRefusal(t *testing.T) {
	for _, name := range []string{"other objective", "foreign same name and ID", "changed ID", "changed name", "changed shape", "changed type", "changed operation", "renamed other objective"} {
		t.Run(name, func(t *testing.T) {
			g, snap, loss, other, res := trainingLossGraph(t)
			candidate := snap.NodeByID(loss.ID)
			switch name {
			case "other objective":
				candidate = other
			case "foreign same name and ID":
				_, _, candidate, _, _ = trainingLossGraph(t)
			case "changed ID":
				candidate.ID = other.ID
			case "changed name":
				candidate.Name = other.Name
			case "changed shape":
				candidate.Shape = []int{2}
			case "changed type":
				candidate.Type = cudair.Int32
			case "changed operation":
				candidate.Op = "add"
			case "renamed other objective":
				candidate = snap.NodeByID(other.ID)
				candidate.Name = loss.Name
			}
			opt, err := cudaautograd.NewSGDOptimizer(.1, .5)
			if err != nil {
				t.Fatal(err)
			}
			control, err := cudaautograd.NewSGDOptimizer(.1, .5)
			if err != nil {
				t.Fatal(err)
			}
			params := map[string][]float32{"x": {2}}
			want := map[string][]float32{"x": {2}}
			// Populate momentum so refusal must preserve existing velocity too.
			for _, step := range []struct {
				opt    *cudaautograd.SGDOptimizer
				params map[string][]float32
			}{{opt, params}, {control, want}} {
				if err := step.opt.Step(step.params, map[string][]float32{"x": {1}}); err != nil {
					t.Fatal(err)
				}
			}
			before := slices.Clone(params["x"])
			if _, _, err := cudaautograd.TrainingStep(g, candidate, res, opt, params, nil); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
				t.Fatalf("TrainingStep error = %v, want ErrActivationsMismatch", err)
			}
			if !slices.Equal(params["x"], before) {
				t.Fatalf("refusal changed parameters: %v, want %v", params["x"], before)
			}
			// Missing inputs would fail evaluation; identity refusal comes first.
			if _, _, err := cudaautograd.TrainingStep(g, candidate, res, opt, nil, nil); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
				t.Fatalf("before evaluation: error = %v, want ErrActivationsMismatch", err)
			}
			for _, step := range []struct {
				opt    *cudaautograd.SGDOptimizer
				params map[string][]float32
			}{{opt, params}, {control, want}} {
				if _, _, err := cudaautograd.TrainingStep(g, loss, res, step.opt, step.params, nil); err != nil {
					t.Fatal(err)
				}
			}
			if !slices.Equal(params["x"], want["x"]) {
				t.Fatalf("refusal changed optimizer state: next step = %v, control = %v", params["x"], want["x"])
			}
		})
	}
}

func TestTrainingLossPreservesGraphValidation(t *testing.T) {
	g, _, loss, _, res := trainingLossGraph(t)
	opt, err := cudaautograd.NewSGDOptimizer(.1, 0)
	if err != nil {
		t.Fatal(err)
	}
	g.AddInput("extra", []int{1}, cudair.Float32)
	params := map[string][]float32{"x": {2}}
	if _, _, err := cudaautograd.TrainingStep(g, loss, res, opt, params, nil); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("TrainingStep error = %v, want ErrActivationsMismatch", err)
	}
	if !slices.Equal(params["x"], []float32{2}) {
		t.Fatalf("invalid graph changed parameters: %v", params["x"])
	}
}

func ExampleTrainingStep_lossIdentity() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	loss := g.AddBinary("mul", x, x)
	other := g.AddUnary("neg", x)
	if err := g.AddOutput(loss); err != nil {
		panic(err)
	}
	if err := g.AddOutput(other); err != nil {
		panic(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		panic(err)
	}
	res, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
	if err != nil {
		panic(err)
	}
	opt, err := cudaautograd.NewSGDOptimizer(.1, 0)
	if err != nil {
		panic(err)
	}
	params := map[string][]float32{"x": {2}}
	_, _, err = cudaautograd.TrainingStep(g, other, res, opt, params, nil)
	fmt.Println(errors.Is(err, cudaautograd.ErrActivationsMismatch), params["x"])
	value, _, err := cudaautograd.TrainingStep(g, snap.NodeByID(loss.ID), res, opt, params, nil)
	fmt.Println(value, params["x"], err)
	// Output:
	// true [2]
	// 4 [1.6] <nil>
}
