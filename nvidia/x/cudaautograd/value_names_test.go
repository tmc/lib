package cudaautograd_test

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

type valueNamesFixture struct {
	graph  *cudair.Graph
	x      *cudair.Node
	loss   *cudair.Node
	first  *cudair.Node
	second *cudair.Node
	result *cudaautograd.BackwardResult
}

func newValueNamesFixture(t *testing.T, kind string) valueNamesFixture {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	loss := g.AddBinary("mul", x, x)
	name := "unrelated"
	first := loss
	switch kind {
	case "loss":
		name = loss.Name
	case "empty":
		loss.Name = ""
		name = ""
	case "intermediate":
		name = "intermediate"
		first = g.AddConstant(name, []int{1}, cudair.Float32, []float32{97})
	case "input overlap":
		loss.Name = x.Name
	case "one empty":
		loss.Name = ""
	}
	second := g.AddConstant(name, []int{1}, cudair.Float32, []float32{99})
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
	return valueNamesFixture{g, x, loss, first, second, result}
}

func (f valueNamesFixture) wantError() string {
	return fmt.Sprintf("cudaautograd: forward name %q is shared by nodes %d and %d", f.first.Name, f.first.ID, f.second.ID)
}

func TestForwardValueNamesUnique(t *testing.T) {
	for _, kind := range []string{"unique", "input overlap", "one empty"} {
		t.Run(kind, func(t *testing.T) {
			f := newValueNamesFixture(t, kind)
			inputs := map[string][]float32{"x": {2}}
			values, acts, err := cudaautograd.Forward(f.result, f.graph, inputs)
			if err != nil || acts == nil || len(values) != 2 || !slices.Equal(values[f.loss.Name], []float32{4}) || !slices.Equal(values[f.second.Name], []float32{99}) {
				t.Fatalf("Forward = %v, %v, %v", values, acts, err)
			}
			evaluated, err := cudaautograd.EvaluateForward(f.graph, inputs)
			if err != nil || len(evaluated) != 2 || !slices.Equal(evaluated[f.loss.Name], []float32{4}) {
				t.Fatalf("EvaluateForward = %v, %v", evaluated, err)
			}
			grads, err := cudaautograd.Backward(f.result, acts)
			if err != nil || !slices.Equal(grads[f.x.ID], []float32{4}) {
				t.Fatalf("Backward = %v, %v; want dx4", grads, err)
			}
			opt, err := cudaautograd.NewSGDOptimizer(.1, 0)
			if err != nil {
				t.Fatal(err)
			}
			params := map[string][]float32{"x": {2}}
			loss, updated, err := cudaautograd.TrainingStep(f.graph, f.loss, f.result, opt, params, nil)
			if err != nil || loss != 4 || !slices.Equal(updated["x"], []float32{1.6}) {
				t.Fatalf("TrainingStep = %v, %v, %v; want loss4,x1.6", loss, updated, err)
			}
		})
	}
}

func TestForwardValueNamesAmbiguous(t *testing.T) {
	for _, kind := range []string{"loss", "empty", "intermediate"} {
		t.Run(kind, func(t *testing.T) {
			f := newValueNamesFixture(t, kind)
			inputs := map[string][]float32{"x": {2}}
			values, acts, err := cudaautograd.Forward(f.result, f.graph, inputs)
			if err == nil || err.Error() != f.wantError() || values != nil || acts != nil {
				t.Fatalf("Forward = %v, %v, %v; want nil,nil,%q", values, acts, err, f.wantError())
			}
			values, err = cudaautograd.EvaluateForward(f.graph, inputs)
			if err == nil || err.Error() != f.wantError() || values != nil {
				t.Fatalf("EvaluateForward = %v, %v; want nil,%q", values, err, f.wantError())
			}
			// Name-keyed consumers refuse ambiguity; ID-keyed evaluation remains usable.
			snapshot, err := f.graph.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			all, err := cudaref.EvaluateSnapshotAll(snapshot, inputs)
			if err != nil || !slices.Equal(all[f.loss.ID], []float32{4}) || !slices.Equal(all[f.second.ID], []float32{99}) {
				t.Fatalf("EvaluateSnapshotAll = %v, %v", all, err)
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
			for _, step := range []struct {
				opt    *cudaautograd.SGDOptimizer
				params map[string][]float32
			}{{opt, params}, {control, want}} {
				if err := step.opt.Step(step.params, map[string][]float32{"x": {1}}); err != nil {
					t.Fatal(err)
				}
			}
			before := slices.Clone(params["x"])
			loss, updated, err := cudaautograd.TrainingStep(f.graph, f.loss, f.result, opt, params, nil)
			if err == nil || !strings.Contains(err.Error(), f.wantError()) || loss != 0 || updated != nil || !slices.Equal(params["x"], before) {
				t.Fatalf("TrainingStep refusal = %v, %v, %v; params=%v want %v", loss, updated, err, params["x"], before)
			}
			// A subsequent step must observe the original momentum as well as parameters.
			for _, step := range []struct {
				opt    *cudaautograd.SGDOptimizer
				params map[string][]float32
			}{{opt, params}, {control, want}} {
				if err := step.opt.Step(step.params, map[string][]float32{"x": {3}}); err != nil {
					t.Fatal(err)
				}
			}
			if !slices.Equal(params["x"], want["x"]) {
				t.Fatalf("refusal changed momentum: params=%v control=%v", params, want)
			}
		})
	}
}

func TestForwardValueNamesPrecedence(t *testing.T) {
	f := newValueNamesFixture(t, "loss")
	// A representability error is independent of missing input buffers.
	values, acts, err := cudaautograd.Forward(f.result, f.graph, nil)
	if err == nil || err.Error() != f.wantError() || values != nil || acts != nil {
		t.Fatalf("Forward missing inputs = %v, %v, %v", values, acts, err)
	}
	values, err = cudaautograd.EvaluateForward(f.graph, nil)
	if err == nil || err.Error() != f.wantError() || values != nil {
		t.Fatalf("EvaluateForward missing inputs = %v, %v", values, err)
	}
	// Existing result identity validation still precedes the projection check.
	f.result.TargetNames[f.x.ID] = "foreign"
	values, acts, err = cudaautograd.Forward(f.result, f.graph, nil)
	if !errors.Is(err, cudaautograd.ErrActivationsMismatch) || values != nil || acts != nil {
		t.Fatalf("Forward invalid metadata = %v, %v, %v", values, acts, err)
	}
}
