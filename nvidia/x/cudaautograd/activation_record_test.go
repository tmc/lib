package cudaautograd_test

import (
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
)

type recordFixture struct {
	graph         *cudair.Graph
	result        *cudaautograd.BackwardResult
	saved, target *cudair.Node
	inputs        map[string][]float32
	want          float32
}

func newRecordFixture(t *testing.T, contributing bool) recordFixture {
	t.Helper()
	g := cudair.NewGraph()
	saved := g.AddInput("saved_only", []int{1}, cudair.Float32)
	target := g.AddInput("target_only", []int{1}, cudair.Float32)
	loss := g.AddUnary("log", saved)
	want := float32(0)
	if contributing {
		loss = g.AddBinary("add", loss, target)
		want = 1
	}
	if err := g.AddOutput(loss); err != nil {
		t.Fatal(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	result, err := cudaautograd.Differentiate(snapshot, loss, []*cudair.Node{target})
	if err != nil {
		t.Fatal(err)
	}
	if result.SavedActivations[saved.ID] == nil || result.SavedActivations[target.ID] != nil {
		t.Fatal("fixture must separate saved-only and target-only leaves")
	}
	return recordFixture{g, result, saved, target, map[string][]float32{"saved_only": {2}, "target_only": {7}}, want}
}

func (f recordFixture) forward(t *testing.T) *cudaautograd.Activations {
	t.Helper()
	_, acts, err := cudaautograd.Forward(f.result, f.graph, f.inputs)
	if err != nil {
		t.Fatal(err)
	}
	return acts
}

func (f recordFixture) checkGradient(t *testing.T, acts *cudaautograd.Activations) {
	t.Helper()
	grads, err := cudaautograd.Backward(f.result, acts)
	if err != nil || len(grads) != 1 || !slices.Equal(grads[f.target.ID], []float32{f.want}) {
		t.Fatalf("Backward = %v, %v; want {%d:[%g]}", grads, err, f.target.ID, f.want)
	}
}

var recordMutations = []struct {
	name   string
	change func(recordFixture)
}{
	{"nil graph", func(f recordFixture) { f.result.BackwardGraph = nil }},
	{"nil seed", func(f recordFixture) { f.result.LossGradInput = nil }},
	{"nil saved map", func(f recordFixture) { f.result.SavedActivations = nil }},
	{"nil gradient map", func(f recordFixture) { f.result.Gradients = nil }},
	{"nil forward map", func(f recordFixture) { f.result.ForwardNodes = nil }},
	{"extra forward key", func(f recordFixture) { f.result.ForwardNodes[999] = f.result.ForwardNodes[f.saved.ID] }},
	{"nil target names", func(f recordFixture) { f.result.TargetNames = nil }},
	{"nil saved node", func(f recordFixture) { f.result.SavedActivations[f.saved.ID] = nil }},
	{"nil gradient node", func(f recordFixture) { f.result.Gradients[f.target.ID] = nil }},
	{"nil saved forward", func(f recordFixture) { f.result.ForwardNodes[f.saved.ID] = nil }},
	{"saved forward type", func(f recordFixture) { f.result.ForwardNodes[f.saved.ID].Type = cudair.Float16 }},
	{"saved forward op", func(f recordFixture) { f.result.ForwardNodes[f.saved.ID].Op = "changed" }},
	{"saved input strides", func(f recordFixture) { f.result.SavedActivations[f.saved.ID].Strides = []int{99} }},
	{"target forward type", func(f recordFixture) { f.result.ForwardNodes[f.target.ID].Type = cudair.Float16 }},
	{"saved input type", func(f recordFixture) { f.result.SavedActivations[f.saved.ID].Type = cudair.Float16 }},
	{"gradient type", func(f recordFixture) { f.result.Gradients[f.target.ID].Type = cudair.Float16 }},
	{"seed shape", func(f recordFixture) { f.result.LossGradInput.Shape = []int{2} }},
	{"saved input shape", func(f recordFixture) { f.result.SavedActivations[f.saved.ID].Shape = []int{99} }},
	{"gradient shape", func(f recordFixture) { f.result.Gradients[f.target.ID].Shape = []int{99} }},
	{"target shape", func(f recordFixture) { f.result.ForwardNodes[f.target.ID].Shape = []int{99} }},
	{"saved input role", func(f recordFixture) { f.result.SavedActivations[f.saved.ID] = f.result.LossGradInput }},
	{"seed role", func(f recordFixture) { f.result.LossGradInput = f.result.SavedActivations[f.saved.ID] }},
	{"gradient role", func(f recordFixture) { f.result.Gradients[f.target.ID] = f.result.SavedActivations[f.saved.ID] }},
	{"target name", func(f recordFixture) { f.result.TargetNames[f.target.ID] = "saved_only" }},
	{"wrong target key", func(f recordFixture) {
		f.result.Gradients[999] = f.result.Gradients[f.target.ID]
		delete(f.result.Gradients, f.target.ID)
	}},
	{"wrong saved key", func(f recordFixture) {
		f.result.SavedActivations[f.target.ID] = f.result.SavedActivations[f.saved.ID]
		delete(f.result.SavedActivations, f.saved.ID)
	}},
	{"detached foreign gradient", func(f recordFixture) {
		n := f.result.Gradients[f.target.ID]
		f.result.Gradients[f.target.ID] = &cudair.Node{ID: n.ID, Name: n.Name, Kind: n.Kind, Shape: slices.Clone(n.Shape), Type: n.Type}
	}},
	{"foreign saved input", func(f recordFixture) {
		n := f.result.SavedActivations[f.saved.ID]
		f.result.SavedActivations[f.saved.ID] = cudair.NewGraph().AddInput(n.Name, n.Shape, n.Type)
	}},
	{"changed backward constant", func(f recordFixture) {
		for _, n := range f.result.BackwardGraph.Nodes() {
			if n.Kind == cudair.KindConstant {
				n.ConstValues[0] = 17
				return
			}
		}
		f.result.BackwardGraph.AddConstant("extra", []int{1}, cudair.Float32, []float32{17})
	}},
}

func TestActivationRecordRejectsMetadata(t *testing.T) {
	for _, contributing := range []bool{false, true} {
		for _, mutation := range recordMutations {
			for _, stage := range []string{"Forward", "Backward"} {
				t.Run(mutation.name+"/"+stage+map[bool]string{false: "/zero", true: "/one"}[contributing], func(t *testing.T) {
					f := newRecordFixture(t, contributing)
					var acts *cudaautograd.Activations
					if stage == "Backward" {
						acts = f.forward(t)
					}
					mutation.change(f)
					if stage == "Forward" {
						values, record, err := cudaautograd.Forward(f.result, f.graph, f.inputs)
						if !errors.Is(err, cudaautograd.ErrActivationsMismatch) || values != nil || record != nil {
							t.Fatalf("Forward = %v %v %v", values, record, err)
						}
					} else {
						grads, err := cudaautograd.Backward(f.result, acts)
						if !errors.Is(err, cudaautograd.ErrActivationsMismatch) || grads != nil {
							t.Fatalf("Backward = %v %v", grads, err)
						}
						if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrSavedActivationStale) {
							t.Fatalf("matched refusal did not consume record: %v", err)
						}
					}
				})
			}
		}
	}
}

func TestActivationRecordDetachedCopies(t *testing.T) {
	for _, contributing := range []bool{false, true} {
		f := newRecordFixture(t, contributing)
		snapshot, err := f.result.BackwardGraph.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		f.result.LossGradInput = snapshot.NodeByID(f.result.LossGradInput.ID)
		for id, n := range f.result.SavedActivations {
			f.result.SavedActivations[id] = snapshot.NodeByID(n.ID)
		}
		for id, n := range f.result.Gradients {
			f.result.Gradients[id] = snapshot.NodeByID(n.ID)
		}
		forward, err := f.graph.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		for id := range f.result.ForwardNodes {
			f.result.ForwardNodes[id] = forward.NodeByID(id)
		}
		f.checkGradient(t, f.forward(t))
	}
}

func TestActivationRecordGraphIdentity(t *testing.T) {
	f := newRecordFixture(t, true)
	other := newRecordFixture(t, true)
	acts := f.forward(t)
	// The graphs have identical semantics and IDs, but distinct owners.
	f.result.BackwardGraph = other.result.BackwardGraph
	if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("foreign graph with identical semantics accepted: %v", err)
	}
}

func TestActivationRecordBackwardSemantics(t *testing.T) {
	f := newRecordFixture(t, true)
	acts := f.forward(t)
	changed := false
	for _, n := range f.result.BackwardGraph.Nodes() {
		if n.Kind == cudair.KindBinary && n.Op == "div" {
			// This internal derivative is not itself a seed, saved input or
			// requested output. Its semantic identity still belongs to res.
			n.Op = "mul"
			changed = true
		}
	}
	if !changed {
		t.Fatal("fixture has no internal derivative operation")
	}
	if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("changed internal backward semantics accepted: %v", err)
	}
}

func TestActivationRecordTargetCoverage(t *testing.T) {
	f := newRecordFixture(t, true)
	acts := f.forward(t)
	gradient := f.result.Gradients[f.target.ID]
	delete(f.result.Gradients, f.target.ID)
	delete(f.result.TargetNames, f.target.ID)
	// A real forward node and an actual output cannot replace the requested
	// target merely by agreeing with each other and the current map counts.
	f.result.Gradients[f.saved.ID] = gradient
	f.result.TargetNames[f.saved.ID] = f.saved.Name
	if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("requested target set changed: %v", err)
	}
}

func TestActivationRecordDescriptorsFrozen(t *testing.T) {
	f := newRecordFixture(t, true)
	f.result.ForwardNodes[f.saved.ID].Type = cudair.Float16
	delete(f.result.SavedActivations, f.saved.ID)
	descriptors := f.result.SavedDescriptors()
	if len(descriptors) != 1 || descriptors[0].NodeID != f.saved.ID || descriptors[0].DataType != cudair.Float32 || descriptors[0].SizeBytes != 4 {
		t.Fatalf("descriptors = %v", descriptors)
	}
	descriptors[0].Shape[0] = 99
	if f.result.SavedDescriptors()[0].Shape[0] != 1 {
		t.Fatal("descriptor shape aliases record")
	}
}

func TestActivationRecordPrecedenceAndRetry(t *testing.T) {
	f := newRecordFixture(t, true)
	acts := f.forward(t)
	if _, err := cudaautograd.Backward(nil, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatal(err)
	}
	other := newRecordFixture(t, true)
	other.result.LossGradInput = nil
	if _, err := cudaautograd.Backward(other.result, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatal(err)
	}
	f.checkGradient(t, acts)
	f.result.LossGradInput = nil
	if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrSavedActivationStale) {
		t.Fatal("consumption must precede metadata check", err)
	}
	f = newRecordFixture(t, true)
	acts = f.forward(t)
	seed := f.result.LossGradInput
	f.result.LossGradInput = nil
	f.inputs["saved_only"][0] = 3
	if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatal("metadata check must precede buffer check", err)
	}
	f.result.LossGradInput = seed
	f.inputs["saved_only"][0] = 2
	if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrSavedActivationStale) {
		t.Fatal("restoring metadata must not revive record", err)
	}
	f.checkGradient(t, f.forward(t))
	f = newRecordFixture(t, true)
	acts = f.forward(t)
	f.inputs["saved_only"][0] = 3
	if _, err := cudaautograd.Backward(f.result, acts); !errors.Is(err, cudaautograd.ErrSavedActivationStale) {
		t.Fatal("unchanged metadata must retain saved-content refusal", err)
	}
}

func TestActivationRecordConcurrentConsumption(t *testing.T) {
	f := newRecordFixture(t, true)
	acts := f.forward(t)
	var wg sync.WaitGroup
	results := make(chan error, 2)
	for range 2 {
		wg.Go(func() { _, err := cudaautograd.Backward(f.result, acts); results <- err })
	}
	wg.Wait()
	close(results)
	success, stale := 0, 0
	for err := range results {
		if err == nil {
			success++
		} else if errors.Is(err, cudaautograd.ErrSavedActivationStale) {
			stale++
		} else {
			t.Fatal(err)
		}
	}
	if success != 1 || stale != 1 {
		t.Fatalf("success %d stale %d", success, stale)
	}
}
