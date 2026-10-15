package cudaautograd_test

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

type nativeBackwardFixture struct {
	snapshot *cudair.Snapshot
	inputs   map[string][]float32
	want     map[string][]float32
}

func newNativeBackwardFixture(t *testing.T) nativeBackwardFixture {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	w := g.AddInput("w", []int{4}, cudair.Float32)
	y := g.AddBinary("mul", x, w)
	square := g.AddBinary("mul", y, y)
	loss := g.AddReduce("sum", square, 0)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	forward, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	result, err := cudaautograd.Differentiate(forward, loss, []*cudair.Node{x, w})
	if err != nil {
		t.Fatal(err)
	}
	values, err := cudaref.EvaluateSnapshotAll(forward, map[string][]float32{
		"x": {1, -2, 0.5, 3},
		"w": {2, 0.25, -4, -0.5},
	})
	if err != nil {
		t.Fatal(err)
	}
	// The forward loss is sum((x*w)^2) = 10.5. Its independently derived
	// gradients are dx = 2*x*w*w and dw = 2*w*x*x. Every value is exactly
	// representable in float32, including intermediate products and sums.
	if !slices.Equal(values[loss.ID], []float32{10.5}) {
		t.Fatalf("forward loss = %v, want [10.5]", values[loss.ID])
	}
	f := nativeBackwardFixture{
		inputs: map[string][]float32{result.LossGradInput.Name: {1}},
		want:   make(map[string][]float32),
	}
	for id, saved := range result.SavedActivations {
		data, ok := values[id]
		if !ok || len(data) != saved.NumElements() {
			t.Fatalf("saved input %q has no matching forward value", saved.Name)
		}
		f.inputs[saved.Name] = slices.Clone(data)
	}
	for id, want := range map[int][]float32{
		x.ID: {8, -0.25, 16, 1.5},
		w.ID: {4, 2, -2, -9},
	} {
		gradient := result.Gradients[id]
		if gradient == nil {
			t.Fatalf("missing differentiated target %d", id)
		}
		f.want[gradient.Name] = want
	}
	if len(result.Gradients) != len(f.want) {
		t.Fatal("unexpected differentiated target coverage")
	}
	f.snapshot, err = result.BackwardGraph.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func (f nativeBackwardFixture) checkOutputs(got map[string][]float32) error {
	if len(got) != len(f.want) {
		return fmt.Errorf("gradient outputs: got %d, want %d", len(got), len(f.want))
	}
	for name, want := range f.want {
		if !slices.Equal(got[name], want) {
			return fmt.Errorf("gradient %q = %v, want %v", name, got[name], want)
		}
	}
	return nil
}

// checkRequirements binds the same saved values and outputs in the CPU and
// native fixtures. It must reject missing or additional plan requirements.
func (f nativeBackwardFixture) checkRequirements(plan *cudaplan.ExecutablePlan) error {
	for _, role := range []struct {
		name     string
		required map[string]int
		values   map[string][]float32
	}{
		{"input", plan.RequiredInputs(), f.inputs},
		{"output", plan.RequiredOutputs(), f.want},
	} {
		if len(role.required) != len(role.values) {
			return fmt.Errorf("%s coverage: got %d, want %d", role.name, len(role.required), len(role.values))
		}
		for name, bytes := range role.required {
			if bytes != 4*len(role.values[name]) {
				return fmt.Errorf("%s %q requires %d bytes, have %d", role.name, name, bytes, 4*len(role.values[name]))
			}
		}
	}
	return nil
}

func nativeBackwardDevice(t *testing.T) (int, string) {
	t.Helper()
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAAUTOGRAD_NATIVE_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("set CUDAAUTOGRAD_NATIVE_DEVICE to an explicit ordinal")
	}
	target := os.Getenv("CUDAAUTOGRAD_NATIVE_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("set CUDAAUTOGRAD_NATIVE_TARGET to sm_100a or sm_121a")
	}
	t.Logf("lease_pid=%d lease_label=%q device=%d target=%s", owner.PID, owner.Label, ordinal, target)
	return ordinal, target
}

func TestNativeBackwardFixtureContract(t *testing.T) {
	checkUnloaded := func() {
		t.Helper()
		if driver.Loaded() || nvrtc.Loaded() || cublas.Loaded() || cublaslt.Loaded() {
			t.Fatal("host fixture loaded an NVIDIA library")
		}
	}
	checkUnloaded()
	defer checkUnloaded()
	f := newNativeBackwardFixture(t)
	got, err := cudaref.EvaluateSnapshot(f.snapshot, f.inputs)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.checkOutputs(got); err != nil {
		t.Fatal(err)
	}
	plan, err := cudaplan.Compile(f.snapshot, cudaplan.WithPeakWorkspaceBudget(16<<20))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := plan.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := f.checkRequirements(plan); err != nil {
		t.Fatal(err)
	}
	bindings := cudaplan.NewBindings()
	ptr := driver.DevicePtr(0x1000)
	for name, data := range f.inputs {
		if err := bindings.SetInput(name, ptr, 4*len(data), nil); err != nil {
			t.Fatal(err)
		}
		ptr += 0x1000
	}
	for name, data := range f.want {
		if err := bindings.SetOutput(name, ptr, 4*len(data), nil); err != nil {
			t.Fatal(err)
		}
		ptr += 0x1000
	}
	if err := bindings.Validate(nil, plan.RequiredInputs(), plan.RequiredOutputs()); err != nil {
		t.Fatal(err)
	}
	completion, err := plan.Execute(nil, bindings)
	if completion != nil || !errors.Is(err, cudaplan.ErrExecutionUnavailable) {
		t.Fatalf("headless execute = %v, %v", completion, err)
	}
	for name, data := range got {
		wrong := make(map[string][]float32, len(got))
		for key, value := range got {
			wrong[key] = slices.Clone(value)
		}
		wrong[name][0] = data[0] / 2 // Omit one shared-expression contribution.
		if err := f.checkOutputs(wrong); err == nil {
			t.Fatalf("incorrect gradient %q was accepted", name)
		}
		delete(wrong, name)
		if err := f.checkOutputs(wrong); err == nil {
			t.Fatalf("omitted gradient %q was accepted", name)
		}
		missing := cudaplan.NewBindings()
		ptr := driver.DevicePtr(0x1000)
		for input, values := range f.inputs {
			if err := missing.SetInput(input, ptr, 4*len(values), nil); err != nil {
				t.Fatal(err)
			}
			ptr += 0x1000
		}
		for output, values := range f.want {
			if output == name {
				continue
			}
			if err := missing.SetOutput(output, ptr, 4*len(values), nil); err != nil {
				t.Fatal(err)
			}
			ptr += 0x1000
		}
		if err := missing.Validate(nil, plan.RequiredInputs(), plan.RequiredOutputs()); !errors.Is(err, cudaplan.ErrMissingBinding) {
			t.Fatalf("omitted output binding %q: got %v, want ErrMissingBinding", name, err)
		}
	}
}
