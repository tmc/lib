package cudaplan

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestFiniteMegaConsumesActualIRAndSideTable(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{512}, cudair.Float32)
	a := g.AddUnary("neg", x)
	b := g.AddUnary("relu", x)
	y := g.AddBinary("add", a, b)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	p, err := Compile(snap, WithFiniteMega(true))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	for _, inst := range []string{"neg.f32 z, x", "max.NaN.f32 z, x", "add.rn.f32 z, x, y", "add.u32 i, i, 256", "bar.sync 0", "setp.ge.u32 pred, i, 512"} {
		if !strings.Contains(string(p.megaArtifact.Raw), inst) {
			t.Fatalf("missing actual operation %q", inst)
		}
	}
	tasks := p.megaLowering.Graph().Tasks()
	if len(tasks) != 3 || tasks[0].Rank != 0 || tasks[1].Rank != 0 || tasks[2].Rank != 1 {
		t.Fatalf("depth ranks=%+v", tasks)
	}
	for i, step := range p.steps {
		binding, ok := p.megaLowering.Binding(step.NodeName)
		if !ok || binding.StepIndex != i || binding.Artifact != p.megaArtifact || binding.Backend != BackendMega || binding.WorkspaceBytes != step.WorkspaceBytes || binding.WorkspaceOffset != step.WorkspaceOffset || binding.OutputName != step.OutputName || binding.IsOutput != step.IsOutput || len(binding.InputNames) != len(step.InputNames) {
			t.Fatalf("side-table %+v step %+v", binding, step)
		}
	}
}

func TestFiniteMegaReductionUsesActualAxis(t *testing.T) {
	for _, axis := range []int{0, 1, 2} {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{2, 3, 4}, cudair.Float32)
		y := g.AddReduce("sum", x, axis)
		if err := g.SetOutputs(y); err != nil {
			t.Fatal(err)
		}
		snap, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		artifact, _, err := finiteMegaArtifact(snap, "sm_80")
		if err != nil {
			t.Fatal(err)
		}
		stride := []string{"add.u64 a, a, 48", "add.u64 a, a, 16", "add.u64 a, a, 4"}[axis]
		if !strings.Contains(string(artifact.Raw), stride) || !strings.Contains(string(artifact.Raw), "add.rn.f64 total, total, widened") || !strings.Contains(string(artifact.Raw), "cvt.rn.f32.f64 z, total") {
			t.Fatal("reduction ignores shape/axis")
		}
	}
}

func TestFiniteMegaRefusesUnsupportedOperations(t *testing.T) {
	for _, op := range []string{"exp", "gelu", "sqrt"} {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{4}, cudair.Float32)
		y := g.AddUnary(op, x)
		if err := g.SetOutputs(y); err != nil {
			t.Fatal(err)
		}
		snap, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if p, err := Compile(snap, WithFiniteMega(true)); p != nil || !errors.Is(err, ErrUnsupportedBackend) {
			t.Fatalf("%s p=%v err=%v", op, p, err)
		}
	}
}
