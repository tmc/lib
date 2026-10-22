package cudaplan

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func immutablePlan(t *testing.T) (*ExecutablePlan, *KernelRegistry, *cudakernel.Artifact, *cudair.Snapshot) {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "first", schema, "sm_80")
	if err != nil {
		t.Fatal(err)
	}
	r := NewKernelRegistry()
	if err := r.Register("neg", a); err != nil {
		t.Fatal(err)
	}
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{8}, cudair.Float32)
	y := g.AddUnary("neg", x)
	z := g.AddUnary("neg", y)
	if err := g.SetOutputs(z); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	p, err := Compile(snap, WithKernelRegistry(r))
	if err != nil {
		t.Fatal(err)
	}
	return p, r, a, snap
}

func changedArtifact(t *testing.T, a *cudakernel.Artifact) *cudakernel.Artifact {
	t.Helper()
	s := a.Spec()
	s.EntryPoint = "second"
	s.Schema, _ = cudakernel.NewSchema(cudakernel.Param{Name: "n", Type: "u32"})
	b, err := cudakernel.NewArtifactFromSpec(s)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func assertCapturedPlan(t *testing.T, p *ExecutablePlan, r *KernelRegistry) {
	t.Helper()
	if got := r.Get("neg"); got == nil || got.EntryPoint() != "first" || len(got.Schema().Params) != 2 {
		t.Fatal("registry capture changed")
	}
	for _, step := range p.steps {
		if step.Artifact.EntryPoint() != "first" {
			t.Fatal("execution entry changed")
		}
		pointers := map[string]driver.DevicePtr{step.InputNames[0]: 0x1000, step.OutputName: 0x2000}
		args, blocks, err := p.kernelArguments(step, pointers)
		if err != nil || blocks != 1 || !reflect.DeepEqual(args, []any{driver.DevicePtr(0x1000), driver.DevicePtr(0x2000)}) {
			t.Fatalf("actual execution argument consumer = %v, %d, %v", args, blocks, err)
		}
	}
}

func TestArtifactCapturesThroughRegistryAndPlan(t *testing.T) {
	for _, boundary := range []string{"register", "get", "steps", "lower input", "binding", "JSON"} {
		t.Run(boundary, func(t *testing.T) {
			p, r, caller, _ := immutablePlan(t)
			other := changedArtifact(t, caller)
			low, err := LowerRegion(p.Steps(), Region{0, len(p.steps)})
			if err != nil {
				t.Fatal(err)
			}
			switch boundary {
			case "register":
				*caller = *other
			case "get":
				*r.Get("neg") = *other
			case "steps":
				*p.Steps()[0].Artifact = *other
			case "lower input":
				steps := p.Steps()
				low, err = LowerRegion(steps, Region{0, len(steps)})
				if err != nil {
					t.Fatal(err)
				}
				*steps[0].Artifact = *other
			case "binding":
				b, _ := low.Binding(low.TaskIDs()[0])
				*b.Artifact = *other
			case "JSON":
				data, err := json.Marshal(other)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(data, r.Get("neg")); err != nil {
					t.Fatal(err)
				}
			}
			assertCapturedPlan(t, p, r)
			for _, id := range low.TaskIDs() {
				b, _ := low.Binding(id)
				if b.Artifact.EntryPoint() != "first" || len(b.Artifact.Schema().Params) != 2 {
					t.Fatal("lowering capture changed")
				}
			}
		})
	}
}

func TestArtifactHandleModuleIdentity(t *testing.T) {
	p, r, a, snap := immutablePlan(t)
	first, second := r.Get("neg"), r.Get("neg")
	if first == second || *first != *second {
		t.Fatal("Get must detach wrappers and preserve sealed identity")
	}
	if *p.steps[0].Artifact != *p.steps[1].Artifact {
		t.Fatal("shared registry handle lost in plan")
	}
	keys := map[cudakernel.Artifact]bool{*first: true, *second: true}
	if len(keys) != 1 {
		t.Fatal("wrapper identity duplicated module key")
	}
	other := changedArtifact(t, a)
	if a.Digest() != other.Digest() || *a == *other {
		t.Fatal("equal bytes conflated distinct metadata")
	}
	keys[*other] = true
	if len(keys) != 2 {
		t.Fatal("module identity used raw digest alone")
	}
	owner := &cuda.Module{}
	if err := p.loadModule(first, func() (driver.Module, error) { return 7, nil }, func(driver.Module) (*cuda.Module, error) { return owner, nil }); err != nil {
		t.Fatal(err)
	}
	if p.modules[*second] != owner {
		t.Fatal("actual module owner lookup lost shared sealed identity")
	}
	otherOwner := &cuda.Module{}
	if err := p.loadModule(other, func() (driver.Module, error) { return 8, nil }, func(driver.Module) (*cuda.Module, error) { return otherOwner, nil }); err != nil {
		t.Fatal(err)
	}
	if len(p.modules) != 2 || p.modules[*first] != owner || p.modules[*other] != otherOwner {
		t.Fatal("actual module owner storage conflated distinct metadata")
	}
	if err := r.Register("neg", other); err != nil {
		t.Fatal(err)
	}
	if p.steps[0].Artifact.EntryPoint() != "first" {
		t.Fatal("replacement registration changed old plan")
	}
	next, err := Compile(snap, WithKernelRegistry(r))
	if err != nil {
		t.Fatal(err)
	}
	if next.steps[0].Artifact.EntryPoint() != "second" {
		t.Fatal("new plan did not capture replacement")
	}
}
