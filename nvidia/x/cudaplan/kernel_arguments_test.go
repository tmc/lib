package cudaplan

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestKernelArgumentsGeometry(t *testing.T) {
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name        string
		elements    int
		blocks      uint32
		unsupported bool
	}{
		{"one", 1, 1, false}, {"block", 256, 1, false}, {"tail", 257, 2, false},
		{"max", math.MaxUint32, 16777216, false}, {"overflow", math.MaxUint32 + 1, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{tc.elements}, cudair.Float32)
			y := g.AddUnary("neg", x)
			if err := g.SetOutputs(y); err != nil {
				t.Fatal(err)
			}
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			p := &ExecutablePlan{snap: snap}
			step := StepPlan{NodeID: y.ID, NodeName: y.Name, InputNames: []string{"x"}, OutputName: y.Name, Artifact: &cudakernel.Artifact{Schema: schema}}
			pointers := map[string]driver.DevicePtr{"x": 0x1000, y.Name: 0x2000}
			args, blocks, err := p.kernelArguments(step, pointers)
			if tc.unsupported {
				if !errors.Is(err, ErrUnsupportedBackend) {
					t.Fatalf("overflow error=%v", err)
				}
				return
			}
			if err != nil || blocks != tc.blocks || !reflect.DeepEqual(args, []any{driver.DevicePtr(0x1000), driver.DevicePtr(0x2000)}) {
				t.Fatalf("args=%v blocks=%d err=%v", args, blocks, err)
			}
			delete(pointers, "x")
			if _, _, err := p.kernelArguments(step, pointers); !errors.Is(err, ErrMissingBinding) {
				t.Fatalf("missing input accepted: %v", err)
			}
		})
	}
}

func TestReductionKernelArguments(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{8, 512}, cudair.Float32)
	y := g.AddReduce("sum", x, 1)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	p := &ExecutablePlan{snap: snap}
	step := StepPlan{NodeID: y.ID, NodeName: y.Name, InputNames: []string{"x"}, OutputName: y.Name, Artifact: &cudakernel.Artifact{Schema: schema}}
	args, blocks, err := p.kernelArguments(step, map[string]driver.DevicePtr{"x": 0x1000, y.Name: 0x2000})
	if err != nil || blocks != 1 || len(args) != 2 {
		t.Fatalf("reduction args=%v blocks=%d err=%v", args, blocks, err)
	}
	// A registered reduction must still agree with the plan's pointer ABI.
	step.Artifact.Schema.Params = step.Artifact.Schema.Params[:1]
	if _, _, err := p.kernelArguments(step, map[string]driver.DevicePtr{"x": 0x1000, y.Name: 0x2000}); err == nil {
		t.Fatal("mismatched reduction schema accepted")
	}
}
