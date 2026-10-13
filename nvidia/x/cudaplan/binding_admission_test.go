package cudaplan_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func admissionBindings(t *testing.T) *cudaplan.Bindings {
	t.Helper()
	b := cudaplan.NewBindings()
	if err := b.SetInput("x", 0x1000, 16, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.SetOutput("y", 0x3000, 16, nil); err != nil {
		t.Fatal(err)
	}
	return b
}

func TestBindingAdmissionNames(t *testing.T) {
	tests := []struct {
		name  string
		input bool
		key   string
	}{
		{"output shadows input", false, "x"},
		{"input shadows output", true, "y"},
		{"input shadows intermediate", true, "intermediate"},
		{"output shadows intermediate", false, "intermediate"},
		{"unknown input", true, "unknown"},
		{"unknown output", false, "unknown"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := admissionBindings(t)
			var err error
			if tt.input {
				err = b.SetInput(tt.key, 0x2000, 16, nil)
			} else {
				err = b.SetOutput(tt.key, 0x2000, 16, nil)
			}
			if err != nil {
				t.Fatal(err)
			}
			inputs := map[string]int{"x": 16}
			outputs := map[string]int{"y": 16}
			err = b.Validate(nil, inputs, outputs)
			if !errors.Is(err, cudaplan.ErrUnexpectedBinding) || !strings.Contains(err.Error(), tt.key) {
				t.Fatalf("Validate = %v, want unexpected binding %q", err, tt.key)
			}
			// Validation must not discard the refused binding or mutate requirements.
			if len(inputs) != 1 || len(outputs) != 1 {
				t.Fatal("validation changed requirements")
			}
			if err := b.Validate(nil, inputs, outputs); !errors.Is(err, cudaplan.ErrUnexpectedBinding) {
				t.Fatalf("second Validate = %v, want same refusal", err)
			}
		})
	}
}

func TestBindingAdmissionConflictingRequirements(t *testing.T) {
	b := cudaplan.NewBindings()
	err := b.Validate(nil, map[string]int{"x": 16}, map[string]int{"x": 16})
	if err == nil || !strings.Contains(err.Error(), "input and output requirements share name") {
		t.Fatalf("Validate = %v, want conflicting requirement roles", err)
	}
}

func TestBindingAdmissionValidInputsMayAlias(t *testing.T) {
	b := admissionBindings(t)
	if err := b.SetInput("z", 0x1000, 16, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.Validate(nil, map[string]int{"x": 16, "z": 16}, map[string]int{"y": 16}); err != nil {
		t.Fatalf("aliased inputs refused: %v", err)
	}
	if err := cudaplan.NewBindings().Validate(nil, nil, nil); err != nil {
		t.Fatalf("empty bindings refused: %v", err)
	}
}

func TestBindingAdmissionCompiledPlanRoles(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	intermediate := g.AddUnary("neg", x)
	y := g.AddUnary("exp", intermediate)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snapshot, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	plan, err := cudaplan.Compile(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := plan.Close(); err != nil {
			t.Error(err)
		}
	}()
	b := cudaplan.NewBindings()
	if err := b.SetInput(x.Name, 0x1000, 16, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.SetOutput(y.Name, 0x3000, 16, nil); err != nil {
		t.Fatal(err)
	}
	inputs, outputs := plan.RequiredInputs(), plan.RequiredOutputs()
	if err := b.Validate(nil, inputs, outputs); err != nil {
		t.Fatalf("compiled plan's required bindings refused: %v", err)
	}
	if err := b.SetOutput(intermediate.Name, 0x2000, 16, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.Validate(nil, inputs, outputs); !errors.Is(err, cudaplan.ErrUnexpectedBinding) {
		t.Fatalf("compiled intermediate override = %v, want unexpected binding", err)
	}
}

func TestBindingAdmissionExistingChecks(t *testing.T) {
	tests := []struct {
		name    string
		input   driver.DevicePtr
		output  driver.DevicePtr
		bytes   int
		device  *cuda.Device
		missing bool
		want    error
	}{
		{"missing input", 0x1000, 0x3000, 16, nil, true, cudaplan.ErrMissingBinding},
		{"short input", 0x1000, 0x3000, 8, nil, false, cudaplan.ErrShapeMismatch},
		{"unaligned input", 0x1001, 0x3000, 16, nil, false, cudaplan.ErrAlignmentMismatch},
		{"cross device", 0x1000, 0x3000, 16, &cuda.Device{}, false, cudaplan.ErrCrossDeviceBuffer},
		{"input output overlap", 0x1000, 0x1008, 16, nil, false, cudaplan.ErrOverlappingOutputs},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := cudaplan.NewBindings()
			if !tt.missing {
				if err := b.SetInput("x", tt.input, tt.bytes, nil); err != nil {
					t.Fatal(err)
				}
			}
			if err := b.SetOutput("y", tt.output, 16, nil); err != nil {
				t.Fatal(err)
			}
			if err := b.Validate(tt.device, map[string]int{"x": 16}, map[string]int{"y": 16}); !errors.Is(err, tt.want) {
				t.Fatalf("Validate = %v, want %v", err, tt.want)
			}
		})
	}
	// Distinct output names still cannot alias each other.
	b := admissionBindings(t)
	if err := b.SetOutput("z", 0x3008, 16, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.Validate(nil, map[string]int{"x": 16}, map[string]int{"y": 16, "z": 16}); !errors.Is(err, cudaplan.ErrOverlappingOutputs) {
		t.Fatalf("output overlap = %v", err)
	}
}
