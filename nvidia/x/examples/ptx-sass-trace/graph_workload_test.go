package main

import (
	"context"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestGraphWorkloadValidation(t *testing.T) {
	steps := []PipelineStep{
		{Name: "step1"},
	}

	tests := []struct {
		name    string
		mode    string
		steps   []PipelineStep
		wantErr string
	}{
		{
			name:    "invalid mode",
			mode:    "invalid_mode",
			steps:   steps,
			wantErr: "unsupported graph mode",
		},
		{
			name:    "capture nil stream",
			mode:    "capture",
			steps:   steps,
			wantErr: "nil stream",
		},
		{
			name:    "capture empty steps",
			mode:    "capture",
			steps:   nil,
			wantErr: "no pipeline steps",
		},
		{
			name:    "pdl nil stream",
			mode:    "pdl",
			steps:   steps,
			wantErr: "nil stream",
		},
		{
			name:    "pdl empty steps",
			mode:    "pdl",
			steps:   nil,
			wantErr: "no pipeline steps",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := runWorkloadInGraph(context.Background(), nil, nil, tt.mode, 1, tt.steps)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestStepDescription(t *testing.T) {
	fn := driver.Function(0xcafe0000)

	t.Run("launcher configured parameters", func(t *testing.T) {
		step := PipelineStep{
			Name:     "configured_step",
			Launcher: cudalaunch.Kernel(fn).Grid(4, 2, 1).Block(128, 2, 1).SharedMem(1024),
		}
		desc, err := stepDescription(step)
		if err != nil {
			t.Fatalf("stepDescription: %v", err)
		}
		if desc.Function != fn {
			t.Errorf("Function = %v, want %v", desc.Function, fn)
		}
		if want := (cudalaunch.Dim3{X: 4, Y: 2, Z: 1}); desc.Grid != want {
			t.Errorf("Grid = %+v, want %+v", desc.Grid, want)
		}
		if want := (cudalaunch.Dim3{X: 128, Y: 2, Z: 1}); desc.Block != want {
			t.Errorf("Block = %+v, want %+v", desc.Block, want)
		}
		if desc.SharedMem != 1024 {
			t.Errorf("SharedMem = %d, want 1024", desc.SharedMem)
		}
	})

	t.Run("launcher default dimensions", func(t *testing.T) {
		desc, err := stepDescription(PipelineStep{Name: "default_step", Launcher: cudalaunch.Kernel(fn)})
		if err != nil {
			t.Fatalf("stepDescription: %v", err)
		}
		one := cudalaunch.Dim3{X: 1, Y: 1, Z: 1}
		if desc.Grid != one || desc.Block != one {
			t.Errorf("default dims = grid %+v block %+v, want %+v each", desc.Grid, desc.Block, one)
		}
		if desc.SharedMem != 0 {
			t.Errorf("default SharedMem = %d, want 0", desc.SharedMem)
		}
	})

	t.Run("step overrides", func(t *testing.T) {
		desc, err := stepDescription(PipelineStep{
			Name:           "override_step",
			Func:           fn,
			GridDimX:       16,
			BlockDimX:      64,
			SharedMemBytes: 256,
		})
		if err != nil {
			t.Fatalf("stepDescription: %v", err)
		}
		if desc.Function != fn {
			t.Errorf("Function = %v, want %v", desc.Function, fn)
		}
		if want := (cudalaunch.Dim3{X: 16, Y: 1, Z: 1}); desc.Grid != want {
			t.Errorf("Grid = %+v, want %+v", desc.Grid, want)
		}
		if want := (cudalaunch.Dim3{X: 64, Y: 1, Z: 1}); desc.Block != want {
			t.Errorf("Block = %+v, want %+v", desc.Block, want)
		}
		if desc.SharedMem != 256 {
			t.Errorf("SharedMem = %d, want 256", desc.SharedMem)
		}
	})

	// The reflection-based inspector this replaced never read the launcher's
	// attributes, so a cooperative or PDL launch became a plain graph node
	// with nothing to show it had been changed.
	t.Run("attributes reach the description", func(t *testing.T) {
		step := PipelineStep{
			Name:     "pdl_step",
			Launcher: cudalaunch.Kernel(fn).WithProgrammaticStreamSerialization(true),
		}
		desc, err := stepDescription(step)
		if err != nil {
			t.Fatalf("stepDescription: %v", err)
		}
		if len(desc.Attrs) != 1 || desc.Attrs[0].Id != driver.LaunchAttributeProgrammaticStreamSerialization {
			t.Fatalf("Attrs = %+v, want the programmatic stream serialization attribute", desc.Attrs)
		}
		// cudagraph.AddLaunch then refuses it rather than dropping it; that
		// half is checked in TestAddLaunchUnsupportedAttribute, which can
		// build a graph handle without a device.
	})

	t.Run("null function error", func(t *testing.T) {
		_, err := stepDescription(PipelineStep{Name: "null_step"})
		if err == nil {
			t.Fatal("expected error for null function, got nil")
		}
		if !strings.Contains(err.Error(), "kernel function is null") {
			t.Errorf("error = %v, want 'kernel function is null'", err)
		}
	})
}

func TestGraphWorkloadRealKernelNodes(t *testing.T) {
	dev := cudatest.Require(t)

	const testKernel = `
extern "C" __global__ void add_one(unsigned int *p, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i] += 1;
}
`
	compiled, err := nvrtcx.Compile(testKernel, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("compiled.Load: %v", err)
	}
	defer dev.WithContext(mod.Unload)

	fn, err := mod.GetFunction("add_one")
	if err != nil {
		t.Fatalf("mod.GetFunction: %v", err)
	}

	buf, err := cudamem.Alloc[uint32](dev, 64)
	if err != nil {
		t.Fatalf("cudamem.Alloc: %v", err)
	}
	defer dev.WithContext(buf.Free)

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("dev.NewStream: %v", err)
	}
	defer stream.Destroy()

	n := int32(64)
	steps := []PipelineStep{
		{
			Name:     "step1",
			Launcher: cudalaunch.Kernel(fn).Grid(1).Block(64).On(stream),
			Args:     []any{buf.Ptr(), n},
		},
		{
			Name:     "step2",
			Launcher: cudalaunch.Kernel(fn).Grid(1).Block(64).On(stream),
			Args:     []any{buf.Ptr(), n},
		},
	}

	ctx := context.Background()

	t.Run("pdl", func(t *testing.T) {
		res, intervals, err := runWorkloadInGraph(ctx, dev, stream, "pdl", 2, steps)
		if err != nil {
			t.Fatalf("runWorkloadInGraph pdl: %v", err)
		}
		if res.Mode != "pdl" {
			t.Errorf("res.Mode = %q, want pdl", res.Mode)
		}
		if res.NodeCount != 2 {
			t.Errorf("res.NodeCount = %d, want 2", res.NodeCount)
		}
		if res.PDLEdges != 1 {
			t.Errorf("res.PDLEdges = %d, want 1", res.PDLEdges)
		}
		if len(intervals) != 2 {
			t.Errorf("len(intervals) = %d, want 2", len(intervals))
		}
	})

	t.Run("capture", func(t *testing.T) {
		res, intervals, err := runWorkloadInGraph(ctx, dev, stream, "capture", 2, steps)
		if err != nil {
			t.Fatalf("runWorkloadInGraph capture: %v", err)
		}
		if res.Mode != "capture" {
			t.Errorf("res.Mode = %q, want capture", res.Mode)
		}
		if res.NodeCount != 2 {
			t.Errorf("res.NodeCount = %d, want 2", res.NodeCount)
		}
		if len(intervals) != 2 {
			t.Errorf("len(intervals) = %d, want 2", len(intervals))
		}
	})
}
