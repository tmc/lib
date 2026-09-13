package main

import (
	"context"
	"strings"
	"testing"
	"unsafe"

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

func TestStepToKernelParams(t *testing.T) {
	fn := driver.Function(0xcafe0000)
	dptr := driver.DevicePtr(0xdeadbeef00)
	n := int32(256)
	scale := float32(0.5)

	t.Run("launcher configured parameters", func(t *testing.T) {
		launcher := cudalaunch.Kernel(fn).
			Grid(4, 2, 1).
			Block(128, 2, 1).
			SharedMem(1024)

		step := PipelineStep{
			Name:     "configured_step",
			Launcher: launcher,
			Args:     []any{dptr, n, scale},
		}

		params, cleanup, err := stepToKernelParams(step)
		if err != nil {
			t.Fatalf("stepToKernelParams: %v", err)
		}
		defer cleanup()

		if params.Func != fn {
			t.Errorf("params.Func = %v, want %v", params.Func, fn)
		}
		if params.GridDimX != 4 || params.GridDimY != 2 || params.GridDimZ != 1 {
			t.Errorf("grid dims = (%d, %d, %d), want (4, 2, 1)", params.GridDimX, params.GridDimY, params.GridDimZ)
		}
		if params.BlockDimX != 128 || params.BlockDimY != 2 || params.BlockDimZ != 1 {
			t.Errorf("block dims = (%d, %d, %d), want (128, 2, 1)", params.BlockDimX, params.BlockDimY, params.BlockDimZ)
		}
		if params.SharedMemBytes != 1024 {
			t.Errorf("params.SharedMemBytes = %d, want 1024", params.SharedMemBytes)
		}
		if params.KernelParams == 0 {
			t.Fatal("expected non-zero KernelParams pointer")
		}

		// Verify packed pointers point to values matching args
		argPtrs := unsafe.Slice((*unsafe.Pointer)(unsafe.Pointer(params.KernelParams)), 3)
		if got := *(*driver.DevicePtr)(argPtrs[0]); got != dptr {
			t.Errorf("arg 0 = 0x%x, want 0x%x", got, dptr)
		}
		if got := *(*int32)(argPtrs[1]); got != n {
			t.Errorf("arg 1 = %d, want %d", got, n)
		}
		if got := *(*float32)(argPtrs[2]); got != scale {
			t.Errorf("arg 2 = %g, want %g", got, scale)
		}
	})

	t.Run("launcher default dimensions", func(t *testing.T) {
		launcher := cudalaunch.Kernel(fn)
		step := PipelineStep{
			Name:     "default_step",
			Launcher: launcher,
		}

		params, cleanup, err := stepToKernelParams(step)
		if err != nil {
			t.Fatalf("stepToKernelParams: %v", err)
		}
		defer cleanup()

		if params.GridDimX != 1 || params.GridDimY != 1 || params.GridDimZ != 1 {
			t.Errorf("default grid dims = (%d, %d, %d), want (1, 1, 1)", params.GridDimX, params.GridDimY, params.GridDimZ)
		}
		if params.BlockDimX != 1 || params.BlockDimY != 1 || params.BlockDimZ != 1 {
			t.Errorf("default block dims = (%d, %d, %d), want (1, 1, 1)", params.BlockDimX, params.BlockDimY, params.BlockDimZ)
		}
		if params.SharedMemBytes != 0 {
			t.Errorf("default SharedMemBytes = %d, want 0", params.SharedMemBytes)
		}
		if params.KernelParams != 0 {
			t.Errorf("empty args KernelParams = 0x%x, want 0", params.KernelParams)
		}
	})

	t.Run("step overrides", func(t *testing.T) {
		step := PipelineStep{
			Name:           "override_step",
			Func:           fn,
			GridDimX:       16,
			BlockDimX:      64,
			SharedMemBytes: 256,
		}

		params, cleanup, err := stepToKernelParams(step)
		if err != nil {
			t.Fatalf("stepToKernelParams: %v", err)
		}
		defer cleanup()

		if params.Func != fn {
			t.Errorf("params.Func = %v, want %v", params.Func, fn)
		}
		if params.GridDimX != 16 || params.GridDimY != 1 || params.GridDimZ != 1 {
			t.Errorf("grid dims = (%d, %d, %d), want (16, 1, 1)", params.GridDimX, params.GridDimY, params.GridDimZ)
		}
		if params.BlockDimX != 64 || params.BlockDimY != 1 || params.BlockDimZ != 1 {
			t.Errorf("block dims = (%d, %d, %d), want (64, 1, 1)", params.BlockDimX, params.BlockDimY, params.BlockDimZ)
		}
		if params.SharedMemBytes != 256 {
			t.Errorf("params.SharedMemBytes = %d, want 256", params.SharedMemBytes)
		}
	})

	t.Run("null function error", func(t *testing.T) {
		step := PipelineStep{Name: "null_step"}
		_, _, err := stepToKernelParams(step)
		if err == nil {
			t.Fatal("expected error for null function, got nil")
		}
		if !strings.Contains(err.Error(), "kernel function is null") {
			t.Errorf("error = %v, want 'kernel function is null'", err)
		}
	})

	t.Run("invalid args pack error", func(t *testing.T) {
		step := PipelineStep{
			Name: "bad_args",
			Func: fn,
			Args: []any{nil},
		}
		_, _, err := stepToKernelParams(step)
		if err == nil {
			t.Fatal("expected error for nil arg, got nil")
		}
	})
}

func TestInspectLauncher(t *testing.T) {
	t.Run("nil launcher", func(t *testing.T) {
		_, _, _, _, err := inspectLauncher(nil)
		if err == nil {
			t.Fatal("expected error for nil launcher, got nil")
		}
	})

	t.Run("valid launcher", func(t *testing.T) {
		fn := driver.Function(0x12345678)
		l := cudalaunch.Kernel(fn).Grid(8, 4, 2).Block(32, 2, 1).SharedMem(2048)

		gotFn, gotGrid, gotBlock, gotShmem, err := inspectLauncher(l)
		if err != nil {
			t.Fatalf("inspectLauncher: %v", err)
		}
		if gotFn != fn {
			t.Errorf("gotFn = %v, want %v", gotFn, fn)
		}
		if gotGrid.X != 8 || gotGrid.Y != 4 || gotGrid.Z != 2 {
			t.Errorf("gotGrid = %+v, want (8, 4, 2)", gotGrid)
		}
		if gotBlock.X != 32 || gotBlock.Y != 2 || gotBlock.Z != 1 {
			t.Errorf("gotBlock = %+v, want (32, 2, 1)", gotBlock)
		}
		if gotShmem != 2048 {
			t.Errorf("gotShmem = %d, want 2048", gotShmem)
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
