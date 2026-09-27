package cudalaunch_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func mustShape(t *testing.T, dims ...int) cudashape.Shape {
	t.Helper()
	s, err := cudashape.New(dims...)
	if err != nil {
		t.Fatalf("cudashape.New(%v): unexpected error %v", dims, err)
	}
	return s
}

func mustPartition(t *testing.T, ptr driver.DevicePtr, tensorShape, partitionShape cudashape.Shape, elemBytes int, opts ...cudalayout.PartitionOption) cudalayout.Partition {
	t.Helper()
	p, err := cudalayout.NewPartition(ptr, tensorShape, partitionShape, elemBytes, opts...)
	if err != nil {
		t.Fatalf("NewPartition: unexpected error %v", err)
	}
	return p
}

func mustPartitionMut(t *testing.T, ptr driver.DevicePtr, tensorShape, partitionShape cudashape.Shape, elemBytes int, opts ...cudalayout.PartitionOption) cudalayout.PartitionMut {
	t.Helper()
	p, err := cudalayout.NewPartitionMut(ptr, tensorShape, partitionShape, elemBytes, opts...)
	if err != nil {
		t.Fatalf("NewPartitionMut: unexpected error %v", err)
	}
	return p
}

func TestLaunch_InferGrid_SinglePartition(t *testing.T) {
	t.Run("rank 1 partition", func(t *testing.T) {
		tShape := mustShape(t, 1024)
		pShape := mustShape(t, 128)
		p := mustPartition(t, driver.DevicePtr(0x1000), tShape, pShape, 4)

		grid, err := cudalaunch.InferLaunchGrid(cudashape.Grid3D{}, []cudalayout.Partition{p})
		if err != nil {
			t.Fatalf("InferLaunchGrid failed: %v", err)
		}
		want := cudashape.Grid3D{X: 8, Y: 1, Z: 1}
		if !grid.Equal(want) {
			t.Errorf("InferLaunchGrid = %v, want %v", grid, want)
		}
	})

	t.Run("rank 2 partition with matching explicit grid", func(t *testing.T) {
		tShape := mustShape(t, 512, 1024)
		pShape := mustShape(t, 32, 64)
		p := mustPartition(t, driver.DevicePtr(0x1000), tShape, pShape, 4)

		explicit := cudashape.Grid3D{X: 16, Y: 16, Z: 1}
		grid, err := cudalaunch.InferLaunchGrid(explicit, []cudalayout.Partition{p})
		if err != nil {
			t.Fatalf("InferLaunchGrid with matching explicit grid failed: %v", err)
		}
		if !grid.Equal(explicit) {
			t.Errorf("InferLaunchGrid = %v, want %v", grid, explicit)
		}
	})

	t.Run("explicit grid mismatch with partition grid", func(t *testing.T) {
		tShape := mustShape(t, 512, 1024)
		pShape := mustShape(t, 32, 64)
		p := mustPartition(t, driver.DevicePtr(0x1000), tShape, pShape, 4)

		mismatchedExplicit := cudashape.Grid3D{X: 8, Y: 16, Z: 1}
		_, err := cudalaunch.InferLaunchGrid(mismatchedExplicit, []cudalayout.Partition{p})
		if !errors.Is(err, cudalaunch.ErrGridMismatch) {
			t.Fatalf("expected ErrGridMismatch, got %v", err)
		}
	})

	t.Run("empty partitions with explicit grid", func(t *testing.T) {
		explicit := cudashape.Grid3D{X: 4, Y: 2, Z: 1}
		grid, err := cudalaunch.InferLaunchGrid(explicit, nil)
		if err != nil {
			t.Fatalf("InferLaunchGrid failed: %v", err)
		}
		if !grid.Equal(explicit) {
			t.Errorf("InferLaunchGrid = %v, want %v", grid, explicit)
		}
	})
}

func TestLaunch_InferGrid_MultiPartitionMatching(t *testing.T) {
	tShapeA := mustShape(t, 1024, 2048)
	pShapeA := mustShape(t, 64, 128)
	partA := mustPartition(t, driver.DevicePtr(0x1000), tShapeA, pShapeA, 4)

	tShapeB := mustShape(t, 1024, 2048)
	pShapeB := mustShape(t, 64, 128)
	partB := mustPartition(t, driver.DevicePtr(0x2000), tShapeB, pShapeB, 4)

	t.Run("infer across multiple matching partitions", func(t *testing.T) {
		grid, err := cudalaunch.InferLaunchGrid(cudashape.Grid3D{}, []cudalayout.Partition{partA, partB})
		if err != nil {
			t.Fatalf("InferLaunchGrid multi-partition failed: %v", err)
		}
		want := cudashape.Grid3D{X: 16, Y: 16, Z: 1}
		if !grid.Equal(want) {
			t.Errorf("InferLaunchGrid = %v, want %v", grid, want)
		}
	})

	t.Run("Config.AutoGrid integration", func(t *testing.T) {
		var cfg cudalaunch.Config
		if err := cfg.AutoGrid(partA, partB); err != nil {
			t.Fatalf("cfg.AutoGrid failed: %v", err)
		}
		want := cudashape.Grid3D{X: 16, Y: 16, Z: 1}
		if !cfg.Grid.Equal(want) {
			t.Errorf("cfg.Grid = %v, want %v", cfg.Grid, want)
		}
	})

	t.Run("NewConfig integration", func(t *testing.T) {
		cfg, err := cudalaunch.NewConfig(partA, partB)
		if err != nil {
			t.Fatalf("NewConfig failed: %v", err)
		}
		want := cudashape.Grid3D{X: 16, Y: 16, Z: 1}
		if !cfg.Grid.Equal(want) {
			t.Errorf("cfg.Grid = %v, want %v", cfg.Grid, want)
		}

		l := cudalaunch.Kernel(0x10).Config(cfg)
		desc, err := l.Describe()
		if err != nil {
			t.Fatalf("l.Describe failed: %v", err)
		}
		if desc.Grid != (driver.Dim3{X: 16, Y: 16, Z: 1}) {
			t.Errorf("desc.Grid = %v, want {16, 16, 1}", desc.Grid)
		}
	})
}

func TestLaunch_InferGrid_MultiPartitionMismatch_Error(t *testing.T) {
	// Partition 1 has grid (4, 4, 1): shape (64, 64) with tile (16, 16)
	tShape1 := mustShape(t, 64, 64)
	pShape1 := mustShape(t, 16, 16)
	part1 := mustPartition(t, driver.DevicePtr(0x1000), tShape1, pShape1, 4)

	// Partition 2 has grid (8, 2, 1): shape (64, 64) with tile (8, 32)
	tShape2 := mustShape(t, 64, 64)
	pShape2 := mustShape(t, 8, 32)
	part2 := mustPartition(t, driver.DevicePtr(0x2000), tShape2, pShape2, 4)

	// Mismatched partition grids must be rejected with ErrGridMismatch
	_, err := cudalaunch.InferLaunchGrid(cudashape.Grid3D{}, []cudalayout.Partition{part1, part2})
	if err == nil {
		t.Fatalf("expected error for mismatched partition grids (4, 4, 1) vs (8, 2, 1), got nil")
	}
	if !errors.Is(err, cudalaunch.ErrGridMismatch) {
		t.Fatalf("expected ErrGridMismatch, got %v", err)
	}

	// Also verify that Config.AutoGrid rejects mismatched partitions
	var cfg cudalaunch.Config
	if err := cfg.AutoGrid(part1, part2); !errors.Is(err, cudalaunch.ErrGridMismatch) {
		t.Fatalf("cfg.AutoGrid: expected ErrGridMismatch, got %v", err)
	}
}

func TestLaunch_BindPartitionArguments(t *testing.T) {
	tShape := mustShape(t, 64, 64)
	pShape := mustShape(t, 16, 16)
	ptrA := driver.DevicePtr(0x1000)
	ptrB := driver.DevicePtr(0x2000)

	partIn := mustPartition(t, ptrA, tShape, pShape, 4, cudalayout.WithPartitionStrides([]int{64, 1}))
	partOut := mustPartitionMut(t, ptrB, tShape, pShape, 4, cudalayout.WithPartitionStrides([]int{64, 1}))

	t.Run("BindPartitionArgs helper", func(t *testing.T) {
		args := cudalaunch.BindPartitionArgs(partIn)
		if len(args) != 4 {
			t.Fatalf("expected pointer, row stride, and tile shape, got %d args", len(args))
		}
		if p, ok := args[0].(driver.DevicePtr); !ok || p != ptrA {
			t.Errorf("args[0] = %v, want %v", args[0], ptrA)
		}
		if s, ok := args[1].(int); !ok || s != 64 {
			t.Errorf("args[1] = %v, want 64", args[1])
		}
	})

	t.Run("BindPartitionArgs32 helper", func(t *testing.T) {
		args, err := cudalaunch.BindPartitionArgs32(partIn)
		if err != nil {
			t.Fatal(err)
		}
		if len(args) != 4 {
			t.Fatalf("expected pointer, row stride, and tile shape, got %d args", len(args))
		}
		if p, ok := args[0].(driver.DevicePtr); !ok || p != ptrA {
			t.Errorf("args[0] = %v, want %v", args[0], ptrA)
		}
		if s, ok := args[1].(int32); !ok || s != 64 {
			t.Errorf("args[1] = %v, want 64 (int32)", args[1])
		}
	})

	t.Run("PackArgs auto-unpacking partitions into kernel parameters", func(t *testing.T) {
		extraScalar := int32(42)
		params, err := cudalaunch.PackArgs(partIn, partOut, extraScalar)
		if err != nil {
			t.Fatalf("PackArgs failed: %v", err)
		}
		// Each rank-2 partition binds its pointer, non-unit stride, and tile extents.
		// extraScalar is 1 param
		// Total: 9 params.
		if len(params) != 9 {
			t.Fatalf("expected 9 packed params, got %d", len(params))
		}

		// Verify param 0: ptrA
		gotPtrA := *(*driver.DevicePtr)(params[0])
		if gotPtrA != ptrA {
			t.Errorf("param 0 ptr = %v, want %v", gotPtrA, ptrA)
		}

		// Verify param 1: strideA
		gotStrideA := *(*int64)(params[1])
		if gotStrideA != 64 {
			t.Errorf("param 1 stride = %v, want 64", gotStrideA)
		}

		// Verify param 4: ptrB (after partIn's bound tile shape).
		gotPtrB := *(*driver.DevicePtr)(params[4])
		if gotPtrB != ptrB {
			t.Errorf("param 4 ptr = %v, want %v", gotPtrB, ptrB)
		}

		// Verify param 5: strideB
		gotStrideB := *(*int64)(params[5])
		if gotStrideB != 64 {
			t.Errorf("param 5 stride = %v, want 64", gotStrideB)
		}

		// Verify param 8: extraScalar
		gotScalar := *(*int32)(params[8])
		if gotScalar != 42 {
			t.Errorf("param 8 scalar = %v, want 42", gotScalar)
		}
	})
}

func TestLaunch_BindRank3PartitionAndRejectInt32Overflow(t *testing.T) {
	p := mustPartition(t, 1, mustShape(t, 16, 32, 64), mustShape(t, 2, 4, 8), 4,
		cudalayout.WithPartitionStrides([]int{2048, 64, 1}))
	args := cudalaunch.BindPartitionArgs(p)
	if len(args) != 6 {
		t.Fatalf("args = %v, want ptr, S0, S1, and three tile extents", args)
	}
	if args[1] != 2048 || args[2] != 64 {
		t.Fatalf("bound strides = %v, want [2048 64]", args[1:3])
	}
	args, err := cudalaunch.BindPartitionArgs32(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cudalaunch.BindPartitionArgs32(mustPartition(t, 1, mustShape(t, 4, 4), mustShape(t, 2, 2), 4,
		cudalayout.WithPartitionStrides([]int{1 << 32, 1}))); err == nil {
		t.Fatal("expected int32 overflow error")
	}
}

func TestLaunch_PhysicalHardwareExecution_PendingOrPass(t *testing.T) {
	dev := cudatest.Require(t)
	if dev == nil {
		t.Fatal("RequireDevice returned nil without skipping")
	}
	defer dev.Close()

	const kernelSrc = `
extern "C" __global__ void tile_scale_copy(const float* in, long long in_stride, int tile_rows, int tile_cols, float* out, long long out_stride, int out_tile_rows, int out_tile_cols, int rows, int cols) {
    int r = blockIdx.x * 16 + threadIdx.x;
    int c = blockIdx.y * 16 + threadIdx.y;
    if (r < rows && c < cols) {
        out[r * out_stride + c] = in[r * in_stride + c] * 3.0f;
    }
}
`
	cm, err := nvrtcx.Compile(kernelSrc, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}

	mod, _, err := cm.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatalf("LoadOwnedOutput failed: %v", err)
	}
	defer mod.Close()

	fn, err := mod.Function("tile_scale_copy")
	if err != nil {
		t.Fatalf("mod.Function failed: %v", err)
	}

	const rows = 48
	const cols = 32
	const count = rows * cols

	hostIn := make([]float32, count)
	for i := range hostIn {
		hostIn[i] = float32(i + 1)
	}

	dIn, err := cudamem.Alloc[float32](dev, count)
	if err != nil {
		t.Fatalf("cudamem.Alloc dIn failed: %v", err)
	}
	defer dIn.Free()

	dOut, err := cudamem.Alloc[float32](dev, count)
	if err != nil {
		t.Fatalf("cudamem.Alloc dOut failed: %v", err)
	}
	defer dOut.Free()

	if err := dIn.CopyFromHost(hostIn); err != nil {
		t.Fatalf("dIn.CopyFromHost failed: %v", err)
	}

	tShape := mustShape(t, rows, cols)
	pShape := mustShape(t, 16, 16)
	inPart := mustPartition(t, dIn.Ptr(), tShape, pShape, 4)
	outPart := mustPartitionMut(t, dOut.Ptr(), tShape, pShape, 4)

	// Launch automatically infers Grid(3, 2, 1) and binds pointers, strides, and tile shapes.
	launcher := cudalaunch.Kernel(fn.Raw()).Block(16, 16, 1)
	if err := launcher.Launch(inPart, outPart, int32(rows), int32(cols)); err != nil {
		t.Fatalf("launcher.Launch failed: %v", err)
	}

	if err := dev.WithContext(driver.CtxSynchronize); err != nil {
		t.Fatalf("CtxSynchronize failed: %v", err)
	}

	hostOut := make([]float32, count)
	if err := dOut.CopyToHost(hostOut); err != nil {
		t.Fatalf("dOut.CopyToHost failed: %v", err)
	}

	for i := range hostOut {
		want := hostIn[i] * 3.0
		if hostOut[i] != want {
			t.Fatalf("output mismatch at index %d: got %f, want %f", i, hostOut[i], want)
		}
	}
}
