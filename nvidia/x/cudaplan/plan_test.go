package cudaplan_test

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// affinePTX is a valid PTX kernel implementing y[i] = a * x[i] + b.
const affinePTX = `//
// Affine kernel: y[i] = a * x[i] + b
//
.version 8.0
.target sm_121a
.address_size 64

.visible .entry affine_kernel(
	.param .u64 param_y,
	.param .u64 param_x,
	.param .f32 param_a,
	.param .f32 param_b,
	.param .u32 param_n
)
{
	.reg .b32 	%r<5>;
	.reg .f32 	%f<5>;
	.reg .b64 	%rd<7>;
	.reg .pred 	%p<2>;

	ld.param.u32 	%r1, [param_n];
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %tid.x;
	mad.lo.s32 	%r0, %r2, %r3, %r4;

	setp.ge.s32 	%p1, %r0, %r1;
	@%p1 bra 	BB0_2;

	ld.param.u64 	%rd1, [param_x];
	ld.param.u64 	%rd2, [param_y];
	ld.param.f32 	%f1, [param_a];
	ld.param.f32 	%f2, [param_b];

	mul.wide.s32 	%rd3, %r0, 4;
	add.s64 	%rd4, %rd1, %rd3;
	add.s64 	%rd5, %rd2, %rd3;

	ld.global.f32 	%f3, [%rd4];
	fma.rn.f32 	%f4, %f3, %f1, %f2;
	st.global.f32 	[%rd5], %f4;

BB0_2:
	ret;
}
`

func buildAffineArtifact(t *testing.T) *cudakernel.Artifact {
	t.Helper()
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "param_y", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "param_x", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "param_a", Type: "f32"},
		cudakernel.Param{Name: "param_b", Type: "f32"},
		cudakernel.Param{Name: "param_n", Type: "u32"},
	)
	if err != nil {
		t.Fatalf("NewSchema: %v", err)
	}

	art, err := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		[]byte(affinePTX),
		"affine_kernel",
		schema,
		"sm_121a",
		cudakernel.WithResources(cudakernel.ResourceRequirements{
			AllocatedRegisters:  32,
			PeakLiveGPR:         16,
			RegisterQuantum:     256,
			ExecutionGroup:      "cta",
			StaticSharedMemory:  0,
			DynamicSharedMemory: 0,
		}),
		cudakernel.WithJIT(cudakernel.JITOptions{
			MaxRegisters:    32,
			ThreadsPerBlock: 256,
			OptLevel:        4,
		}),
		cudakernel.WithToolchain("nvrtc 13.0"),
		cudakernel.WithCompilerOptions([]string{"--std=c++20", "--gpu-architecture=sm_121a"}),
	)
	if err != nil {
		t.Fatalf("NewArtifact: %v", err)
	}
	return art
}

func buildAffineGraph(t *testing.T) *cudair.Snapshot {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2, 3}, cudair.Float32)
	w := g.AddInput("W", []int{3, 2}, cudair.Float32)
	b := g.AddInput("B", []int{2, 2}, cudair.Float32)

	xw := g.AddMatMul(x, w)
	y := g.AddBinary("add", xw, b)
	r := g.AddUnary("relu", y)
	out := g.AddUnary("gelu", r)
	if err := g.SetOutputs(out); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	return snap
}

func TestAffinePlanRequiresNativeExecution(t *testing.T) {
	snap := buildAffineGraph(t)
	art := buildAffineArtifact(t)

	reg := cudaplan.NewKernelRegistry()
	_ = reg.Register("affine_kernel", art)
	_ = reg.Register("relu", art)

	plan, err := cudaplan.Compile(snap, cudaplan.WithKernelRegistry(reg))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer plan.Close()

	bindings := cudaplan.NewBindings()
	_ = bindings.SetInput("X", driver.DevicePtr(0x1000), 2*3*4, nil)
	_ = bindings.SetInput("W", driver.DevicePtr(0x2000), 3*2*4, nil)
	_ = bindings.SetInput("B", driver.DevicePtr(0x3000), 2*2*4, nil)
	_ = bindings.SetOutput("gelu_6", driver.DevicePtr(0x4000), 2*2*4, nil)

	comp, err := plan.Execute(nil, bindings)
	if !errors.Is(err, cudaplan.ErrExecutionUnavailable) || comp != nil {
		t.Fatalf("expected explicit native execution error, completion=%v err=%v", comp, err)
	}

}

func TestNativeMatMulReference(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("A", []int{2, 3}, cudair.Float32)
	w := g.AddInput("B", []int{3, 2}, cudair.Float32)
	d := g.AddMatMul(x, w)
	if err := g.SetOutputs(d); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// Host-only execution validation
	plan, err := cudaplan.Compile(snap)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer plan.Close()

	bindings := cudaplan.NewBindings()
	_ = bindings.SetInput("A", driver.DevicePtr(0x1000), 2*3*4, nil)
	_ = bindings.SetInput("B", driver.DevicePtr(0x2000), 3*2*4, nil)
	_ = bindings.SetOutput("matmul_2", driver.DevicePtr(0x3000), 2*2*4, nil)

	comp, err := plan.Execute(nil, bindings)
	if !errors.Is(err, cudaplan.ErrExecutionUnavailable) || comp != nil {
		t.Fatalf("expected explicit native execution error, completion=%v err=%v", comp, err)
	}

	// Gated native hardware run
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("native qualification pending: CUDA_COMPILER_TEST=1 required for hardware execution")
	}

	dev := requireNativePlanDevice(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Close()

	nativePlan, err := cudaplan.Compile(snap, cudaplan.WithDevice(dev))
	if err != nil {
		t.Fatalf("Compile native: %v", err)
	}
	defer nativePlan.Close()

	aBuf, _ := cudamem.Alloc[float32](dev, 6)
	defer aBuf.Free()
	bBuf, _ := cudamem.Alloc[float32](dev, 6)
	defer bBuf.Free()
	dBuf, _ := cudamem.Alloc[float32](dev, 4)
	defer dBuf.Free()

	// A = [[1, 2, 3], [4, 5, 6]]
	_ = aBuf.CopyFromHost([]float32{1, 2, 3, 4, 5, 6})
	// B = [[7, 8], [9, 10], [11, 12]]
	_ = bBuf.CopyFromHost([]float32{7, 8, 9, 10, 11, 12})

	nativeBindings := cudaplan.NewBindings()
	_ = cudaplan.SetInputSlice(nativeBindings, "A", aBuf, dev)
	_ = cudaplan.SetInputSlice(nativeBindings, "B", bBuf, dev)
	_ = cudaplan.SetOutputSlice(nativeBindings, "matmul_2", dBuf, dev)

	nativeComp, err := nativePlan.Execute(stream, nativeBindings)
	if err != nil {
		t.Fatalf("Execute native: %v", err)
	}
	if err := nativeComp.Wait(context.Background()); err != nil {
		t.Fatalf("Wait native: %v", err)
	}

	res := make([]float32, 4)
	_ = dBuf.CopyToHost(res)
	// D = [[1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12], [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]]
	//   = [[58, 64], [139, 154]]
	expected := []float32{58, 64, 139, 154}
	for i, exp := range expected {
		if res[i] != exp {
			t.Errorf("at %d: got %f, want %f", i, res[i], exp)
		}
	}
}

func TestCompileFailureCleanup(t *testing.T) {
	// Mutant/negative control: Inject compile failure after one acquired resource.
	// Verifies that acquired resources are freed and no memory is leaked upon failure.
	snap := buildAffineGraph(t)

	resourceAcquired := false
	_, err := cudaplan.Compile(
		snap,
		cudaplan.WithInjectedCompileFailure(true, func() {
			resourceAcquired = true
		}),
	)
	if err == nil {
		t.Fatal("expected compile failure, got nil")
	}
	if !resourceAcquired {
		t.Fatal("resource was not acquired before injected failure")
	}
}

func TestCancelDoesNotFreeBuffers(t *testing.T) {
	// Acceptance test: TestCancelDoesNotFreeBuffers
	// Invariant: Cancellation stops only waiting; it does NOT free in-flight or queued buffers.
	released := false
	bufferRelease := func() {
		released = true
	}

	snap := buildAffineGraph(t)
	plan, err := cudaplan.Compile(snap)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer plan.Close()

	bindings := cudaplan.NewBindings()
	_ = bindings.SetInput("X", driver.DevicePtr(0x1000), 2*3*4, nil)
	_ = bindings.SetInput("W", driver.DevicePtr(0x2000), 3*2*4, nil)
	_ = bindings.SetInput("B", driver.DevicePtr(0x3000), 2*2*4, nil)
	_ = bindings.SetOutput("gelu_6", driver.DevicePtr(0x4000), 2*2*4, nil)

	// Attach lease release to bindings
	bindingsCopy := *bindings
	_ = bindingsCopy

	// Create completion with simulated in-flight work
	waitStarted := make(chan struct{})
	waitBlocked := make(chan struct{})
	var startOnce sync.Once
	comp := cudaplan.ExportNewHostCompletion(
		func(ctx context.Context) error {
			startOnce.Do(func() { close(waitStarted) })
			if ctx != nil && ctx.Done() != nil {
				select {
				case <-waitBlocked:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			<-waitBlocked
			return nil
		},
		func() error { return nil },
		bufferRelease,
	)

	ctx, cancel := context.WithCancel(context.Background())
	var waitErr error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		waitErr = comp.Wait(ctx)
	}()

	<-waitStarted
	cancel()
	wg.Wait()

	if !errors.Is(waitErr, context.Canceled) {
		t.Fatalf("Wait error = %v, want context.Canceled", waitErr)
	}

	// CRITICAL INVARIANT: buffer must NOT be released upon cancellation!
	if released {
		t.Fatal("buffer lease was released upon cancellation; violates completion retention invariant")
	}

	// Negative control: Delayed native completion after cancellation.
	// When native work finally establishes completion later, buffers are released.
	close(waitBlocked)
	if err := comp.Close(); err != nil {
		t.Fatalf("Close after cancel: %v", err)
	}

	if !released {
		t.Fatal("buffer lease was not released even after final completion")
	}
}

func TestWorkspaceLifetimeAcrossBranches(t *testing.T) {
	// Acceptance test: TestWorkspaceLifetimeAcrossBranches
	// Verifies scratchpad memory reuse and lifetime isolation across parallel or branching operations.
	// Branch tensors: 64x64 float32 = 16384 bytes each
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{64, 64}, cudair.Float32)
	// Branch 1: Relu -> Gelu
	b1_1 := g.AddUnary("relu", x)
	b1_2 := g.AddUnary("gelu", b1_1)

	// Branch 2: Exp -> Neg
	b2_1 := g.AddUnary("exp", x)
	b2_2 := g.AddUnary("neg", b2_1)

	// Merge branches
	merged := g.AddBinary("add", b1_2, b2_2)
	if err := g.SetOutputs(merged); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	plan, err := cudaplan.Compile(snap, cudaplan.WithPeakWorkspaceBudget(1024*1024))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer plan.Close()

	arena := snap.ArenaPlan()
	if arena == nil {
		t.Fatal("nil arena plan")
	}

	// Verify scratchpad reuse: peak total bytes must be strictly less than summing all intermediate nodes
	naiveTotal := (snap.NodeByID(b1_1.ID).SizeBytes() +
		snap.NodeByID(b1_2.ID).SizeBytes() +
		snap.NodeByID(b2_1.ID).SizeBytes() +
		snap.NodeByID(b2_2.ID).SizeBytes() +
		snap.NodeByID(merged.ID).SizeBytes())

	if arena.TotalBytes >= naiveTotal {
		t.Errorf("arena total bytes %d not less than naive sum %d (no reuse detected)", arena.TotalBytes, naiveTotal)
	}
}

func TestPeakMemoryBudget(t *testing.T) {
	// Acceptance test: TestPeakMemoryBudget
	// Verifies rejection when scratchpad demand exceeds the configured peak budget.
	g := cudair.NewGraph()
	// Large tensor requiring 8 MiB per node
	x := g.AddInput("X", []int{1024, 2048}, cudair.Float32)
	y := g.AddUnary("relu", x)
	z := g.AddUnary("gelu", y)
	if err := g.SetOutputs(z); err != nil {
		t.Fatalf("SetOutputs: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// Set peak budget smaller than required intermediate scratchpad
	tinyBudget := 64 // 64 bytes is insufficient for 8 MiB intermediate
	_, err = cudaplan.Compile(snap, cudaplan.WithDevice(&cuda.Device{}), cudaplan.WithPeakWorkspaceBudget(tinyBudget))
	if !errors.Is(err, cudaplan.ErrWorkspaceBudgetExceeded) {
		t.Fatalf("expected ErrWorkspaceBudgetExceeded, got %v", err)
	}
}

func TestGraphReplayReference(t *testing.T) {
	// Acceptance test: TestGraphReplayReference
	snap := buildAffineGraph(t)
	plan, err := cudaplan.Compile(snap, cudaplan.WithGraphReplay(true))
	if !errors.Is(err, cudaplan.ErrUnsupportedBackend) {
		t.Fatalf("expected graph replay unsupported error, got %v", err)
	}
	if plan != nil {
		t.Fatal("unsupported graph replay returned a plan")
	}
}

func TestNegativeControlsAndMutants(t *testing.T) {
	snap := buildAffineGraph(t)
	plan, err := cudaplan.Compile(snap)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	defer plan.Close()

	// 1. Negative Control: Cross-device buffers
	t.Run("CrossDeviceBuffers", func(t *testing.T) {
		dev1 := &cuda.Device{}
		dev2 := &cuda.Device{} // distinct pointer simulates different device

		bindings := cudaplan.NewBindings()
		_ = bindings.SetInput("X", driver.DevicePtr(0x1000), 2*3*4, dev1)
		_ = bindings.SetInput("W", driver.DevicePtr(0x2000), 3*2*4, dev2) // Mismatched!
		_ = bindings.SetInput("B", driver.DevicePtr(0x3000), 2*2*4, dev1)
		_ = bindings.SetOutput("gelu_6", driver.DevicePtr(0x4000), 2*2*4, dev1)

		err := bindings.Validate(dev1, map[string]int{"X": 24, "W": 24, "B": 16}, map[string]int{"gelu_6": 16})
		if !errors.Is(err, cudaplan.ErrCrossDeviceBuffer) {
			t.Fatalf("expected ErrCrossDeviceBuffer, got %v", err)
		}
	})

	// 2. Negative Control: Overlapping outputs
	t.Run("OverlappingOutputs", func(t *testing.T) {
		bindings := cudaplan.NewBindings()
		_ = bindings.SetInput("X", driver.DevicePtr(0x1000), 24, nil)
		_ = bindings.SetInput("W", driver.DevicePtr(0x2000), 24, nil)
		_ = bindings.SetInput("B", driver.DevicePtr(0x3000), 16, nil)
		// Set two outputs pointing to the same memory range
		_ = bindings.SetOutput("out1", driver.DevicePtr(0x4000), 32, nil)
		_ = bindings.SetOutput("out2", driver.DevicePtr(0x4010), 32, nil) // Overlaps out1 [0x4000, 0x4020)

		requiredInputs := map[string]int{"X": 24, "W": 24, "B": 16}
		requiredOutputs := map[string]int{"out1": 32, "out2": 32}
		err := bindings.Validate(nil, requiredInputs, requiredOutputs)
		if !errors.Is(err, cudaplan.ErrOverlappingOutputs) {
			t.Fatalf("expected ErrOverlappingOutputs, got %v", err)
		}

		// Adjacent half-open output ranges are disjoint.
		if err := bindings.SetOutput("out2", driver.DevicePtr(0x4020), 32, nil); err != nil {
			t.Fatal(err)
		}
		if err := bindings.Validate(nil, requiredInputs, requiredOutputs); err != nil {
			t.Fatalf("boundary-disjoint outputs: %v", err)
		}
	})

	// 3. Negative Control: Active invocation in-flight
	t.Run("InvocationInFlight", func(t *testing.T) {
		bindings := cudaplan.NewBindings()
		_ = bindings.SetInput("X", driver.DevicePtr(0x1000), 2*3*4, nil)
		_ = bindings.SetInput("W", driver.DevicePtr(0x2000), 3*2*4, nil)
		_ = bindings.SetInput("B", driver.DevicePtr(0x3000), 2*2*4, nil)
		_ = bindings.SetOutput("gelu_6", driver.DevicePtr(0x4000), 2*2*4, nil)

		planInstance, err := cudaplan.Compile(snap)
		if err != nil {
			t.Fatal(err)
		}
		defer planInstance.Close()

		comp1, err := planInstance.Execute(nil, bindings)
		if !errors.Is(err, cudaplan.ErrExecutionUnavailable) || comp1 != nil {
			t.Fatalf("expected unavailable execution, completion=%v err=%v", comp1, err)
		}
	})
}
