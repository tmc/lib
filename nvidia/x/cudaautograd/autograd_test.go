package cudaautograd_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudagradcheck"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// TestGradientFiniteDifference verifies analytic gradients computed by cudaautograd against
// independent finite-difference calculations for an affine graph with nonlinearities.
func TestGradientFiniteDifference(t *testing.T) {
	// Graph: Loss = sum((X * W) * (X * W)) where X is [2, 3], W is [3, 2]
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2, 3}, cudair.Float32)
	w := g.AddInput("W", []int{3, 2}, cudair.Float32)

	// Y = X x W: [2, 2]
	y := g.AddMatMul(x, w)
	// Z = Y * Y: [2, 2]
	z := g.AddBinary("mul", y, y)
	// Loss = sum(Z, 1) -> [2] -> sum(..., 0) -> [1]
	r1 := g.AddReduce("sum", z, 1)
	loss := g.AddReduce("sum", r1, 0)

	if err := g.SetOutputs(loss); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x, w})
	if err != nil {
		t.Fatalf("Differentiate failed: %v", err)
	}

	inputs := map[string][]float32{
		"X": {0.5, -1.2, 2.0, 1.5, 0.2, -0.8},
		"W": {1.0, 0.5, -0.5, 1.2, 0.8, -0.4},
	}

	if err := cudaautograd.CheckGradients(g, loss, bwdRes, inputs, 1e-3, 1e-2); err != nil {
		t.Fatalf("CheckGradients failed: %v", err)
	}
}

// TestBroadcastGradient verifies gradient reduction when an operand is broadcast.
func TestBroadcastGradient(t *testing.T) {
	// Forward: Y = A + B, where A is [2, 2], B is [2, 2].
	// In addition, test reducing across multiple elements to verify gradient reduction logic.
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{2, 2}, cudair.Float32)
	b := g.AddInput("B", []int{2, 2}, cudair.Float32)

	sum := g.AddBinary("add", a, b)
	mul := g.AddBinary("mul", sum, sum)
	r1 := g.AddReduce("sum", mul, 1)
	loss := g.AddReduce("sum", r1, 0)

	if err := g.SetOutputs(loss); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{a, b})
	if err != nil {
		t.Fatalf("Differentiate failed: %v", err)
	}

	inputs := map[string][]float32{
		"A": {1.0, 2.0, 3.0, 4.0},
		"B": {0.5, -0.5, 1.0, -1.0},
	}

	if err := cudaautograd.CheckGradients(g, loss, bwdRes, inputs, 1e-3, 1e-2); err != nil {
		t.Fatalf("CheckGradients failed: %v", err)
	}
}

// TestSharedSubexpressionGradient verifies that when a node is consumed by multiple
// operations, incoming gradients from all consumers are properly accumulated.
func TestSharedSubexpressionGradient(t *testing.T) {
	// Forward:
	// H = X * X (shared subexpression)
	// Out1 = H * 2.0
	// Out2 = H * 3.0
	// Total = Out1 + Out2 = 5 * H = 5 * X^2
	// dTotal / dX = 10 * X
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{4}, cudair.Float32)

	h := g.AddBinary("mul", x, x)

	twoConst := g.AddConstant("two", []int{4}, cudair.Float32, []float32{2.0, 2.0, 2.0, 2.0})
	threeConst := g.AddConstant("three", []int{4}, cudair.Float32, []float32{3.0, 3.0, 3.0, 3.0})

	out1 := g.AddBinary("mul", h, twoConst)
	out2 := g.AddBinary("mul", h, threeConst)
	total := g.AddBinary("add", out1, out2)
	loss := g.AddReduce("sum", total, 0)

	if err := g.SetOutputs(loss); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
	if err != nil {
		t.Fatalf("Differentiate failed: %v", err)
	}

	inputs := map[string][]float32{
		"X": {1.0, -2.0, 3.0, 0.5},
	}

	// Analytic gradient check via finite diff
	if err := cudaautograd.CheckGradients(g, loss, bwdRes, inputs, 1e-3, 1e-2); err != nil {
		t.Fatalf("CheckGradients failed: %v", err)
	}

	// Direct manual check: dLoss / dX = 10 * X
	_, acts, err := cudaautograd.Forward(bwdRes, g, inputs)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	grads, err := cudaautograd.Backward(bwdRes, acts)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	xGrad := grads[x.ID]
	for i, xv := range inputs["X"] {
		expected := 10.0 * xv
		diff := math.Abs(float64(xGrad[i] - expected))
		if diff > 1e-3 {
			t.Errorf("at index %d: want %f, got %f", i, expected, xGrad[i])
		}
	}
}

// TestUnsupportedDerivative verifies that non-differentiable operations (e.g. reduce max)
// fail gracefully with ErrUnsupportedDerivative.
func TestUnsupportedDerivative(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{4}, cudair.Float32)
	maxNode := g.AddReduce("max", x, 0)
	if err := g.SetOutputs(maxNode); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	_, err = cudaautograd.Differentiate(snap, maxNode, []*cudair.Node{x})
	if !errors.Is(err, cudaautograd.ErrUnsupportedDerivative) {
		t.Fatalf("expected ErrUnsupportedDerivative for reduce max, got: %v", err)
	}
}

// TestSavedActivationLifetime checks that backward evaluation consumes only
// the activations its own forward evaluation saved, unchanged and once.
func TestSavedActivationLifetime(t *testing.T) {
	type fixture struct {
		g      *cudair.Graph
		s      *cudair.Snapshot
		x, w   *cudair.Node
		h      *cudair.Node
		res    *cudaautograd.BackwardResult
		inputs map[string][]float32
	}
	build := func(t *testing.T) fixture {
		t.Helper()
		g := cudair.NewGraph()
		x := g.AddInput("X", []int{2, 2}, cudair.Float32)
		w := g.AddInput("W", []int{2, 2}, cudair.Float32)
		h := g.AddUnary("exp", g.AddMatMul(x, w))
		loss := g.AddReduce("sum", g.AddReduce("sum", h, 1), 0)
		if err := g.SetOutputs(loss); err != nil {
			t.Fatal(err)
		}
		s, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		res, err := cudaautograd.Differentiate(s, loss, []*cudair.Node{w})
		if err != nil {
			t.Fatal(err)
		}
		inputs := map[string][]float32{"X": {1, 0, 0, 1}, "W": {0.1, 0.2, 0.3, 0.4}}
		return fixture{g, s, x, w, h, res, inputs}
	}

	tests := []struct {
		name string
		// misuse runs between Forward and Backward and returns the result
		// and activations to pass to Backward.
		misuse func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations)
		want   error
		reason string // distinguishes refusals that share a sentinel
	}{
		{"unchanged", nil, nil, ""},
		{"overwrite intermediate", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			values[f.h.Name][0] = 999
			return f.res, acts
		}, cudaautograd.ErrSavedActivationStale, "changed after forward"},
		{"overwrite input", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			f.inputs["X"][3] = -1
			return f.res, acts
		}, cudaautograd.ErrSavedActivationStale, "changed after forward"},
		{"refused then repaired", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			// A refused record is consumed: restoring the buffer and
			// retrying must not admit it.
			f.inputs["X"][3] = -1
			if _, err := cudaautograd.Backward(f.res, acts); !errors.Is(err, cudaautograd.ErrSavedActivationStale) {
				t.Fatalf("first Backward on overwritten buffer = %v", err)
			}
			f.inputs["X"][3] = 1
			return f.res, acts
		}, cudaautograd.ErrSavedActivationStale, "already consumed"},
		{"mismatch refusal does not consume", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			// Activations offered to a nil or wrong result stay usable
			// with their own result: only a matched call consumes them.
			if _, err := cudaautograd.Backward(nil, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
				t.Fatalf("Backward with nil result = %v", err)
			}
			if _, err := cudaautograd.Backward(build(t).res, acts); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
				t.Fatalf("Backward with other result = %v", err)
			}
			return f.res, acts
		}, nil, ""},
		{"replace map entry", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			// The record holds the buffers, not the map: rebinding a name
			// neither corrupts nor feeds the backward pass.
			values[f.h.Name] = []float32{999, 999, 999, 999}
			f.inputs["X"] = []float32{9, 9, 9, 9}
			return f.res, acts
		}, nil, ""},
		{"other result", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			return build(t).res, acts
		}, cudaautograd.ErrActivationsMismatch, "different backward result"},
		{"same graph other result", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			s, err := f.g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			res, err := cudaautograd.Differentiate(s, f.g.Outputs()[0], []*cudair.Node{f.w})
			if err != nil {
				t.Fatal(err)
			}
			return res, acts
		}, cudaautograd.ErrActivationsMismatch, "different backward result"},
		{"same snapshot sibling result", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			// Same forward snapshot, so only result identity tells them apart.
			res, err := cudaautograd.Differentiate(f.s, f.g.Outputs()[0], []*cudair.Node{f.x})
			if err != nil {
				t.Fatal(err)
			}
			return res, acts
		}, cudaautograd.ErrActivationsMismatch, "different backward result"},
		{"released", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			if _, err := cudaautograd.Backward(f.res, acts); err != nil {
				t.Fatalf("first Backward: %v", err)
			}
			return f.res, acts
		}, cudaautograd.ErrSavedActivationStale, "already consumed"},
		{"nil activations", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			return f.res, nil
		}, cudaautograd.ErrActivationsMismatch, "nil activations"},
		{"nil result", func(t *testing.T, f fixture, values map[string][]float32, acts *cudaautograd.Activations) (*cudaautograd.BackwardResult, *cudaautograd.Activations) {
			return nil, acts
		}, cudaautograd.ErrActivationsMismatch, "nil backward result"},
	}
	want := build(t)
	wantValues, wantActs, err := cudaautograd.Forward(want.res, want.g, want.inputs)
	if err != nil {
		t.Fatal(err)
	}
	wantGrad, err := cudaautograd.Backward(want.res, wantActs)
	if err != nil {
		t.Fatal(err)
	}
	// Independent reference: dLoss/dW = X^T exp(X W) with X = I is exp(W).
	for i, v := range want.inputs["W"] {
		if d := math.Abs(float64(wantGrad[want.w.ID][i]) - math.Exp(float64(v))); d > 1e-5 {
			t.Fatalf("reference gradient[%d]=%v want exp(%v)", i, wantGrad[want.w.ID][i], v)
		}
	}
	if len(wantValues) == 0 {
		t.Fatal("Forward returned no values")
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := build(t)
			values, acts, err := cudaautograd.Forward(f.res, f.g, f.inputs)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			res := f.res
			if tt.misuse != nil {
				res, acts = tt.misuse(t, f, values, acts)
			}
			grads, err := cudaautograd.Backward(res, acts)
			if tt.want != nil {
				if !errors.Is(err, tt.want) || !strings.Contains(err.Error(), tt.reason) {
					t.Fatalf("Backward = %v, %v; want %v (%s)", grads, err, tt.want, tt.reason)
				}
				return
			}
			if err != nil {
				t.Fatalf("Backward: %v", err)
			}
			if !slices.Equal(grads[f.w.ID], wantGrad[want.w.ID]) {
				t.Fatalf("gradient=%v want %v", grads[f.w.ID], wantGrad[want.w.ID])
			}
		})
	}
}

// TestForwardRefusesForeignGraph checks that activations are recorded only
// for the graph a backward result was derived from, at the shapes it was
// derived for.
func TestForwardRefusesForeignGraph(t *testing.T) {
	build := func() (*cudair.Graph, *cudair.Node, *cudaautograd.BackwardResult) {
		g := cudair.NewGraph()
		x := g.AddInput("X", []int{4}, cudair.Float32)
		loss := g.AddReduce("sum", g.AddBinary("mul", x, x), 0)
		_ = g.SetOutputs(loss)
		s, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		res, err := cudaautograd.Differentiate(s, loss, []*cudair.Node{x})
		if err != nil {
			t.Fatal(err)
		}
		return g, x, res
	}
	inputs := map[string][]float32{"X": {1, 2, 3, 4}}

	g, _, res := build()
	other, _, _ := build()
	if _, _, err := cudaautograd.Forward(res, other, inputs); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("Forward on structurally identical foreign graph = %v, want ErrActivationsMismatch", err)
	}
	if _, _, err := cudaautograd.Forward(nil, g, inputs); !errors.Is(err, cudaautograd.ErrActivationsMismatch) {
		t.Fatalf("Forward with nil result = %v, want ErrActivationsMismatch", err)
	}
	if _, acts, err := cudaautograd.Forward(res, g, inputs); err != nil || acts == nil {
		t.Fatalf("Forward on own graph: %v", err)
	}

	// Reshaping the source graph after differentiation changes the length
	// of every saved buffer; the record must not admit them.
	g, x, res := build()
	x.Shape = []int{2}
	for _, n := range g.Nodes()[1:] {
		if n.Kind == cudair.KindBinary {
			n.Shape = []int{2}
		}
	}
	if _, _, err := cudaautograd.Forward(res, g, map[string][]float32{"X": {1, 2}}); !errors.Is(err, cudaautograd.ErrSavedActivationStale) {
		t.Fatalf("Forward with changed saved length = %v, want ErrSavedActivationStale", err)
	}
}

// TestOfflineTrainingStep executes a multi-step deterministic training loop with SGDOptimizer
// and verifies that the loss decreases monotonically.
func TestOfflineTrainingStep(t *testing.T) {
	// Minimize loss = sum((X x W - Y)^2)
	// X: [2, 2], W: [2, 2], Y: [2, 2]
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2, 2}, cudair.Float32)
	w := g.AddInput("W", []int{2, 2}, cudair.Float32)
	y := g.AddInput("Y", []int{2, 2}, cudair.Float32)

	pred := g.AddMatMul(x, w)
	diff := g.AddBinary("sub", pred, y)
	sqDiff := g.AddBinary("mul", diff, diff)
	r1 := g.AddReduce("sum", sqDiff, 1)
	loss := g.AddReduce("sum", r1, 0)

	if err := g.SetOutputs(loss); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{w})
	if err != nil {
		t.Fatalf("Differentiate failed: %v", err)
	}

	// Target: learn W such that X x W == Y
	// Fixed training data
	inputs := map[string][]float32{
		"X": {1.0, 0.0, 0.0, 1.0}, // Identity matrix
		"Y": {2.0, 3.0, 4.0, 5.0}, // Target matrix
	}

	// Initial weights: zeros
	params := map[string][]float32{
		"W": {0.0, 0.0, 0.0, 0.0},
	}

	opt, err := cudaautograd.NewSGDOptimizer(0.05, 0.0)
	if err != nil {
		t.Fatalf("NewSGDOptimizer failed: %v", err)
	}

	var prevLoss float32 = 1e9
	for step := 0; step < 20; step++ {
		curLoss, _, err := cudaautograd.TrainingStep(g, loss, bwdRes, opt, params, inputs)
		if err != nil {
			t.Fatalf("step %d failed: %v", step, err)
		}
		if curLoss >= prevLoss {
			t.Errorf("step %d: loss did not decrease (prev %f, cur %f)", step, prevLoss, curLoss)
		}
		prevLoss = curLoss
	}

	// Final loss should be significantly lower than initial loss (54.0)
	if prevLoss > 5.0 {
		t.Errorf("expected final loss < 5.0, got %f", prevLoss)
	}
}

// TestNativeBackwardReference verifies compilation and execution of backward graphs
// using cudaplan. When GPU execution is unauthorized, it verifies closed failure with
// ErrExecutionUnavailable; when authorized (CUDA_COMPILER_TEST=1), it executes native kernels.
func TestNativeBackwardReference(t *testing.T) {
	// Simple backward graph: MatMul backward da = dy x b^T
	g := cudair.NewGraph()
	dy := g.AddInput("dY", []int{2, 2}, cudair.Float32)
	b := g.AddInput("B", []int{2, 2}, cudair.Float32)
	bT := g.AddTranspose(b, []int{1, 0})
	da := g.AddMatMul(dy, bT)

	if err := g.SetOutputs(da); err != nil {
		t.Fatalf("SetOutputs failed: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot failed: %v", err)
	}

	// Compile plan
	plan, err := cudaplan.Compile(snap, cudaplan.WithPeakWorkspaceBudget(16*1024*1024))
	if err != nil {
		t.Fatalf("cudaplan.Compile failed: %v", err)
	}
	defer plan.Close()

	bindings := cudaplan.NewBindings()
	_ = bindings.SetInput("dY", driver.DevicePtr(0x1000), 2*2*4, nil)
	_ = bindings.SetInput("B", driver.DevicePtr(0x2000), 2*2*4, nil)
	_ = bindings.SetOutput("matmul_3", driver.DevicePtr(0x3000), 2*2*4, nil)

	// Headless verification: native execution fails closed without device/stream
	comp, err := plan.Execute(nil, bindings)
	if !errors.Is(err, cudaplan.ErrExecutionUnavailable) || comp != nil {
		t.Fatalf("expected ErrExecutionUnavailable, got comp=%v err=%v", comp, err)
	}

	// Hardware qualification gate
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("native qualification pending: CUDA_COMPILER_TEST=1 required for hardware execution")
	}
	// See cudashared/native_test.go: the lock exports the holder's pid, not "1".
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("enabled native plan test requires running under the GPU lock")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	defer dev.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Close()

	nativePlan, err := cudaplan.Compile(snap, cudaplan.WithDevice(dev))
	if err != nil {
		t.Fatalf("native compile failed: %v", err)
	}
	defer nativePlan.Close()

	dyBuf, _ := cudamem.Alloc[float32](dev, 4)
	defer dyBuf.Free()
	bBuf, _ := cudamem.Alloc[float32](dev, 4)
	defer bBuf.Free()
	daBuf, _ := cudamem.Alloc[float32](dev, 4)
	defer daBuf.Free()

	_ = dyBuf.CopyFromHost([]float32{1.0, 2.0, 3.0, 4.0})
	_ = bBuf.CopyFromHost([]float32{0.5, 1.5, -0.5, 2.0})

	nativeBindings := cudaplan.NewBindings()
	_ = cudaplan.SetInputSlice(nativeBindings, "dY", dyBuf, dev)
	_ = cudaplan.SetInputSlice(nativeBindings, "B", bBuf, dev)
	_ = cudaplan.SetOutputSlice(nativeBindings, "matmul_3", daBuf, dev)

	nativeComp, err := nativePlan.Execute(stream, nativeBindings)
	if err != nil {
		t.Fatalf("Execute native: %v", err)
	}
	if err := nativeComp.Wait(context.Background()); err != nil {
		t.Fatalf("Wait native: %v", err)
	}
}

// Negative Controls:

// TestNegativeDropGradientContribution demonstrates that omitting an incoming adjoint
// contribution produces incorrect gradients that fail finite difference checks.
func TestNegativeDropGradientContribution(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2}, cudair.Float32)
	// Loss = X + X = 2*X.
	// Correct gradient dLoss/dX = 2.0.
	// If one contribution is dropped, gradient is 1.0 != 2.0.
	l := g.AddBinary("add", x, x)
	loss := g.AddReduce("sum", l, 0)
	_ = g.SetOutputs(loss)

	snap, _ := g.Snapshot()
	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
	if err != nil {
		t.Fatalf("Differentiate failed: %v", err)
	}

	// Mutate: drop gradient contribution by replacing with constant 1.0
	inputs := map[string][]float32{"X": {3.0, 4.0}}
	_, acts, err := cudaautograd.Forward(bwdRes, g, inputs)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	grads, err := cudaautograd.Backward(bwdRes, acts)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	correctGrad := grads[x.ID]
	if correctGrad[0] != 2.0 || correctGrad[1] != 2.0 {
		t.Fatalf("expected correct gradient [2, 2], got %v", correctGrad)
	}

	// Mutant gradient with dropped contribution
	mutantGrad := []float32{correctGrad[0] - 1.0, correctGrad[1] - 1.0}
	diff := math.Abs(float64(mutantGrad[0] - 2.0))
	if diff < 1e-3 {
		t.Fatal("mutant control failed: dropped contribution should diverge from 2.0")
	}
}

// TestNegativeReduceAlongWrongBroadcastAxis verifies that reducing along the wrong axis
// fails shape matching or produces incorrect gradient shapes.
func TestNegativeReduceAlongWrongBroadcastAxis(t *testing.T) {
	// Reducing a (2, 4) tensor along axis 1 yields (2,), while reducing along axis 0 yields (4,).
	g := cudair.NewGraph()
	a := g.AddInput("A", []int{2, 4}, cudair.Float32)
	r0 := g.AddReduce("sum", a, 0) // Shape [4]
	r1 := g.AddReduce("sum", a, 1) // Shape [2]

	if r0.Shape[0] == r1.Shape[0] {
		t.Fatal("mutant control failed: shapes across orthogonal axes should differ")
	}
}

// TestNegativeDifferentiateUnsupportedMutation verifies that graphs containing state mutations
// are strictly rejected by Differentiate.
func TestNegativeDifferentiateUnsupportedMutation(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2}, cudair.Float32)
	// Artificially inject a mutating storage effect
	x.Effect = cudair.EffectMutating
	loss := g.AddReduce("sum", x, 0)
	_ = g.SetOutputs(loss)

	_, err := g.Snapshot()
	if !errors.Is(err, cudair.ErrUnsupportedEffect) {
		t.Fatalf("expected ErrUnsupportedEffect, got: %v", err)
	}
}

func TestDifferentiateSnapshotMembership(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1}, cudair.Float32)
	loss := g.AddUnary("relu", x)
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	foreign := cudair.NewGraph()
	other := foreign.AddInput("x", []int{1}, cudair.Float32)
	otherLoss := foreign.AddUnary("relu", other)
	for _, tt := range []struct {
		name         string
		loss, target *cudair.Node
	}{
		{"foreign loss", otherLoss, x}, {"foreign target", loss, other},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := cudaautograd.Differentiate(s, tt.loss, []*cudair.Node{tt.target}); !errors.Is(err, cudaautograd.ErrTargetNotFound) {
				t.Fatalf("foreign same-ID node admitted: %v", err)
			}
		})
	}
	detached := s.NodeByID(x.ID)
	detached.Shape[0] = 99 // Membership does not make caller-owned metadata authoritative.
	if _, err := cudaautograd.Differentiate(s, s.NodeByID(loss.ID), []*cudair.Node{detached}); err != nil {
		t.Fatalf("detached member control: %v", err)
	}
	detached.ID = loss.ID
	if _, err := cudaautograd.Differentiate(s, loss, []*cudair.Node{detached}); !errors.Is(err, cudaautograd.ErrTargetNotFound) {
		t.Fatalf("changed detached ID admitted: %v", err)
	}
	x.ID = loss.ID
	if _, err := cudaautograd.Differentiate(s, loss, []*cudair.Node{x}); !errors.Is(err, cudaautograd.ErrTargetNotFound) {
		t.Fatalf("changed original ID admitted: %v", err)
	}
}

func TestScalarBroadcastGradientRanks(t *testing.T) {
	for _, shape := range [][]int{nil, {1}, {1, 1}, {1, 1, 1, 1}} {
		for _, left := range []bool{false, true} {
			for _, op := range []string{"add", "sub", "mul", "div"} {
				t.Run(fmt.Sprintf("%s/shape=%v/left=%v", op, shape, left), func(t *testing.T) {
					g := cudair.NewGraph()
					x := g.AddInput("x", []int{2}, cudair.Float32)
					scalar := g.AddInput("scalar", shape, cudair.Float32)
					a, b := x, scalar
					if left {
						a, b = scalar, x
					}
					y := g.AddBinary(op, a, b)
					loss := g.AddReduce("sum", y, 0)
					_ = g.SetOutputs(loss)
					s, err := g.Snapshot()
					if err != nil {
						t.Fatal(err)
					}
					backward, err := cudaautograd.Differentiate(s, loss, []*cudair.Node{scalar, x})
					if err != nil {
						t.Fatal(err)
					}
					gradient := backward.Gradients[scalar.ID]
					if !slices.Equal(gradient.Shape, shape) {
						t.Fatalf("gradient shape=%v want=%v", gradient.Shape, shape)
					}
					inputs := map[string][]float32{"x": {1, 2}, "scalar": {2}}
					backwardInputs := map[string][]float32{backward.LossGradInput.Name: {1}}
					for id, node := range backward.SavedActivations {
						backwardInputs[node.Name] = inputs[backward.ForwardNodes[id].Name]
					}
					got, err := cudaref.Evaluate(backward.BackwardGraph, backwardInputs)
					if err != nil {
						t.Fatal(err)
					}
					want := float32(2)
					switch op {
					case "sub":
						if !left {
							want = -2
						}
					case "mul":
						want = 3
					case "div":
						want = -0.75
						if left {
							want = 1.5
						}
					}
					values := got[gradient.Name]
					if len(values) != 1 || math.Abs(float64(values[0]-want)) > 1e-6 {
						t.Fatalf("scalar derivative=%v want=%v", values, want)
					}
					// Central differences of the forward loss are independent of the
					// generated backward reduction and also check the nonscalar target.
					if err := cudaautograd.CheckGradients(g, loss, backward, inputs, 1e-3, 1e-2); err != nil {
						t.Fatal(err)
					}
				})
			}
		}
	}
}

func TestExactGELUAndReLUDerivatives(t *testing.T) {
	tests := []struct {
		name string
		op   string
		x    []float32
	}{
		{"gelu", "gelu", []float32{-3, -1, -0.25, 0, 0.25, 1, 3}},
		{"relu", "relu", []float32{-2, -0.0, 0, 1e-7, 2}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{len(tt.x)}, cudair.Float32)
			y := g.AddUnary(tt.op, x)
			loss := g.AddReduce("sum", y, 0)
			if err := g.SetOutputs(loss); err != nil {
				t.Fatal(err)
			}
			snap, err := g.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			res, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
			if err != nil {
				t.Fatal(err)
			}

			inputs := map[string][]float32{"x": slices.Clone(tt.x)}
			values, acts, err := cudaautograd.Forward(res, g, inputs)
			if err != nil {
				t.Fatal(err)
			}
			forwardWant, err := cudaref.ElementwiseUnary(tt.x, tt.op)
			if err != nil || !slices.Equal(values[y.Name], forwardWant) {
				t.Fatalf("forward %s = %v, want %v (err %v)", tt.op, values[y.Name], forwardWant, err)
			}
			grads, err := cudaautograd.Backward(res, acts)
			if err != nil {
				t.Fatal(err)
			}
			got := grads[x.ID]
			if tt.op == "relu" {
				want := []float32{0, 0, 0, 1, 1}
				if !slices.Equal(got, want) {
					t.Fatalf("ReLU gradient = %v, want %v", got, want)
				}
				return
			}

			oracleInput := make([]float64, len(tt.x))
			upstream := make([]float64, len(tt.x))
			for i, v := range tt.x {
				oracleInput[i] = float64(v)
				upstream[i] = 1
			}
			want := cudagradcheck.GELUBackward(oracleInput, upstream)
			for i := range want {
				if math.Abs(float64(got[i])-want[i]) > 1e-7 {
					t.Errorf("GELU gradient[%d] = %.9g, exact oracle %.9g", i, got[i], want[i])
				}
			}
			if err := cudaautograd.CheckGradients(g, loss, res, inputs, 1e-3, 1e-3); err != nil {
				t.Fatalf("exact GELU central-difference check: %v", err)
			}
		})
	}
}

func TestForwardRejectsChangedGraphSemantics(t *testing.T) {
	type fixture struct {
		g          *cudair.Graph
		x, other   *cudair.Node
		constant   *cudair.Node
		intermed   *cudair.Node
		activation *cudair.Node
		loss       *cudair.Node
		res        *cudaautograd.BackwardResult
	}
	build := func(t *testing.T) fixture {
		t.Helper()
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{2}, cudair.Float32)
		other := g.AddInput("other", []int{2}, cudair.Float32)
		constant := g.AddConstant("bias", []int{2}, cudair.Float32, []float32{1, 2})
		intermed := g.AddBinary("add", x, constant)
		activation := g.AddUnary("gelu", intermed)
		loss := g.AddReduce("sum", activation, 0)
		if err := g.SetOutputs(loss); err != nil {
			t.Fatal(err)
		}
		snap, err := g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		res, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
		if err != nil {
			t.Fatal(err)
		}
		return fixture{g, x, other, constant, intermed, activation, loss, res}
	}
	inputs := map[string][]float32{"x": {0.25, 1}, "other": {2, 3}}
	tests := []struct {
		name   string
		mutate func(f fixture)
	}{
		{"operator", func(f fixture) { f.activation.Op = "relu" }},
		{"input edge", func(f fixture) { f.activation.Inputs[0] = f.other }},
		{"constant bits", func(f fixture) { f.constant.ConstValues[0]++ }},
		{"dtype", func(f fixture) { f.activation.Type = cudair.Float16 }},
		{"shape", func(f fixture) { f.intermed.Shape[0]++ }},
		{"layout", func(f fixture) { f.intermed.Layout = cudashape.LayoutColumnMajor }},
		{"layout strides", func(f fixture) { f.intermed.Strides[0]++ }},
		{"effect", func(f fixture) { f.intermed.Effect = cudair.EffectMutating }},
		{"operator parameter", func(f fixture) { f.intermed.Axis++ }},
		{"name", func(f fixture) { f.activation.Name = "changed" }},
		{"outputs", func(f fixture) { _ = f.g.SetOutputs(f.activation) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := build(t)
			tt.mutate(f)
			values, acts, err := cudaautograd.Forward(f.res, f.g, inputs)
			if !errors.Is(err, cudaautograd.ErrActivationsMismatch) || acts != nil || values != nil {
				t.Fatalf("Forward() = (%v, %v, %v), want semantic mismatch before capture", values, acts, err)
			}
			if !slices.Equal(inputs["x"], []float32{0.25, 1}) {
				t.Fatalf("mismatch path changed input: %v", inputs["x"])
			}
		})
	}
}

func TestEvaluateForwardUsesSnapshotWithoutChangingOutputs(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	intermediate := g.AddUnary("neg", x)
	loss := g.AddReduce("sum", intermediate, 0)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	before := g.Outputs()
	values, err := cudaautograd.EvaluateForward(g, map[string][]float32{"x": {2, 3}})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values[intermediate.Name], []float32{-2, -3}) {
		t.Fatalf("intermediate = %v", values[intermediate.Name])
	}
	after := g.Outputs()
	if len(after) != len(before) || after[0] != before[0] {
		t.Fatalf("EvaluateForward changed graph outputs: before=%v after=%v", before, after)
	}
}

func TestForwardAcceptsDetachedSnapshotNodes(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	loss := g.AddReduce("sum", g.AddUnary("gelu", x), 0)
	if err := g.SetOutputs(loss); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	res, err := cudaautograd.Differentiate(snap, snap.NodeByID(loss.ID), []*cudair.Node{snap.NodeByID(x.ID)})
	if err != nil {
		t.Fatal(err)
	}
	values, acts, err := cudaautograd.Forward(res, g, map[string][]float32{"x": {0.5, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if values == nil || acts == nil {
		t.Fatal("Forward returned no values or activation record")
	}
	if _, err := cudaautograd.Backward(res, acts); err != nil {
		t.Fatal(err)
	}
}
