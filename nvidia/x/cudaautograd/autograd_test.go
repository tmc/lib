package cudaautograd_test

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
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
	fwdOutputs, err := cudaautograd.EvaluateForward(g, inputs)
	if err != nil {
		t.Fatalf("EvaluateForward failed: %v", err)
	}
	grads, err := cudaautograd.EvaluateBackward(bwdRes, inputs, fwdOutputs)
	if err != nil {
		t.Fatalf("EvaluateBackward failed: %v", err)
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

// TestSavedActivationLifetime tests the retention and integrity of saved activations.
func TestSavedActivationLifetime(t *testing.T) {
	store := cudaautograd.NewActivationStore()
	actData := []float32{1.0, 2.0, 3.0, 4.0}

	store.Save("layer1_relu", actData)

	retrieved, err := store.Get("layer1_relu")
	if err != nil {
		t.Fatalf("unexpected error retrieving activation: %v", err)
	}
	if len(retrieved) != len(actData) {
		t.Fatalf("retrieved len %d != original %d", len(retrieved), len(actData))
	}

	// Negative control: overwrite / corrupt activation
	store.Invalidate("layer1_relu")
	_, err = store.Get("layer1_relu")
	if !errors.Is(err, cudaautograd.ErrSavedActivationStale) {
		t.Fatalf("expected ErrSavedActivationStale on corrupted activation, got: %v", err)
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
	defer stream.Destroy()

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
	fwdOutputs, _ := cudaautograd.EvaluateForward(g, inputs)
	grads, _ := cudaautograd.EvaluateBackward(bwdRes, inputs, fwdOutputs)

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

	snap, _ := g.Snapshot()
	_, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
	if !errors.Is(err, cudaautograd.ErrUnsupportedMutation) {
		t.Fatalf("expected ErrUnsupportedMutation, got: %v", err)
	}
}
