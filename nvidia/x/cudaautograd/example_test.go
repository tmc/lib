package cudaautograd_test

import (
	"errors"
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// ExampleDifferentiate demonstrates differentiating an affine transformation loss with respect to weights.
func ExampleDifferentiate() {
	// 1. Build forward graph: loss = sum(X x W)
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2, 2}, cudair.Float32)
	w := g.AddInput("W", []int{2, 2}, cudair.Float32)

	prod := g.AddMatMul(x, w)
	r1 := g.AddReduce("sum", prod, 1)
	loss := g.AddReduce("sum", r1, 0)
	if err := g.SetOutputs(loss); err != nil {
		log.Fatalf("SetOutputs: %v", err)
	}

	snap, err := g.Snapshot()
	if err != nil {
		log.Fatalf("Snapshot: %v", err)
	}

	// 2. Synthesize backward graph without mutating forward graph
	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{w})
	if err != nil {
		log.Fatalf("Differentiate: %v", err)
	}

	// 3. Evaluate forward and backward passes
	inputs := map[string][]float32{
		"X": {1.0, 2.0, 3.0, 4.0},
		"W": {0.5, 0.5, 0.5, 0.5},
	}
	fwdOutputs, acts, err := cudaautograd.Forward(bwdRes, g, inputs)
	if err != nil {
		log.Fatalf("Forward: %v", err)
	}

	grads, err := cudaautograd.Backward(bwdRes, acts)
	if err != nil {
		log.Fatalf("Backward: %v", err)
	}

	// For loss = sum(X x W), dLoss/dW = X^T x ones = [[4, 4], [6, 6]]
	wGrad := grads[w.ID]
	fmt.Printf("Loss: %.1f, dLoss/dW: [%.1f, %.1f, %.1f, %.1f]\n",
		fwdOutputs[loss.Name][0], wGrad[0], wGrad[1], wGrad[2], wGrad[3])

	// Output:
	// Loss: 10.0, dLoss/dW: [4.0, 4.0, 6.0, 6.0]
}

// ExampleBackward shows that activations recorded by Forward are consumed
// once and refused if a saved buffer is written before Backward.
func ExampleBackward() {
	g := cudair.NewGraph()
	x := g.AddInput("X", []int{2}, cudair.Float32)
	loss := g.AddReduce("sum", g.AddBinary("mul", x, x), 0)
	_ = g.SetOutputs(loss)
	snap, err := g.Snapshot()
	if err != nil {
		log.Fatalf("Snapshot: %v", err)
	}
	res, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{x})
	if err != nil {
		log.Fatalf("Differentiate: %v", err)
	}

	inputs := map[string][]float32{"X": {1, 2}}
	_, acts, err := cudaautograd.Forward(res, g, inputs)
	if err != nil {
		log.Fatalf("Forward: %v", err)
	}
	grads, err := cudaautograd.Backward(res, acts)
	fmt.Println(grads[x.ID], err)
	_, err = cudaautograd.Backward(res, acts)
	fmt.Println(errors.Is(err, cudaautograd.ErrSavedActivationStale))

	_, acts, err = cudaautograd.Forward(res, g, inputs)
	if err != nil {
		log.Fatalf("Forward: %v", err)
	}
	inputs["X"][0] = 5 // overwrite a saved activation
	_, err = cudaautograd.Backward(res, acts)
	fmt.Println(errors.Is(err, cudaautograd.ErrSavedActivationStale))

	// Output:
	// [2 4] <nil>
	// true
	// true
}

// ExampleTrainingStep demonstrates running an offline training step with SGDOptimizer.
func ExampleTrainingStep() {
	// Simple loss = sum(W * W)
	g := cudair.NewGraph()
	w := g.AddInput("W", []int{2}, cudair.Float32)
	sq := g.AddBinary("mul", w, w)
	loss := g.AddReduce("sum", sq, 0)
	_ = g.SetOutputs(loss)

	snap, err := g.Snapshot()
	if err != nil {
		log.Fatalf("Snapshot: %v", err)
	}

	bwdRes, err := cudaautograd.Differentiate(snap, loss, []*cudair.Node{w})
	if err != nil {
		log.Fatalf("Differentiate: %v", err)
	}

	opt, err := cudaautograd.NewSGDOptimizer(0.1, 0.0)
	if err != nil {
		log.Fatalf("NewSGDOptimizer: %v", err)
	}

	params := map[string][]float32{"W": {2.0, 3.0}}
	inputs := map[string][]float32{}

	initialLoss, updatedParams, err := cudaautograd.TrainingStep(g, loss, bwdRes, opt, params, inputs)
	if err != nil {
		log.Fatalf("TrainingStep: %v", err)
	}

	// W_new = W - 0.1 * (2*W) = [2.0 - 0.4, 3.0 - 0.6] = [1.6, 2.4]
	fmt.Printf("Initial loss: %.1f, Updated W: [%.1f, %.1f]\n",
		initialLoss, updatedParams["W"][0], updatedParams["W"][1])

	// Output:
	// Initial loss: 13.0, Updated W: [1.6, 2.4]
}
