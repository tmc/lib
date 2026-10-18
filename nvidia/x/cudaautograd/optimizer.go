package cudaautograd

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// SGDOptimizer implements Stochastic Gradient Descent with optional momentum.
// Weight update formula:
//
//	velocity = momentum * velocity + lr * grad
//	param = param - velocity
type SGDOptimizer struct {
	lr       float32
	momentum float32
	velocity map[string][]float32
}

// NewSGDOptimizer creates a new SGD optimizer with the given learning rate and momentum.
// The learning rate must be finite and positive; momentum must be finite and in [0, 1).
func NewSGDOptimizer(learningRate, momentum float32) (*SGDOptimizer, error) {
	if math.IsNaN(float64(learningRate)) || math.IsInf(float64(learningRate), 0) || learningRate <= 0 {
		return nil, fmt.Errorf("cudaautograd: invalid learning rate: %g", learningRate)
	}
	if math.IsNaN(float64(momentum)) || math.IsInf(float64(momentum), 0) || momentum < 0 || momentum >= 1.0 {
		return nil, fmt.Errorf("cudaautograd: invalid momentum (must be in [0, 1)): %g", momentum)
	}
	return &SGDOptimizer{
		lr:       learningRate,
		momentum: momentum,
		velocity: make(map[string][]float32),
	}, nil
}

// LearningRate returns the optimizer learning rate.
func (opt *SGDOptimizer) LearningRate() float32 {
	return opt.lr
}

// Momentum returns the optimizer momentum factor.
func (opt *SGDOptimizer) Momentum() float32 {
	return opt.momentum
}

// Step updates parameters in-place using their corresponding gradients.
// Parameters without a gradient are left unchanged, and gradients for unknown
// parameters are ignored. Step rejects a nil optimizer or map, mismatched
// slice lengths, and nonfinite gradients for present parameters before
// changing any parameter or velocity.
func (opt *SGDOptimizer) Step(params map[string][]float32, grads map[string][]float32) error {
	if opt == nil {
		return errors.New("cudaautograd: nil optimizer")
	}
	if params == nil {
		return errors.New("cudaautograd: nil params map")
	}
	if grads == nil {
		return errors.New("cudaautograd: nil grads map")
	}
	for name, paramSlice := range params {
		gradSlice, ok := grads[name]
		if !ok {
			continue
		}
		if len(paramSlice) != len(gradSlice) {
			return fmt.Errorf("cudaautograd: param %q len %d != grad len %d", name, len(paramSlice), len(gradSlice))
		}
		for i, grad := range gradSlice {
			if math.IsNaN(float64(grad)) || math.IsInf(float64(grad), 0) {
				return fmt.Errorf("cudaautograd: param %q gradient[%d] is nonfinite: %g", name, i, grad)
			}
		}
	}

	if opt.velocity == nil {
		opt.velocity = make(map[string][]float32)
	}
	for name, paramSlice := range params {
		gradSlice, ok := grads[name]
		if !ok {
			continue
		}
		vel, hasVel := opt.velocity[name]
		if !hasVel || len(vel) != len(paramSlice) {
			vel = make([]float32, len(paramSlice))
			opt.velocity[name] = vel
		}
		for i := range paramSlice {
			if opt.momentum > 0 {
				vel[i] = opt.momentum*vel[i] + opt.lr*gradSlice[i]
				paramSlice[i] -= vel[i]
			} else {
				paramSlice[i] -= opt.lr * gradSlice[i]
			}
		}
	}
	return nil
}

// TrainingStep executes a complete deterministic forward-backward-optimize step:
// 1. Evaluates forward graph to obtain current loss and intermediate activations.
// 2. Evaluates backward graph to obtain parameter gradients.
// 3. Updates parameters using optimizer.
// Returns the scalar loss before the step and the updated parameters.
func TrainingStep(
	fwdGraph *cudair.Graph,
	lossNode *cudair.Node,
	bwdRes *BackwardResult,
	opt *SGDOptimizer,
	params map[string][]float32,
	inputs map[string][]float32,
) (float32, map[string][]float32, error) {
	if opt == nil {
		return 0, nil, errors.New("cudaautograd: nil optimizer")
	}
	if bwdRes == nil {
		return 0, nil, errors.New("cudaautograd: nil backward result")
	}
	if lossNode == nil {
		return 0, nil, errors.New("cudaautograd: nil loss node")
	}

	// Merge params into inputs for forward evaluation
	fullInputs := make(map[string][]float32, len(inputs)+len(params))
	for k, v := range inputs {
		fullInputs[k] = v
	}
	for k, v := range params {
		fullInputs[k] = v
	}

	// 1. Forward pass
	fwdOutputs, acts, err := Forward(bwdRes, fwdGraph, fullInputs)
	if err != nil {
		return 0, nil, fmt.Errorf("cudaautograd: forward pass: %w", err)
	}

	lossVal, ok := fwdOutputs[lossNode.Name]
	if !ok || len(lossVal) != 1 {
		return 0, nil, fmt.Errorf("cudaautograd: missing or invalid scalar loss %q", lossNode.Name)
	}
	currentLoss := lossVal[0]

	// 2. Backward pass
	grads, err := Backward(bwdRes, acts)
	if err != nil {
		return 0, nil, fmt.Errorf("cudaautograd: backward pass: %w", err)
	}

	// Map backward gradient outputs to parameter names
	paramGrads := make(map[string][]float32, len(params))
	for id, gradSlice := range grads {
		paramName := bwdRes.TargetNames[id]
		if _, isParam := params[paramName]; isParam {
			paramGrads[paramName] = gradSlice
		}
	}

	// 3. Optimizer update
	if err := opt.Step(params, paramGrads); err != nil {
		return 0, nil, fmt.Errorf("cudaautograd: optimizer step: %w", err)
	}

	return currentLoss, params, nil
}
