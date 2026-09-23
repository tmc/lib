package cudaautograd

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

// FiniteDifferenceGrad computes numeric gradients of loss with respect to target inputs using central differences.
// eps is the finite perturbation step size (e.g. 1e-3 or 1e-4).
// forwardEval evaluates the forward loss scalar given an input map.
func FiniteDifferenceGrad(
	fwdGraph *cudair.Graph,
	lossNode *cudair.Node,
	targetInputName string,
	inputs map[string][]float32,
	eps float32,
) ([]float32, error) {
	if fwdGraph == nil {
		return nil, errors.New("cudaautograd: nil forward graph")
	}
	if lossNode == nil {
		return nil, errors.New("cudaautograd: nil loss node")
	}
	buf, ok := inputs[targetInputName]
	if !ok {
		return nil, fmt.Errorf("cudaautograd: target input %q not in inputs map", targetInputName)
	}
	if eps <= 0 {
		eps = 1e-3
	}

	grad := make([]float32, len(buf))

	// Shallow copy map of inputs
	clonedInputs := make(map[string][]float32, len(inputs))
	for k, v := range inputs {
		cp := make([]float32, len(v))
		copy(cp, v)
		clonedInputs[k] = cp
	}

	targetBuf := clonedInputs[targetInputName]

	for i := 0; i < len(buf); i++ {
		orig := targetBuf[i]

		// f(x + eps)
		targetBuf[i] = orig + eps
		outPlus, err := cudaref.Evaluate(fwdGraph, clonedInputs)
		if err != nil {
			return nil, fmt.Errorf("finite diff f(x+eps) evaluation failed: %w", err)
		}
		lossPlus := outPlus[lossNode.Name]
		if len(lossPlus) != 1 {
			return nil, fmt.Errorf("loss node %s output length %d != 1", lossNode.Name, len(lossPlus))
		}

		// f(x - eps)
		targetBuf[i] = orig - eps
		outMinus, err := cudaref.Evaluate(fwdGraph, clonedInputs)
		if err != nil {
			return nil, fmt.Errorf("finite diff f(x-eps) evaluation failed: %w", err)
		}
		lossMinus := outMinus[lossNode.Name]
		if len(lossMinus) != 1 {
			return nil, fmt.Errorf("loss node %s output length %d != 1", lossNode.Name, len(lossMinus))
		}

		// Central difference: (f(x+eps) - f(x-eps)) / (2 * eps)
		grad[i] = (lossPlus[0] - lossMinus[0]) / (2.0 * eps)

		// Restore original value
		targetBuf[i] = orig
	}

	return grad, nil
}

// CheckGradients compares analytic gradients computed by BackwardResult against finite difference gradients.
// Returns nil if maximum relative difference is within tolerance, or a descriptive error detailing mismatch.
func CheckGradients(
	fwdGraph *cudair.Graph,
	lossNode *cudair.Node,
	bwdRes *BackwardResult,
	inputs map[string][]float32,
	eps float32,
	tolerance float32,
) error {
	// First evaluate forward graph with saved activations
	fwdOutputs, err := EvaluateForward(fwdGraph, inputs)
	if err != nil {
		return fmt.Errorf("cudaautograd: forward evaluation failed: %w", err)
	}

	// Prepare inputs for backward graph
	bwdInputs := make(map[string][]float32)
	// Seed gradient dLoss/dLoss = 1.0
	bwdInputs[bwdRes.LossGradInput.Name] = []float32{1.0}

	// Supply saved activations
	for id, bwdInp := range bwdRes.SavedActivations {
		fwdNode := bwdRes.ForwardNodes[id]
		if fwdNode == nil {
			return fmt.Errorf("missing forward node for id %d", id)
		}
		var actData []float32
		if fwdNode.Kind == cudair.KindInput {
			actData = inputs[fwdNode.Name]
		} else if val, ok := fwdOutputs[fwdNode.Name]; ok {
			actData = val
		} else {
			// Intermediate activation: evaluate up to this node
			subGraphOutputs, err := cudaref.Evaluate(fwdGraph, inputs)
			if err != nil {
				return err
			}
			actData = subGraphOutputs[fwdNode.Name]
		}
		bwdInputs[bwdInp.Name] = actData
	}

	// Evaluate backward graph
	bwdOutputs, err := cudaref.Evaluate(bwdRes.BackwardGraph, bwdInputs)
	if err != nil {
		return fmt.Errorf("cudaautograd: backward evaluation failed: %w", err)
	}

	// For each target input, compare against finite differences
	for id, gradNode := range bwdRes.Gradients {
		fwdTarget := bwdRes.ForwardNodes[id]
		if fwdTarget.Kind != cudair.KindInput {
			// Finite differences check operates on input variables
			continue
		}

		analyticGrad := bwdOutputs[gradNode.Name]
		numericGrad, err := FiniteDifferenceGrad(fwdGraph, lossNode, fwdTarget.Name, inputs, eps)
		if err != nil {
			return fmt.Errorf("numeric grad for %s failed: %w", fwdTarget.Name, err)
		}

		if len(analyticGrad) != len(numericGrad) {
			return fmt.Errorf("target %s grad len mismatch: analytic %d vs numeric %d",
				fwdTarget.Name, len(analyticGrad), len(numericGrad))
		}

		for i := range analyticGrad {
			a := analyticGrad[i]
			n := numericGrad[i]
			diff := float32(math.Abs(float64(a - n)))
			denom := float32(math.Max(math.Max(math.Abs(float64(a)), math.Abs(float64(n))), 1.0))
			relDiff := diff / denom
			if relDiff > tolerance {
				return fmt.Errorf("target %s at index %d: analytic=%g numeric=%g (rel diff %g > tol %g)",
					fwdTarget.Name, i, a, n, relDiff, tolerance)
			}
		}
	}

	return nil
}
