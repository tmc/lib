package cudaautograd

import (
	"errors"
	"fmt"
	"math"
	"slices"

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
		value, err := func() (float32, error) {
			defer func() { targetBuf[i] = orig }()

			// f(x + eps)
			targetBuf[i] = orig + eps
			outPlus, err := cudaref.Evaluate(fwdGraph, clonedInputs)
			if err != nil {
				return 0, fmt.Errorf("finite diff f(x+eps) evaluation failed: %w", err)
			}
			lossPlus := outPlus[lossNode.Name]
			if len(lossPlus) != 1 {
				return 0, fmt.Errorf("loss node %s output length %d != 1", lossNode.Name, len(lossPlus))
			}

			// f(x - eps)
			targetBuf[i] = orig - eps
			outMinus, err := cudaref.Evaluate(fwdGraph, clonedInputs)
			if err != nil {
				return 0, fmt.Errorf("finite diff f(x-eps) evaluation failed: %w", err)
			}
			lossMinus := outMinus[lossNode.Name]
			if len(lossMinus) != 1 {
				return 0, fmt.Errorf("loss node %s output length %d != 1", lossNode.Name, len(lossMinus))
			}

			// Central difference: (f(x+eps) - f(x-eps)) / (2 * eps)
			return (lossPlus[0] - lossMinus[0]) / (2.0 * eps), nil
		}()
		if err != nil {
			return nil, err
		}
		grad[i] = value
	}

	return grad, nil
}

// CheckGradients compares analytic gradients computed by BackwardResult against finite difference gradients.
// Returns nil if maximum relative difference is within tolerance, or a descriptive error detailing mismatch.
// eps must be finite; finite values less than or equal to zero use [FiniteDifferenceGrad]'s 1e-3 default.
func CheckGradients(
	fwdGraph *cudair.Graph,
	lossNode *cudair.Node,
	bwdRes *BackwardResult,
	inputs map[string][]float32,
	eps float32,
	tolerance float32,
) error {
	if fwdGraph == nil {
		return errors.New("cudaautograd: nil forward graph")
	}
	if lossNode == nil {
		return errors.New("cudaautograd: nil loss node")
	}
	if bwdRes == nil || bwdRes.forward == nil {
		return fmt.Errorf("%w: missing backward result", ErrActivationsMismatch)
	}
	if bwdRes.BackwardGraph == nil || bwdRes.LossGradInput == nil {
		return fmt.Errorf("%w: incomplete backward result", ErrActivationsMismatch)
	}
	if math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return errors.New("cudaautograd: finite-difference step must be finite")
	}
	if math.IsNaN(float64(tolerance)) || math.IsInf(float64(tolerance), 0) || tolerance < 0 {
		return errors.New("cudaautograd: gradient tolerance must be finite and non-negative")
	}
	if !bwdRes.forward.Contains(lossNode) {
		return fmt.Errorf("%w: loss node does not belong to differentiated graph", ErrActivationsMismatch)
	}
	frozenLoss := bwdRes.forward.NodeByID(lossNode.ID)
	if frozenLoss == nil || frozenLoss.NumElements() != 1 {
		return fmt.Errorf("%w: loss node is not a scalar in differentiated graph", ErrActivationsMismatch)
	}

	if err := checkNodeMetadata("loss", lossNode, frozenLoss); err != nil {
		return err
	}

	bwdSnap, err := bwdRes.BackwardGraph.Snapshot()
	if err != nil {
		return fmt.Errorf("%w: backward graph is invalid: %w", ErrActivationsMismatch, err)
	}
	seed := bwdSnap.NodeByID(bwdRes.LossGradInput.ID)
	if seed == nil || seed.Kind != cudair.KindInput || seed.Name != bwdRes.LossGradInput.Name || !bwdSnap.Contains(bwdRes.LossGradInput) {
		return fmt.Errorf("%w: invalid loss-gradient seed", ErrActivationsMismatch)
	}
	if err := checkNodeMetadata("seed", bwdRes.LossGradInput, seed); err != nil {
		return err
	}
	if seed.NumElements() != 1 || !slices.Equal(seed.Shape, frozenLoss.Shape) {
		return fmt.Errorf("%w: loss-gradient seed is not scalar", ErrActivationsMismatch)
	}
	if len(bwdRes.Gradients) == 0 {
		return errors.New("cudaautograd: backward result has no target gradients")
	}
	if len(bwdRes.Gradients) != len(bwdRes.TargetNames) {
		return fmt.Errorf("%w: target gradient and name coverage differ", ErrActivationsMismatch)
	}

	ids := make([]int, 0, len(bwdRes.Gradients))
	for id := range bwdRes.Gradients {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	// Validate all public node descriptions before checking payloads or
	// evaluating either graph. Snapshot membership alone also admits detached
	// copies, so their mutable metadata must agree with the frozen node.
	for _, id := range ids {
		grad := bwdRes.Gradients[id]
		if grad == nil || !bwdSnap.Contains(grad) {
			return fmt.Errorf("%w: missing or foreign backward gradient for node id %d", ErrActivationsMismatch, id)
		}
		if err := checkNodeMetadata("gradient", grad, bwdSnap.NodeByID(grad.ID)); err != nil {
			return err
		}
		fwd := bwdRes.forward.NodeByID(id)
		mapped := bwdRes.ForwardNodes[id]
		if fwd == nil || mapped == nil || !bwdRes.forward.Contains(mapped) || mapped.ID != id || mapped.Name != fwd.Name || bwdRes.TargetNames[id] != fwd.Name {
			return fmt.Errorf("%w: missing forward node mapping for id %d", ErrActivationsMismatch, id)
		}
		if err := checkNodeMetadata("target", mapped, fwd); err != nil {
			return err
		}
		if !slices.Equal(grad.Shape, fwd.Shape) {
			return fmt.Errorf("%w: gradient shape for forward node id %d", ErrActivationsMismatch, id)
		}
	}
	if len(bwdRes.SavedActivations) != len(bwdSnap.Inputs())-1 {
		return fmt.Errorf("%w: saved activation input coverage differs", ErrActivationsMismatch)
	}
	savedIDs := make([]int, 0, len(bwdRes.SavedActivations))
	for id := range bwdRes.SavedActivations {
		savedIDs = append(savedIDs, id)
	}
	slices.Sort(savedIDs)
	for _, id := range savedIDs {
		fwd := bwdRes.forward.NodeByID(id)
		mapped := bwdRes.ForwardNodes[id]
		if fwd == nil {
			return fmt.Errorf("%w: missing saved forward node id %d", ErrActivationsMismatch, id)
		}
		if mapped != nil && !bwdRes.forward.Contains(mapped) {
			return fmt.Errorf("%w: foreign saved forward node id %d", ErrActivationsMismatch, id)
		}
		if err := checkNodeMetadata("saved forward", mapped, fwd); err != nil {
			return err
		}
		in := bwdRes.SavedActivations[id]
		if in == nil {
			return fmt.Errorf("%w: nil saved input for forward node id %d", ErrActivationsMismatch, id)
		}
		canonical := bwdSnap.NodeByID(in.ID)
		if err := checkNodeMetadata("saved input", in, canonical); err != nil {
			return err
		}
		if !bwdSnap.Contains(in) || canonical.Kind != cudair.KindInput || canonical.Name != "saved_"+fwd.Name || !slices.Equal(canonical.Shape, fwd.Shape) {
			return fmt.Errorf("%w: saved input role for forward node id %d", ErrActivationsMismatch, id)
		}
	}
	type target struct {
		id   int
		node *cudair.Node
	}
	var targets []target
	for _, id := range ids {
		grad := bwdRes.Gradients[id]
		if grad == nil || !bwdSnap.Contains(grad) {
			return fmt.Errorf("%w: missing or foreign backward gradient for node id %d", ErrActivationsMismatch, id)
		}
		gradFrozen := bwdSnap.NodeByID(grad.ID)
		if gradFrozen == nil || gradFrozen.Name != grad.Name {
			return fmt.Errorf("%w: changed backward gradient node for forward id %d", ErrActivationsMismatch, id)
		}
		isOutput := false
		for _, out := range bwdSnap.Outputs() {
			if out.ID == grad.ID {
				isOutput = true
				break
			}
		}
		if !isOutput {
			return fmt.Errorf("%w: backward gradient for node id %d is not an output", ErrActivationsMismatch, id)
		}
		fwdNode := bwdRes.forward.NodeByID(id)
		mappedNode := bwdRes.ForwardNodes[id]
		if fwdNode == nil || mappedNode == nil || mappedNode.ID != id || mappedNode.Name != fwdNode.Name || bwdRes.TargetNames[id] != fwdNode.Name {
			return fmt.Errorf("%w: missing forward node mapping for id %d", ErrActivationsMismatch, id)
		}
		if fwdNode.Kind != cudair.KindInput {
			continue
		}
		buf, ok := inputs[fwdNode.Name]
		if !ok {
			return fmt.Errorf("cudaautograd: missing input buffer for target %q", fwdNode.Name)
		}
		if len(buf) != fwdNode.NumElements() {
			return fmt.Errorf("cudaautograd: target input %q length %d != expected %d", fwdNode.Name, len(buf), fwdNode.NumElements())
		}
		targets = append(targets, target{id: id, node: fwdNode})
	}
	if len(targets) == 0 {
		return errors.New("cudaautograd: no input targets to compare")
	}
	isLossOutput := false
	for _, out := range bwdRes.forward.Outputs() {
		if out.ID == frozenLoss.ID {
			isLossOutput = true
			break
		}
	}
	if !isLossOutput {
		return errors.New("cudaautograd: loss node is not a forward graph output")
	}

	if _, acts, err := Forward(bwdRes, fwdGraph, inputs); err != nil {
		return fmt.Errorf("cudaautograd: forward evaluation failed: %w", err)
	} else {
		analytic, err := Backward(bwdRes, acts)
		if err != nil {
			return fmt.Errorf("cudaautograd: backward evaluation failed: %w", err)
		}
		for _, target := range targets {
			analyticGrad, ok := analytic[target.id]
			if !ok {
				return fmt.Errorf("cudaautograd: missing analytic gradient for %s", target.node.Name)
			}
			numericGrad, err := FiniteDifferenceGrad(fwdGraph, frozenLoss, target.node.Name, inputs, eps)
			if err != nil {
				return fmt.Errorf("numeric grad for %s failed: %w", target.node.Name, err)
			}
			if len(analyticGrad) != len(numericGrad) {
				return fmt.Errorf("target %s grad len mismatch: analytic %d vs numeric %d",
					target.node.Name, len(analyticGrad), len(numericGrad))
			}
			for i := range analyticGrad {
				a := analyticGrad[i]
				n := numericGrad[i]
				if math.IsNaN(float64(a)) || math.IsInf(float64(a), 0) || math.IsNaN(float64(n)) || math.IsInf(float64(n), 0) {
					return fmt.Errorf("target %s at index %d: non-finite gradient analytic=%g numeric=%g", target.node.Name, i, a, n)
				}
				diff := float32(math.Abs(float64(a - n)))
				denom := float32(math.Max(math.Max(math.Abs(float64(a)), math.Abs(float64(n))), 1.0))
				relDiff := diff / denom
				if relDiff > tolerance {
					return fmt.Errorf("target %s at index %d: analytic=%g numeric=%g (rel diff %g > tol %g)",
						target.node.Name, i, a, n, relDiff, tolerance)
				}
			}
		}
	}
	return nil
}

// checkNodeMetadata admits original nodes and detached snapshot copies with
// the same public identity, shape, role and Float32 type.
func checkNodeMetadata(role string, node, canonical *cudair.Node) error {
	if canonical == nil {
		return fmt.Errorf("%w: missing %s snapshot node", ErrActivationsMismatch, role)
	}
	if node == nil {
		return fmt.Errorf("%w: missing %s node id %d", ErrActivationsMismatch, role, canonical.ID)
	}
	if node.ID != canonical.ID || node.Name != canonical.Name || node.Kind != canonical.Kind || node.Type != canonical.Type || node.Type != cudair.Float32 || !slices.Equal(node.Shape, canonical.Shape) {
		return fmt.Errorf("%w: changed %s metadata for node id %d", ErrActivationsMismatch, role, canonical.ID)
	}
	return nil
}
