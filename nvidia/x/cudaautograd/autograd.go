package cudaautograd

import (
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
)

var (
	// ErrUnsupportedDerivative indicates an operator or data type for which derivatives are undefined or unsupported.
	ErrUnsupportedDerivative = errors.New("cudaautograd: unsupported derivative")

	// ErrUnsupportedMutation indicates that a forward graph contains mutating state effects that cannot be differentiated.
	ErrUnsupportedMutation = errors.New("cudaautograd: differentiation of mutating state is unsupported")

	// ErrTargetNotFound indicates a requested target node is not part of the forward graph.
	ErrTargetNotFound = errors.New("cudaautograd: differentiation target not in graph")

	// ErrLossNotScalar indicates the specified loss node is not a 1-element scalar.
	ErrLossNotScalar = errors.New("cudaautograd: loss must be a scalar with element count 1")

	// ErrSavedActivationStale indicates that a saved activation changed
	// length or contents between forward and backward evaluation, or that
	// the activations were already consumed.
	ErrSavedActivationStale = errors.New("cudaautograd: saved activation was invalidated or overwritten")

	// ErrActivationsMismatch indicates activations or a forward graph that do
	// not belong to the backward result they are used with.
	ErrActivationsMismatch = errors.New("cudaautograd: activations do not belong to backward result")
)

// BackwardResult holds the constructed backward graph and metadata mapping targets to their gradient outputs.
type BackwardResult struct {
	// BackwardGraph is the synthesized backward DAG.
	BackwardGraph *cudair.Graph

	// LossGradInput is the input node representing dLoss/dLoss (seed gradient, scalar 1.0).
	LossGradInput *cudair.Node

	// SavedActivations maps forward node IDs to the corresponding backward input nodes that supply them.
	SavedActivations map[int]*cudair.Node

	// Gradients maps target forward node IDs to their gradient nodes in BackwardGraph.
	Gradients map[int]*cudair.Node

	// TargetNames maps target forward node IDs to target names.
	TargetNames map[int]string

	// ForwardNodes retains references to original forward nodes for lifetime and identity checks.
	ForwardNodes map[int]*cudair.Node

	forward       *cudair.Snapshot // snapshot the result was derived from
	forwardDigest [32]byte         // semantic identity of forward
}

// SavedActivationDescriptor details an activation tensor needed by the backward pass.
type SavedActivationDescriptor struct {
	NodeID    int
	NodeName  string
	Shape     []int
	DataType  cudair.DataType
	SizeBytes int
}

// SavedDescriptors returns descriptors of all activations required by the backward pass.
func (br *BackwardResult) SavedDescriptors() []SavedActivationDescriptor {
	descs := make([]SavedActivationDescriptor, 0, len(br.SavedActivations))
	for id := range br.SavedActivations {
		fwd := br.ForwardNodes[id]
		if fwd != nil {
			descs = append(descs, SavedActivationDescriptor{
				NodeID:    fwd.ID,
				NodeName:  fwd.Name,
				Shape:     append([]int(nil), fwd.Shape...),
				DataType:  fwd.Type,
				SizeBytes: fwd.SizeBytes(),
			})
		}
	}
	slices.SortFunc(descs, func(a, b SavedActivationDescriptor) int {
		return a.NodeID - b.NodeID
	})
	return descs
}

// TargetGradientNode returns the backward graph node computing the gradient for the given forward node.
func (br *BackwardResult) TargetGradientNode(target *cudair.Node) (*cudair.Node, bool) {
	if target == nil {
		return nil, false
	}
	n, ok := br.Gradients[target.ID]
	return n, ok
}

// Differentiate derives a backward graph computing gradients of loss with respect to targets.
// The forward graph is not modified.
func Differentiate(fwdSnap *cudair.Snapshot, loss *cudair.Node, targets []*cudair.Node) (*BackwardResult, error) {
	if fwdSnap == nil {
		return nil, errors.New("cudaautograd: nil forward snapshot")
	}
	if loss == nil {
		return nil, errors.New("cudaautograd: nil loss node")
	}
	if len(targets) == 0 {
		return nil, errors.New("cudaautograd: no differentiation targets specified")
	}

	if !fwdSnap.Contains(loss) {
		return nil, fmt.Errorf("%w: loss does not belong to snapshot", ErrTargetNotFound)
	}
	// Verify loss belongs to snapshot
	foundLoss := false
	for _, n := range fwdSnap.Nodes() {
		if n.ID == loss.ID {
			foundLoss = true
			loss = n
			break
		}
	}
	if !foundLoss {
		return nil, fmt.Errorf("%w: loss node %q not found in snapshot", ErrTargetNotFound, loss.Name)
	}

	// Verify loss is scalar (1 element)
	if loss.NumElements() != 1 {
		return nil, fmt.Errorf("%w: loss shape is %v (%d elements)", ErrLossNotScalar, loss.Shape, loss.NumElements())
	}

	// Verify targets belong to snapshot
	fwdNodeMap := make(map[int]*cudair.Node, len(fwdSnap.Nodes()))
	for _, n := range fwdSnap.Nodes() {
		fwdNodeMap[n.ID] = n
	}
	targets = append([]*cudair.Node(nil), targets...)
	targetSet := make(map[int]bool, len(targets))
	targetNames := make(map[int]string, len(targets))
	for i, t := range targets {
		if t == nil {
			return nil, errors.New("cudaautograd: nil target in targets list")
		}
		if !fwdSnap.Contains(t) {
			return nil, fmt.Errorf("%w: target %q (id %d)", ErrTargetNotFound, t.Name, t.ID)
		}
		t = fwdNodeMap[t.ID]
		if err := checkDataType("target", t); err != nil {
			return nil, err
		}
		targets[i] = t
		targetSet[t.ID] = true
		targetNames[t.ID] = t.Name
	}

	// Check for state mutating effects in forward snapshot
	for _, n := range fwdSnap.Nodes() {
		if n.Effect == cudair.EffectMutating {
			return nil, fmt.Errorf("%w: node %s has mutating storage effect", ErrUnsupportedMutation, n.Name)
		}
	}

	// Construct new backward graph
	bwd := cudair.NewGraph()

	// Seed gradient dLoss/dLoss = 1.0
	seedInput := bwd.AddInput("grad_"+loss.Name, loss.Shape, loss.Type)

	// Context for building backward graph
	ctx := &diffContext{
		bwd:              bwd,
		fwdSnap:          fwdSnap,
		fwdNodes:         fwdNodeMap,
		savedActivations: make(map[int]*cudair.Node),
		adjoints:         make(map[int][]*cudair.Node), // forward NodeID -> list of incoming adjoints
	}

	// Initial adjoint for loss
	ctx.adjoints[loss.ID] = []*cudair.Node{seedInput}

	// Reverse topological traversal over order
	order := fwdSnap.Order()
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		inAdjoints := ctx.adjoints[n.ID]
		if len(inAdjoints) == 0 {
			// No gradient flowed to this node
			continue
		}
		// The loss is the first node visited, so this also checks it.
		if err := checkDataType("node", n); err != nil {
			return nil, err
		}

		// Accumulate adjoints if multiple consumers contributed
		gradN := ctx.accumulate(inAdjoints, n.Shape, n.Type, "grad_"+n.Name)

		// Record if this node is a target
		if targetSet[n.ID] {
			ctx.adjoints[n.ID] = []*cudair.Node{gradN}
		}

		if n.Kind == cudair.KindInput || n.Kind == cudair.KindConstant {
			// Inputs and constants have no upstream parents in the forward graph
			continue
		}

		// Propagate backward through node's operator
		if err := ctx.propagate(n, gradN); err != nil {
			return nil, err
		}
	}

	// Ensure all targets have gradients; if a target had no gradient flowing to it, emit zero gradient
	targetGradients := make(map[int]*cudair.Node, len(targets))
	for _, t := range targets {
		grads := ctx.adjoints[t.ID]
		if len(grads) == 0 {
			// Target did not contribute to loss: gradient is constant 0
			zeros := make([]float32, t.NumElements())
			zConst := bwd.AddConstant("zero_grad_"+t.Name, t.Shape, t.Type, zeros)
			targetGradients[t.ID] = zConst
			if err := bwd.AddOutput(zConst); err != nil {
				return nil, err
			}
		} else {
			grad := ctx.accumulate(grads, t.Shape, t.Type, "final_grad_"+t.Name)
			targetGradients[t.ID] = grad
			if err := bwd.AddOutput(grad); err != nil {
				return nil, err
			}
		}
	}

	return &BackwardResult{
		BackwardGraph:    bwd,
		LossGradInput:    seedInput,
		SavedActivations: ctx.savedActivations,
		Gradients:        targetGradients,
		TargetNames:      targetNames,
		ForwardNodes:     fwdNodeMap,
		forward:          fwdSnap,
		forwardDigest:    digestSnapshot(fwdSnap),
	}, nil
}

// checkDataType enforces the differentiation data type policy.
// Only [cudair.Float32] is admitted: seeds, saved activations, and
// gradients take the type of their forward node, and float32 is the only
// type with a defined gradient and accumulation type. Integer types have
// no derivative. [cudair.Float16] and [cudair.BFloat16] are refused until
// a mixed-precision accumulation policy exists.
func checkDataType(role string, n *cudair.Node) error {
	if n.Type != cudair.Float32 {
		return fmt.Errorf("%w: %s %s has data type %v, want %v", ErrUnsupportedDerivative, role, n.Name, n.Type, cudair.Float32)
	}
	return nil
}

type diffContext struct {
	bwd              *cudair.Graph
	fwdSnap          *cudair.Snapshot
	fwdNodes         map[int]*cudair.Node
	savedActivations map[int]*cudair.Node
	adjoints         map[int][]*cudair.Node
}

func (ctx *diffContext) saveActivation(fwd *cudair.Node) *cudair.Node {
	if saved, ok := ctx.savedActivations[fwd.ID]; ok {
		return saved
	}
	saved := ctx.bwd.AddInput("saved_"+fwd.Name, fwd.Shape, fwd.Type)
	ctx.savedActivations[fwd.ID] = saved
	return saved
}

func (ctx *diffContext) addAdjoint(fwdNode *cudair.Node, adj *cudair.Node) {
	ctx.adjoints[fwdNode.ID] = append(ctx.adjoints[fwdNode.ID], adj)
}

func (ctx *diffContext) accumulate(grads []*cudair.Node, shape []int, dt cudair.DataType, name string) *cudair.Node {
	if len(grads) == 1 {
		return grads[0]
	}
	acc := grads[0]
	for i := 1; i < len(grads); i++ {
		acc = ctx.bwd.AddBinary("add", acc, grads[i])
	}
	return acc
}

func (ctx *diffContext) propagate(n *cudair.Node, gradN *cudair.Node) error {
	switch n.Kind {
	case cudair.KindUnary:
		return ctx.propagateUnary(n, gradN)
	case cudair.KindBinary:
		return ctx.propagateBinary(n, gradN)
	case cudair.KindMatMul:
		return ctx.propagateMatMul(n, gradN)
	case cudair.KindReduce:
		return ctx.propagateReduce(n, gradN)
	case cudair.KindReshape:
		return ctx.propagateReshape(n, gradN)
	case cudair.KindTranspose:
		return ctx.propagateTranspose(n, gradN)
	case cudair.KindSlice:
		return ctx.propagateSlice(n, gradN)
	case cudair.KindConcat:
		return ctx.propagateConcat(n, gradN)
	default:
		return fmt.Errorf("%w: node kind %v for node %s", ErrUnsupportedDerivative, n.Kind, n.Name)
	}
}

func (ctx *diffContext) propagateUnary(n *cudair.Node, gradN *cudair.Node) error {
	x := n.Inputs[0]
	switch n.Op {
	case "neg":
		// y = -x => dx = -dy
		dx := ctx.bwd.AddUnary("neg", gradN)
		ctx.addAdjoint(x, dx)
		return nil

	case "exp":
		// y = exp(x) => dx = dy * exp(x) = dy * y
		savedY := ctx.saveActivation(n)
		dx := ctx.bwd.AddBinary("mul", gradN, savedY)
		ctx.addAdjoint(x, dx)
		return nil

	case "log":
		// y = log(x) => dx = dy / x
		savedX := ctx.saveActivation(x)
		dx := ctx.bwd.AddBinary("div", gradN, savedX)
		ctx.addAdjoint(x, dx)
		return nil

	case "sqrt":
		// y = sqrt(x) => dx = dy / (2 * sqrt(x)) = dy / (2 * y)
		savedY := ctx.saveActivation(n)
		twoConst := ctx.bwd.AddConstant(fmt.Sprintf("two_%s", n.Name), n.Shape, n.Type, constFilled(n.NumElements(), 2.0))
		twoY := ctx.bwd.AddBinary("mul", savedY, twoConst)
		dx := ctx.bwd.AddBinary("div", gradN, twoY)
		ctx.addAdjoint(x, dx)
		return nil

	case "relu":
		// relu_grad uses the declared zero subgradient at x == 0.
		savedX := ctx.saveActivation(x)
		mask := ctx.bwd.AddUnary("relu_grad", savedX)
		dx := ctx.bwd.AddBinary("mul", gradN, mask)
		ctx.addAdjoint(x, dx)
		return nil

	case "gelu":
		savedX := ctx.saveActivation(x)
		dGelu := ctx.bwd.AddUnary("gelu_grad", savedX)
		dx := ctx.bwd.AddBinary("mul", gradN, dGelu)
		ctx.addAdjoint(x, dx)
		return nil

	default:
		return fmt.Errorf("%w: unary op %q", ErrUnsupportedDerivative, n.Op)
	}
}

func (ctx *diffContext) propagateBinary(n *cudair.Node, gradN *cudair.Node) error {
	a := n.Inputs[0]
	b := n.Inputs[1]

	switch n.Op {
	case "add":
		// y = a + b => da = dy, db = dy (with broadcast reduction if shapes differ)
		da := ctx.reduceBroadcast(gradN, a.Shape)
		db := ctx.reduceBroadcast(gradN, b.Shape)
		ctx.addAdjoint(a, da)
		ctx.addAdjoint(b, db)
		return nil

	case "sub":
		// y = a - b => da = dy, db = -dy
		da := ctx.reduceBroadcast(gradN, a.Shape)
		negGrad := ctx.bwd.AddUnary("neg", gradN)
		db := ctx.reduceBroadcast(negGrad, b.Shape)
		ctx.addAdjoint(a, da)
		ctx.addAdjoint(b, db)
		return nil

	case "mul":
		// y = a * b => da = dy * b, db = dy * a
		savedA := ctx.saveActivation(a)
		savedB := ctx.saveActivation(b)

		rawDa := ctx.bwd.AddBinary("mul", gradN, savedB)
		rawDb := ctx.bwd.AddBinary("mul", gradN, savedA)

		da := ctx.reduceBroadcast(rawDa, a.Shape)
		db := ctx.reduceBroadcast(rawDb, b.Shape)
		ctx.addAdjoint(a, da)
		ctx.addAdjoint(b, db)
		return nil

	case "div":
		// y = a / b => da = dy / b, db = -dy * a / b^2
		savedA := ctx.saveActivation(a)
		savedB := ctx.saveActivation(b)

		rawDa := ctx.bwd.AddBinary("div", gradN, savedB)

		bSquared := ctx.bwd.AddBinary("mul", savedB, savedB)
		aOverB2 := ctx.bwd.AddBinary("div", savedA, bSquared)
		negAOverB2 := ctx.bwd.AddUnary("neg", aOverB2)
		rawDb := ctx.bwd.AddBinary("mul", gradN, negAOverB2)

		da := ctx.reduceBroadcast(rawDa, a.Shape)
		db := ctx.reduceBroadcast(rawDb, b.Shape)
		ctx.addAdjoint(a, da)
		ctx.addAdjoint(b, db)
		return nil

	default:
		return fmt.Errorf("%w: binary op %q", ErrUnsupportedDerivative, n.Op)
	}
}

func (ctx *diffContext) propagateMatMul(n *cudair.Node, gradN *cudair.Node) error {
	// y = a x b, where a is (M, K), b is (K, N), y is (M, N)
	// da = dy x b^T  => (M, N) x (N, K) = (M, K)
	// db = a^T x dy  => (K, M) x (M, N) = (K, N)
	a := n.Inputs[0]
	b := n.Inputs[1]

	savedA := ctx.saveActivation(a)
	savedB := ctx.saveActivation(b)

	// b^T has perm [1, 0]
	bT := ctx.bwd.AddTranspose(savedB, []int{1, 0})
	da := ctx.bwd.AddMatMul(gradN, bT)

	// a^T has perm [1, 0]
	aT := ctx.bwd.AddTranspose(savedA, []int{1, 0})
	db := ctx.bwd.AddMatMul(aT, gradN)

	ctx.addAdjoint(a, da)
	ctx.addAdjoint(b, db)
	return nil
}

func (ctx *diffContext) propagateReduce(n *cudair.Node, gradN *cudair.Node) error {
	x := n.Inputs[0]
	axis := n.Axes[0]

	switch n.Op {
	case "sum":
		// y = sum(x, axis).
		// Forward reduction collapsed axis. Expand back to x.Shape:
		// We construct an expansion graph by reshaping and tiling/multiplying by ones.
		// For rank-2 (M, K):
		// if axis == 0 (reduced to [K]): reshape grad to [1, K], mul by ones [M, K]
		// if axis == 1 (reduced to [M]): reshape grad to [M, 1], mul by ones [M, K]
		expandedShape := make([]int, len(x.Shape))
		for i, d := range x.Shape {
			if i == axis {
				expandedShape[i] = 1
			} else {
				expandedShape[i] = d
			}
		}

		reshapedGrad := gradN
		if !slices.Equal(gradN.Shape, expandedShape) {
			reshapedGrad = ctx.bwd.AddReshape(gradN, expandedShape)
		}

		if slices.Equal(expandedShape, x.Shape) {
			ctx.addAdjoint(x, reshapedGrad)
			return nil
		}

		// To expand reshapedGrad [M, 1] or [1, K] to [M, K]:
		// In pure cudair without tile operator, we can expand a (1, K) vector to (M, K)
		// via MatMul: ones(M, 1) x grad(1, K) = (M, K)!
		// Or expand a (M, 1) vector to (M, K) via MatMul:
		// grad(M, 1) x ones(1, K) = (M, K)!
		if len(x.Shape) == 2 {
			M, K := x.Shape[0], x.Shape[1]
			if axis == 0 {
				// grad is (1, K). ones is (M, 1). ones x grad = (M, K)
				onesM1 := ctx.bwd.AddConstant(fmt.Sprintf("ones_m_%s", n.Name), []int{M, 1}, x.Type, constFilled(M, 1.0))
				dx := ctx.bwd.AddMatMul(onesM1, reshapedGrad)
				ctx.addAdjoint(x, dx)
				return nil
			} else if axis == 1 {
				// grad is (M, 1). ones is (1, K). grad x ones = (M, K)
				ones1K := ctx.bwd.AddConstant(fmt.Sprintf("ones_k_%s", n.Name), []int{1, K}, x.Type, constFilled(K, 1.0))
				dx := ctx.bwd.AddMatMul(reshapedGrad, ones1K)
				ctx.addAdjoint(x, dx)
				return nil
			}
		}

		// For rank-1: reducing [N] to [1]:
		if len(x.Shape) == 1 {
			N := x.Shape[0]
			// grad is [1]. Expand [1] to [N] using matmul with (1, N) or concat
			// [1] can be reshaped to [1, 1], then [1, 1] x [1, N] = [1, N], then reshape to [N]
			g2d := ctx.bwd.AddReshape(reshapedGrad, []int{1, 1})
			ones1N := ctx.bwd.AddConstant(fmt.Sprintf("ones_1n_%s", n.Name), []int{1, N}, x.Type, constFilled(N, 1.0))
			prod := ctx.bwd.AddMatMul(g2d, ones1N)
			dx := ctx.bwd.AddReshape(prod, x.Shape)
			ctx.addAdjoint(x, dx)
			return nil
		}

		return fmt.Errorf("%w: reduce sum expansion for rank %d", ErrUnsupportedDerivative, len(x.Shape))

	case "max":
		// Discontinuous and non-differentiable at points of equal maximum.
		return fmt.Errorf("%w: reduce max is discontinuous and non-differentiable", ErrUnsupportedDerivative)

	default:
		return fmt.Errorf("%w: reduce op %q", ErrUnsupportedDerivative, n.Op)
	}
}

func (ctx *diffContext) propagateReshape(n *cudair.Node, gradN *cudair.Node) error {
	x := n.Inputs[0]
	dx := ctx.bwd.AddReshape(gradN, x.Shape)
	ctx.addAdjoint(x, dx)
	return nil
}

func (ctx *diffContext) propagateTranspose(n *cudair.Node, gradN *cudair.Node) error {
	x := n.Inputs[0]
	invPerm := make([]int, len(n.Perm))
	for i, p := range n.Perm {
		invPerm[p] = i
	}
	dx := ctx.bwd.AddTranspose(gradN, invPerm)
	ctx.addAdjoint(x, dx)
	return nil
}

func (ctx *diffContext) propagateSlice(n *cudair.Node, gradN *cudair.Node) error {
	return fmt.Errorf("%w: slice backward scatter is not supported", ErrUnsupportedDerivative)
}

func (ctx *diffContext) propagateConcat(n *cudair.Node, gradN *cudair.Node) error {
	axis := n.Axis
	offset := 0
	rank := len(gradN.Shape)
	for _, inp := range n.Inputs {
		inpExtent := inp.Shape[axis]
		starts := make([]int, rank)
		ends := append([]int(nil), gradN.Shape...)
		steps := make([]int, rank)
		for d := 0; d < rank; d++ {
			steps[d] = 1
		}
		starts[axis] = offset
		ends[axis] = offset + inpExtent
		dx := ctx.bwd.AddSlice(gradN, starts, ends, steps)
		ctx.addAdjoint(inp, dx)
		offset += inpExtent
	}
	return nil
}

// reduceBroadcast handles gradient reduction when an operand was broadcast along one or more axes.
func (ctx *diffContext) reduceBroadcast(gradNode *cudair.Node, targetShape []int) *cudair.Node {
	if slices.Equal(gradNode.Shape, targetShape) {
		return gradNode
	}

	current := gradNode
	scalar := true
	for _, extent := range targetShape {
		if extent != 1 {
			scalar = false
			break
		}
	}
	if scalar {
		// Scalar broadcasting duplicates one value across the entire result.
		// Sum every result dimension, then restore the operand's exact rank.
		// AddReduce deliberately retains [1] after reducing rank one, so a
		// rank-based loop down to a zero-dimensional target would never finish.
		for len(current.Shape) > 1 {
			current = ctx.bwd.AddReduce("sum", current, 0)
		}
		if current.NumElements() != 1 {
			current = ctx.bwd.AddReduce("sum", current, 0)
		}
		if !slices.Equal(current.Shape, targetShape) {
			current = ctx.bwd.AddReshape(current, targetShape)
		}
		return current
	}
	// If rank differs, reduce leading axes
	for len(current.Shape) > len(targetShape) {
		current = ctx.bwd.AddReduce("sum", current, 0)
	}

	// Reduce along axes where targetShape has dimension 1 but current has dimension > 1
	for axis := 0; axis < len(targetShape); axis++ {
		if targetShape[axis] == 1 && current.Shape[axis] > 1 {
			current = ctx.bwd.AddReduce("sum", current, axis)
			expShape := make([]int, len(current.Shape)+1)
			copy(expShape, current.Shape[:axis])
			expShape[axis] = 1
			copy(expShape[axis+1:], current.Shape[axis:])
			current = ctx.bwd.AddReshape(current, expShape)
		}
	}

	if !slices.Equal(current.Shape, targetShape) {
		current = ctx.bwd.AddReshape(current, targetShape)
	}
	return current
}

func constFilled(n int, val float32) []float32 {
	buf := make([]float32, n)
	for i := range buf {
		buf[i] = val
	}
	return buf
}
