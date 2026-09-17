package cudair

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

var (
	// ErrCycleDetected is returned when a graph contains a directed cycle.
	ErrCycleDetected = errors.New("cudair: cycle detected in graph")

	// ErrMixedGraphOwnership is returned when nodes from different graphs are wired together.
	ErrMixedGraphOwnership = errors.New("cudair: mixed graph node ownership")

	// ErrUnknownOperator is returned when a node requests unknown operator semantics.
	ErrUnknownOperator = errors.New("cudair: unknown operator semantics")

	// ErrShapeMismatch is returned when tensor dimensions violate operation contracts.
	ErrShapeMismatch = errors.New("cudair: shape mismatch")

	// ErrTypeMismatch is returned when tensor data types are incompatible.
	ErrTypeMismatch = errors.New("cudair: type mismatch")

	// ErrInvalidAxis is returned when an axis index or permutation is out of bounds.
	ErrInvalidAxis = errors.New("cudair: invalid axis")

	// ErrInvalidOutput is returned when an output specification is invalid or unreachable.
	ErrInvalidOutput = errors.New("cudair: invalid graph output")

	// ErrArenaOverflow is returned when total memory arena demand exceeds address limits.
	ErrArenaOverflow = errors.New("cudair: arena memory allocation overflow")
)

// Snapshot represents an immutable, topologically validated execution graph.
type Snapshot struct {
	nodes     []*Node
	inputs    []*Node
	outputs   []*Node
	order     []*Node
	arenaPlan *ArenaPlan
	deps      map[*Node][]*Node
	consumers map[*Node][]*Node
}

// Nodes returns all nodes in the snapshot.
func (s *Snapshot) Nodes() []*Node {
	return append([]*Node(nil), s.nodes...)
}

// Inputs returns all external input nodes.
func (s *Snapshot) Inputs() []*Node {
	return append([]*Node(nil), s.inputs...)
}

// Outputs returns all graph output nodes.
func (s *Snapshot) Outputs() []*Node {
	return append([]*Node(nil), s.outputs...)
}

// Order returns the nodes sorted in safe sequential topological execution order.
func (s *Snapshot) Order() []*Node {
	return append([]*Node(nil), s.order...)
}

// ArenaPlan returns the precomputed memory scratchpad plan.
func (s *Snapshot) ArenaPlan() *ArenaPlan {
	return s.arenaPlan
}

// Dependencies returns all direct prerequisite input nodes for n.
func (s *Snapshot) Dependencies(n *Node) []*Node {
	return append([]*Node(nil), s.deps[n]...)
}

// Consumers returns all direct successor nodes that consume n.
func (s *Snapshot) Consumers(n *Node) []*Node {
	return append([]*Node(nil), s.consumers[n]...)
}

// NodeByID returns the node with the specified ID, or nil if not found.
func (s *Snapshot) NodeByID(id int) *Node {
	for _, n := range s.nodes {
		if n.ID == id {
			return n
		}
	}
	return nil
}

// Snapshot validates the computation graph and freezes it into an immutable [Snapshot].
// It checks for cycles, mixed-graph node ownership, authoritative semantic compliance,
// shape and layout validity, and constructs a safe sequential execution plan.
func (g *Graph) Snapshot() (*Snapshot, error) {
	if g.err != nil {
		return nil, fmt.Errorf("cudair: graph construction error: %w", g.err)
	}

	nodes := g.nodes
	if len(nodes) == 0 {
		return nil, errors.New("cudair: cannot snapshot empty graph")
	}

	nodeSet := make(map[*Node]bool, len(nodes))
	for _, n := range nodes {
		nodeSet[n] = true
	}

	// 1. Verify single graph ownership
	for _, n := range nodes {
		if n.graph != g {
			return nil, fmt.Errorf("%w: node %s (%p) does not belong to graph (%p)",
				ErrMixedGraphOwnership, n.Name, n.graph, g)
		}
		for _, inp := range n.Inputs {
			if inp.graph != g || !nodeSet[inp] {
				return nil, fmt.Errorf("%w: node %s references input %s from different graph",
					ErrMixedGraphOwnership, n.Name, inp.Name)
			}
		}
		if n.StorageAlias != nil && (n.StorageAlias.graph != g || !nodeSet[n.StorageAlias]) {
			return nil, fmt.Errorf("%w: node %s has storage alias from different graph",
				ErrMixedGraphOwnership, n.Name)
		}
	}

	// 2. Validate authoritative operators and shape/type semantics
	for _, n := range nodes {
		if err := validateNodeSemantics(n); err != nil {
			return nil, err
		}
	}

	// 3. Build dependency graph and detect cycles via Kahn's algorithm
	inDegree := make(map[*Node]int, len(nodes))
	deps := make(map[*Node][]*Node, len(nodes))
	consumers := make(map[*Node][]*Node, len(nodes))

	for _, n := range nodes {
		inDegree[n] = len(n.Inputs)
		deps[n] = append([]*Node(nil), n.Inputs...)
		for _, inp := range n.Inputs {
			consumers[inp] = append(consumers[inp], n)
		}
	}

	var queue []*Node
	for _, n := range nodes {
		if inDegree[n] == 0 {
			queue = append(queue, n)
		}
	}

	var order []*Node
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		order = append(order, curr)

		for _, cons := range consumers[curr] {
			inDegree[cons]--
			if inDegree[cons] == 0 {
				queue = append(queue, cons)
			}
		}
	}

	if len(order) != len(nodes) {
		return nil, fmt.Errorf("%w: cycle detected (%d sorted vs %d nodes)", ErrCycleDetected, len(order), len(nodes))
	}

	// 4. Validate outputs
	outputs := g.Outputs()
	if len(outputs) == 0 {
		return nil, fmt.Errorf("%w: no reachable outputs found", ErrInvalidOutput)
	}
	for _, out := range outputs {
		if !nodeSet[out] {
			return nil, fmt.Errorf("%w: output %s does not belong to graph", ErrInvalidOutput, out.Name)
		}
	}

	// 5. Partition inputs
	var inputs []*Node
	for _, n := range nodes {
		if n.Kind == KindInput {
			inputs = append(inputs, n)
		}
	}

	// 6. Compute safe arena plan
	plan, err := planMemorySafe(order, consumers)
	if err != nil {
		return nil, err
	}

	return &Snapshot{
		nodes:     append([]*Node(nil), nodes...),
		inputs:    inputs,
		outputs:   outputs,
		order:     order,
		arenaPlan: plan,
		deps:      deps,
		consumers: consumers,
	}, nil
}

func validateNodeSemantics(n *Node) error {
	// Extent validation
	if _, err := cudashape.New(n.Shape...); err != nil {
		return fmt.Errorf("node %s shape %v: %w", n.Name, n.Shape, err)
	}

	switch n.Kind {
	case KindInput:
		if n.Op != "input" {
			return fmt.Errorf("%w: unexpected op %q for KindInput", ErrUnknownOperator, n.Op)
		}
	case KindConstant:
		if n.Op != "constant" {
			return fmt.Errorf("%w: unexpected op %q for KindConstant", ErrUnknownOperator, n.Op)
		}
	case KindUnary:
		switch n.Op {
		case "relu", "gelu", "exp", "log", "sqrt", "neg", "silu":
			// Authoritative unary ops
		default:
			if !strings.HasPrefix(n.Op, "rmsnorm") {
				return fmt.Errorf("%w: unsupported unary op %q", ErrUnknownOperator, n.Op)
			}
		}
		if len(n.Inputs) != 1 {
			return fmt.Errorf("unary op %s expects 1 input, got %d", n.Op, len(n.Inputs))
		}
	case KindBinary:
		switch n.Op {
		case "add", "mul", "sub", "div":
			// Authoritative binary ops
		default:
			if !strings.HasPrefix(n.Op, "rope") {
				return fmt.Errorf("%w: unsupported binary op %q", ErrUnknownOperator, n.Op)
			}
		}
		if len(n.Inputs) != 2 {
			return fmt.Errorf("binary op %s expects 2 inputs, got %d", n.Op, len(n.Inputs))
		}
	case KindMatMul:
		if n.Op != "matmul" {
			return fmt.Errorf("%w: unexpected op %q for KindMatMul", ErrUnknownOperator, n.Op)
		}
		if len(n.Inputs) != 2 {
			return fmt.Errorf("matmul expects 2 inputs, got %d", len(n.Inputs))
		}
		a, b := n.Inputs[0], n.Inputs[1]
		if len(a.Shape) != 2 || len(b.Shape) != 2 {
			return fmt.Errorf("%w: matmul inputs must be 2D tensors (%v x %v)", ErrShapeMismatch, a.Shape, b.Shape)
		}
		if a.Shape[1] != b.Shape[0] {
			return fmt.Errorf("%w: matmul inner dimension mismatch (%v x %v)", ErrShapeMismatch, a.Shape, b.Shape)
		}
	case KindReduce:
		if n.Op != "sum" && n.Op != "max" && n.Op != "mean" {
			return fmt.Errorf("%w: unsupported reduction op %q", ErrUnknownOperator, n.Op)
		}
		if len(n.Inputs) != 1 {
			return fmt.Errorf("reduction expects 1 input, got %d", len(n.Inputs))
		}
		if len(n.Axes) == 0 {
			return fmt.Errorf("%w: reduction missing axis", ErrInvalidAxis)
		}
		for _, ax := range n.Axes {
			if ax < 0 || ax >= len(n.Inputs[0].Shape) {
				return fmt.Errorf("%w: reduction axis %d out of bounds for shape %v",
					ErrInvalidAxis, ax, n.Inputs[0].Shape)
			}
		}
	case KindReshape:
		if len(n.Inputs) != 1 {
			return fmt.Errorf("reshape expects 1 input, got %d", len(n.Inputs))
		}
		if n.NumElements() != n.Inputs[0].NumElements() {
			return fmt.Errorf("%w: reshape element count mismatch (%d vs %d)",
				ErrShapeMismatch, n.NumElements(), n.Inputs[0].NumElements())
		}
	case KindTranspose:
		if len(n.Inputs) != 1 {
			return fmt.Errorf("transpose expects 1 input, got %d", len(n.Inputs))
		}
		rank := len(n.Inputs[0].Shape)
		if len(n.Perm) != rank {
			return fmt.Errorf("%w: transpose perm len %d != rank %d", ErrInvalidAxis, len(n.Perm), rank)
		}
	case KindSlice:
		if len(n.Inputs) != 1 {
			return fmt.Errorf("slice expects 1 input, got %d", len(n.Inputs))
		}
	case KindConcat:
		if len(n.Inputs) == 0 {
			return fmt.Errorf("concat expects at least 1 input, got 0")
		}
	default:
		return fmt.Errorf("%w: unknown node kind %v", ErrUnknownOperator, n.Kind)
	}

	return nil
}
