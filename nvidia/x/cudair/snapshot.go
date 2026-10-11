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

	// ErrUnsupportedEffect reports a storage effect without admitted execution semantics.
	ErrUnsupportedEffect = errors.New("cudair: unsupported storage effect")
)

// Snapshot represents an immutable, topologically validated execution graph.
// Its accessors return detached copies. Node IDs are stable across copies;
// node pointer addresses and graph pointers are not.
type Snapshot struct {
	source      *Graph
	originalIDs map[*Node]int
	nodes       []*Node
	inputs      []*Node
	outputs     []*Node
	order       []*Node
	arenaPlan   *ArenaPlan
	deps        map[*Node][]*Node
	consumers   map[*Node][]*Node
}

// Nodes returns all nodes in the snapshot.
func (s *Snapshot) Nodes() []*Node {
	return s.clone().nodes
}

// Inputs returns all external input nodes.
func (s *Snapshot) Inputs() []*Node {
	return s.clone().inputs
}

// Outputs returns all graph output nodes.
func (s *Snapshot) Outputs() []*Node {
	return s.clone().outputs
}

// Order returns the nodes sorted in safe sequential topological execution order.
func (s *Snapshot) Order() []*Node {
	return s.clone().order
}

// ArenaPlan returns the precomputed memory scratchpad plan.
func (s *Snapshot) ArenaPlan() *ArenaPlan {
	return s.clone().arenaPlan
}

// Dependencies returns all direct prerequisite input nodes for n.
func (s *Snapshot) Dependencies(n *Node) []*Node {
	if n = s.ownedNode(n); n == nil {
		return nil
	}
	c := s.clone()
	for _, own := range c.nodes {
		if own.ID == n.ID {
			return c.deps[own]
		}
	}
	return nil
}

// Consumers returns all direct successor nodes that consume n.
func (s *Snapshot) Consumers(n *Node) []*Node {
	if n = s.ownedNode(n); n == nil {
		return nil
	}
	c := s.clone()
	for _, own := range c.nodes {
		if own.ID == n.ID {
			return c.consumers[own]
		}
	}
	return nil
}

// NodeByID returns the node with the specified ID, or nil if not found.
func (s *Snapshot) NodeByID(id int) *Node {
	for _, n := range s.clone().nodes {
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
	ids := make(map[int]bool, len(nodes))
	for _, n := range nodes {
		if n == nil || ids[n.ID] {
			return nil, errors.New("cudair: invalid node identity")
		}
		ids[n.ID] = true
		nodeSet[n] = true
	}

	// 1. Verify single graph ownership
	for _, n := range nodes {
		if n.graph != g {
			return nil, fmt.Errorf("%w: node %s (%p) does not belong to graph (%p)",
				ErrMixedGraphOwnership, n.Name, n.graph, g)
		}
		for _, inp := range n.Inputs {
			if inp == nil || inp.graph != g || !nodeSet[inp] {
				return nil, fmt.Errorf("%w: node %s references an input from a different graph",
					ErrMixedGraphOwnership, n.Name)
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

	originalIDs := make(map[*Node]int, len(nodes))
	for _, n := range nodes {
		originalIDs[n] = n.ID
	}
	return (&Snapshot{
		originalIDs: originalIDs,
		source:      g,
		nodes:       append([]*Node(nil), nodes...),
		inputs:      inputs,
		outputs:     outputs,
		order:       order,
		arenaPlan:   plan,
		deps:        deps,
		consumers:   consumers,
	}).clone(), nil
}

func validateNodeSemantics(n *Node) error {
	switch n.Effect {
	case EffectReadOnly:
		if n.StorageAlias != nil {
			return fmt.Errorf("%w: read-only node %s has an alias", ErrUnsupportedEffect, n.Name)
		}
	case EffectInPlaceAlias:
		if n.Kind != KindReshape || len(n.Inputs) != 1 || n.StorageAlias != n.Inputs[0] {
			return fmt.Errorf("%w: node %s has an unsupported alias", ErrUnsupportedEffect, n.Name)
		}
	default:
		return fmt.Errorf("%w: node %s effect %s", ErrUnsupportedEffect, n.Name, n.Effect)
	}
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
		shape, err := binaryShape(n.Inputs[0].Shape, n.Inputs[1].Shape)
		if err != nil {
			return err
		}
		if !equalShape(n.Shape, shape) {
			return fmt.Errorf("%w: binary output shape %v, want %v", ErrShapeMismatch, n.Shape, shape)
		}
		if n.Type != n.Inputs[0].Type || n.Type != n.Inputs[1].Type {
			return fmt.Errorf("%w: binary op %s", ErrTypeMismatch, n.Op)
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

// clone returns a fully detached graph view. IDs, not pointer addresses,
// identify nodes across views. No mutable slice, map, graph or node is shared.
func (s *Snapshot) clone() *Snapshot {
	c := &Snapshot{source: s.source, originalIDs: s.originalIDs, deps: make(map[*Node][]*Node), consumers: make(map[*Node][]*Node)}
	g := &Graph{snapshotSource: s.source}
	copies := make(map[*Node]*Node, len(s.nodes))
	for _, n := range s.nodes {
		v := *n
		v.graph = g
		if _, original := s.originalIDs[n]; original {
			v.snapshotOriginal = n
		}
		v.Inputs = append([]*Node(nil), n.Inputs...)
		v.Shape = append([]int(nil), n.Shape...)
		v.Strides = append([]int(nil), n.Strides...)
		v.ConstValues = append([]float32(nil), n.ConstValues...)
		v.Axes = append([]int(nil), n.Axes...)
		v.Perm = append([]int(nil), n.Perm...)
		v.SliceStarts = append([]int(nil), n.SliceStarts...)
		v.SliceEnds = append([]int(nil), n.SliceEnds...)
		v.SliceSteps = append([]int(nil), n.SliceSteps...)
		v.SourceLoc.Callers = append([]Frame(nil), n.SourceLoc.Callers...)
		copies[n] = &v
	}
	remap := func(nodes []*Node) []*Node {
		result := make([]*Node, len(nodes))
		for i, n := range nodes {
			result[i] = copies[n]
		}
		return result
	}
	for n, v := range copies {
		v.Inputs = remap(n.Inputs)
		v.StorageAlias = copies[n.StorageAlias]
		c.deps[v] = remap(s.deps[n])
		c.consumers[v] = remap(s.consumers[n])
	}
	c.nodes, c.inputs, c.outputs, c.order = remap(s.nodes), remap(s.inputs), remap(s.outputs), remap(s.order)
	g.nodes, g.outputs = c.nodes, c.outputs
	if s.arenaPlan != nil {
		c.arenaPlan = &ArenaPlan{TotalBytes: s.arenaPlan.TotalBytes, Offsets: make(map[*Node]int)}
		for n, offset := range s.arenaPlan.Offsets {
			c.arenaPlan.Offsets[copies[n]] = offset
		}
	}
	return c
}

func (s *Snapshot) ownedNode(n *Node) *Node {
	if n == nil || n.graph == nil {
		return nil
	}
	if n.graph == s.source {
		if id, ok := s.originalIDs[n]; !ok || id != n.ID {
			return nil
		}
	} else if n.graph.snapshotSource != s.source || n.snapshotOriginal == nil {
		return nil
	}
	if n.graph != s.source {
		if id, ok := s.originalIDs[n.snapshotOriginal]; !ok || id != n.ID {
			return nil
		}
	}
	for _, own := range s.nodes {
		if own.ID == n.ID {
			return own
		}
	}
	return nil
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, extent := range a {
		if extent != b[i] {
			return false
		}
	}
	return true
}

// Contains reports whether n belongs to this snapshot and retains its original ID.
// Both original graph nodes and detached nodes from snapshot accessors are accepted.
// Foreign graph nodes and copies with changed IDs are rejected.
func (s *Snapshot) Contains(n *Node) bool { return s != nil && s.ownedNode(n) != nil }

func binaryShape(a, b []int) ([]int, error) {
	as, err := cudashape.New(a...)
	if err != nil {
		return nil, fmt.Errorf("binary input shape: %w", err)
	}
	bs, err := cudashape.New(b...)
	if err != nil {
		return nil, fmt.Errorf("binary input shape: %w", err)
	}
	an, _ := as.NumElementsSafe()
	bn, _ := bs.NumElementsSafe()
	switch {
	case equalShape(a, b), bn == 1:
		return append([]int(nil), a...), nil
	case an == 1:
		return append([]int(nil), b...), nil
	default:
		return nil, fmt.Errorf("%w: binary input shapes %v and %v; nonscalar broadcasting is unsupported", ErrShapeMismatch, a, b)
	}
}
