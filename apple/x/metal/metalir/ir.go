package metalir

import (
	"fmt"
	"math"
)

// DataType identifies tensor primitive data types.
type DataType int

const (
	Float32 DataType = iota
	Float16
	BFloat16
	Int32
	Int64
	Uint8
)

// SizeBytes returns the size in bytes of a single scalar element.
func (dt DataType) SizeBytes() int {
	switch dt {
	case Float32, Int32:
		return 4
	case Float16, BFloat16:
		return 2
	case Int64:
		return 8
	case Uint8:
		return 1
	default:
		return 0
	}
}

func (dt DataType) String() string {
	switch dt {
	case Float32:
		return "float32"
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint8:
		return "uint8"
	default:
		return "unknown"
	}
}

// NodeKind identifies the operator category.
type NodeKind int

const (
	KindInput NodeKind = iota
	KindConstant
	KindUnary
	KindBinary
	KindMatMul
	KindReduce
)

// SourceLocation captures host Go source attribution for an IR node.
type SourceLocation struct {
	File     string
	Line     int
	Function string
}

// Node represents an operation or tensor in the IR graph.
type Node struct {
	ID         int
	Name       string
	Kind       NodeKind
	Op         string
	Inputs     []*Node
	Shape      []int
	Type       DataType
	Parameters map[string]float64
	SourceLoc  SourceLocation
}

// NumElements returns total number of elements in the tensor.
func (n *Node) NumElements() int {
	if len(n.Shape) == 0 {
		return 1
	}
	total := 1
	for _, dim := range n.Shape {
		total *= dim
	}
	return total
}

// SizeBytes returns total buffer size in bytes needed for this tensor.
func (n *Node) SizeBytes() int {
	return n.NumElements() * n.Type.SizeBytes()
}

// Validate reports malformed shapes, types, topology, and operator contracts.
func (g *Graph) Validate() error {
	if g == nil {
		return fmt.Errorf("nil graph")
	}
	seen := make(map[*Node]bool)
	for i, n := range g.Nodes {
		if n == nil {
			return fmt.Errorf("node %d: nil node", i)
		}
		if n.ID != i {
			return fmt.Errorf("node %d: id is %d", i, n.ID)
		}
		if n.Type.SizeBytes() == 0 {
			return fmt.Errorf("node %d: invalid data type %d", i, n.Type)
		}
		elements := 1
		for _, d := range n.Shape {
			if d <= 0 {
				return fmt.Errorf("node %d: invalid dimension %d", i, d)
			}
			if elements > math.MaxInt/d {
				return fmt.Errorf("node %d: shape size overflows int", i)
			}
			elements *= d
		}
		if elements > math.MaxInt/n.Type.SizeBytes() {
			return fmt.Errorf("node %d: byte size overflows int", i)
		}
		for _, in := range n.Inputs {
			if !seen[in] {
				return fmt.Errorf("node %d: input is nil or not earlier in graph", i)
			}
		}
		if err := validateOperator(n); err != nil {
			return fmt.Errorf("node %d: %w", i, err)
		}
		seen[n] = true
	}
	return nil
}

func validateOperator(n *Node) error {
	same := func(a, b *Node) bool {
		if a.Type != b.Type || len(a.Shape) != len(b.Shape) {
			return false
		}
		for i := range a.Shape {
			if a.Shape[i] != b.Shape[i] {
				return false
			}
		}
		return true
	}
	switch n.Kind {
	case KindInput, KindConstant:
		if len(n.Inputs) != 0 {
			return fmt.Errorf("input or constant has inputs")
		}
	case KindUnary:
		if len(n.Inputs) != 1 || !same(n, n.Inputs[0]) {
			return fmt.Errorf("unary shape or type mismatch")
		}
	case KindBinary:
		if len(n.Inputs) != 2 || !same(n, n.Inputs[0]) {
			return fmt.Errorf("binary result shape or type mismatch")
		}
		switch n.Op {
		case "add", "mul":
			if !same(n, n.Inputs[1]) {
				return fmt.Errorf("binary operand shape or type mismatch")
			}
		case "rmsnorm":
			if !same(n, n.Inputs[1]) {
				return fmt.Errorf("binary operand shape or type mismatch")
			}
			epsilon := n.Parameters["epsilon"]
			if !(epsilon > 0) || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
				return fmt.Errorf("rmsnorm epsilon must be finite and positive")
			}
		case "rope":
			if n.Inputs[1].Type != n.Type || n.Inputs[1].NumElements() != 1 {
				return fmt.Errorf("rope position must be a scalar of matching type")
			}
			theta := n.Parameters["theta"]
			if !(theta > 0) || math.IsNaN(theta) || math.IsInf(theta, 0) {
				return fmt.Errorf("rope theta must be finite and positive")
			}
		default:
			return fmt.Errorf("binary operation %q unsupported", n.Op)
		}
	case KindMatMul:
		if len(n.Inputs) != 2 || len(n.Inputs[0].Shape) != 2 || len(n.Inputs[1].Shape) != 2 {
			return fmt.Errorf("matmul requires rank-2 operands")
		}
		a, b := n.Inputs[0], n.Inputs[1]
		if a.Type != b.Type || n.Type != a.Type || a.Shape[1] != b.Shape[0] || len(n.Shape) != 2 || n.Shape[0] != a.Shape[0] || n.Shape[1] != b.Shape[1] {
			return fmt.Errorf("matmul shape or type mismatch")
		}
	default:
		return fmt.Errorf("invalid kind %d", n.Kind)
	}
	return nil
}

// Graph represents a directed acyclic graph (DAG) of tensor operations.
type Graph struct {
	Nodes []*Node
}

// NewGraph creates an empty symbolic computation Graph.
func NewGraph() *Graph {
	return &Graph{
		Nodes: make([]*Node, 0),
	}
}

// AddInput creates an input placeholder node in the graph.
func (g *Graph) AddInput(name string, shape []int, dtype DataType) *Node {
	n := &Node{
		ID:     len(g.Nodes),
		Name:   name,
		Kind:   KindInput,
		Op:     "input",
		Shape:  shape,
		Type:   dtype,
		Inputs: nil,
	}
	g.Nodes = append(g.Nodes, n)
	return n
}

// AddUnary creates a unary operation node.
func (g *Graph) AddUnary(op string, in *Node) *Node {
	n := &Node{
		ID:     len(g.Nodes),
		Name:   fmt.Sprintf("%s_%d", op, len(g.Nodes)),
		Kind:   KindUnary,
		Op:     op,
		Shape:  in.Shape,
		Type:   in.Type,
		Inputs: []*Node{in},
	}
	g.Nodes = append(g.Nodes, n)
	return n
}

// AddBinary creates a binary operation node.
func (g *Graph) AddBinary(op string, a, b *Node) *Node {
	n := &Node{
		ID:     len(g.Nodes),
		Name:   fmt.Sprintf("%s_%d", op, len(g.Nodes)),
		Kind:   KindBinary,
		Op:     op,
		Shape:  a.Shape,
		Type:   a.Type,
		Inputs: []*Node{a, b},
	}
	g.Nodes = append(g.Nodes, n)
	return n
}

// AddRMSNorm creates an RMS normalization node with an explicit epsilon.
func (g *Graph) AddRMSNorm(in, weight *Node, epsilon float32) *Node {
	n := g.AddBinary("rmsnorm", in, weight)
	n.Parameters = map[string]float64{"epsilon": float64(epsilon)}
	return n
}

// AddRoPE creates a rotary-position-embedding node with an explicit theta.
func (g *Graph) AddRoPE(in, position *Node, theta float32) *Node {
	n := g.AddBinary("rope", in, position)
	n.Parameters = map[string]float64{"theta": float64(theta)}
	return n
}

// AddMatMul creates a matrix multiplication node.
func (g *Graph) AddMatMul(a, b *Node) *Node {
	var shape []int
	var dtype DataType
	if a != nil {
		dtype = a.Type
	}
	if a != nil && b != nil && len(a.Shape) == 2 && len(b.Shape) == 2 {
		shape = []int{a.Shape[0], b.Shape[1]}
	}
	n := &Node{
		ID:     len(g.Nodes),
		Name:   fmt.Sprintf("matmul_%d", len(g.Nodes)),
		Kind:   KindMatMul,
		Op:     "matmul",
		Shape:  shape,
		Type:   dtype,
		Inputs: []*Node{a, b},
	}
	g.Nodes = append(g.Nodes, n)
	return n
}
