package metalir

import (
	"fmt"
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
		return 4
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
	ID        int
	Name      string
	Kind      NodeKind
	Op        string
	Inputs    []*Node
	Shape     []int
	Type      DataType
	SourceLoc SourceLocation
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

// AddMatMul creates a matrix multiplication node.
func (g *Graph) AddMatMul(a, b *Node) *Node {
	shape := []int{a.Shape[0], b.Shape[1]}
	n := &Node{
		ID:     len(g.Nodes),
		Name:   fmt.Sprintf("matmul_%d", len(g.Nodes)),
		Kind:   KindMatMul,
		Op:     "matmul",
		Shape:  shape,
		Type:   a.Type,
		Inputs: []*Node{a, b},
	}
	g.Nodes = append(g.Nodes, n)
	return n
}
