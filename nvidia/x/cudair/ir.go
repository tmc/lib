package cudair

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

// Graph represents a directed acyclic computation graph.
type Graph struct {
	nodes []*Node
}

// NewGraph creates an empty IR Graph.
func NewGraph() *Graph {
	return &Graph{}
}

// Nodes returns all nodes in the graph.
func (g *Graph) Nodes() []*Node {
	return g.nodes
}

func (g *Graph) addNode(n *Node) *Node {
	n.ID = len(g.nodes)
	g.nodes = append(g.nodes, n)
	return n
}

// AddInput creates an external input node.
func (g *Graph) AddInput(name string, shape []int, dt DataType) *Node {
	return g.addNode(&Node{
		Name:  name,
		Kind:  KindInput,
		Shape: shape,
		Type:  dt,
	})
}

// AddUnary creates an elementwise unary operation (e.g. "relu", "exp", "neg", "sqrt").
func (g *Graph) AddUnary(op string, in *Node) *Node {
	return g.addNode(&Node{
		Name:   fmt.Sprintf("%s_%d", op, len(g.nodes)),
		Kind:   KindUnary,
		Op:     op,
		Inputs: []*Node{in},
		Shape:  in.Shape,
		Type:   in.Type,
	})
}

// AddBinary creates an elementwise binary operation (e.g. "add", "sub", "mul", "div").
func (g *Graph) AddBinary(op string, a, b *Node) *Node {
	return g.addNode(&Node{
		Name:   fmt.Sprintf("%s_%d", op, len(g.nodes)),
		Kind:   KindBinary,
		Op:     op,
		Inputs: []*Node{a, b},
		Shape:  a.Shape,
		Type:   a.Type,
	})
}

// AddMatMul creates a matrix multiplication node (C = A x B).
func (g *Graph) AddMatMul(a, b *Node) *Node {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("AddMatMul: inputs must be 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("AddMatMul: shape mismatch (%v x %v)", a.Shape, b.Shape))
	}
	outShape := []int{a.Shape[0], b.Shape[1]}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("matmul_%d", len(g.nodes)),
		Kind:   KindMatMul,
		Op:     "matmul",
		Inputs: []*Node{a, b},
		Shape:  outShape,
		Type:   a.Type,
	})
}

// AddReduce creates a reduction operation (e.g. "sum", "max", "mean") along specified axis.
func (g *Graph) AddReduce(op string, in *Node, axis int) *Node {
	outShape := make([]int, 0, len(in.Shape))
	for i, d := range in.Shape {
		if i != axis {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("%s_reduce_%d", op, len(g.nodes)),
		Kind:   KindReduce,
		Op:     op,
		Inputs: []*Node{in},
		Shape:  outShape,
		Type:   in.Type,
	})
}
