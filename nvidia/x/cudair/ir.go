package cudair

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
	return append([]*Node(nil), g.nodes...)
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
		Shape: append([]int(nil), shape...),
		Type:  dt,
	})
}

// AddUnary creates an elementwise unary operation (e.g. "relu", "exp", "neg", "sqrt").
func (g *Graph) AddUnary(op string, in *Node) *Node {
	var shape []int
	var dt DataType
	if in != nil {
		shape = append(shape, in.Shape...)
		dt = in.Type
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("unary_%d", len(g.nodes)),
		Kind:   KindUnary,
		Op:     op,
		Inputs: []*Node{in},
		Shape:  shape,
		Type:   dt,
	})
}

// AddBinary creates an elementwise binary operation (e.g. "add", "sub", "mul", "div").
func (g *Graph) AddBinary(op string, a, b *Node) *Node {
	var shape []int
	var dt DataType
	if a != nil {
		shape = append(shape, a.Shape...)
		dt = a.Type
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("binary_%d", len(g.nodes)),
		Kind:   KindBinary,
		Op:     op,
		Inputs: []*Node{a, b},
		Shape:  shape,
		Type:   dt,
	})
}

// AddMatMul creates a matrix multiplication node (C = A x B).
func (g *Graph) AddMatMul(a, b *Node) *Node {
	var shape []int
	var dt DataType
	if a != nil {
		dt = a.Type
	}
	if a != nil && b != nil && len(a.Shape) == 2 && len(b.Shape) == 2 {
		shape = []int{a.Shape[0], b.Shape[1]}
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("matmul_%d", len(g.nodes)),
		Kind:   KindMatMul,
		Op:     "matmul",
		Inputs: []*Node{a, b},
		Shape:  shape,
		Type:   dt,
	})
}

// AddReduce creates a reduction operation (e.g. "sum", "max", "mean") along specified axis.
func (g *Graph) AddReduce(op string, in *Node, axis int) *Node {
	var shape []int
	var dt DataType
	if in != nil {
		shape = in.Shape
		dt = in.Type
	}
	outShape := make([]int, 0, len(shape))
	for i, d := range shape {
		if i != axis {
			outShape = append(outShape, d)
		}
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("reduce_%d", len(g.nodes)),
		Kind:   KindReduce,
		Op:     op,
		Inputs: []*Node{in},
		Shape:  outShape,
		Type:   dt,
	})
}

// Validate reports malformed shapes, types, topology, and operator contracts.
func (g *Graph) Validate() error {
	if g == nil {
		return fmt.Errorf("nil graph")
	}
	seen := make(map[*Node]bool)
	names := make(map[string]bool)
	for i, n := range g.nodes {
		if n == nil {
			return fmt.Errorf("node %d: nil node", i)
		}
		if n.ID != i {
			return fmt.Errorf("node %d: id is %d", i, n.ID)
		}
		if !identifier(n.Name) {
			return fmt.Errorf("node %d: invalid name %q", i, n.Name)
		}
		if names[n.Name] {
			return fmt.Errorf("node %d: duplicate name %q", i, n.Name)
		}
		names[n.Name] = true
		elemSize := n.Type.SizeBytes()
		if elemSize == 0 {
			return fmt.Errorf("node %d: invalid data type %d", i, n.Type)
		}
		elements := 1
		for _, dim := range n.Shape {
			if dim <= 0 {
				return fmt.Errorf("node %d: invalid dimension %d", i, dim)
			}
			if elements > math.MaxInt/dim {
				return fmt.Errorf("node %d: shape size overflows int", i)
			}
			elements *= dim
		}
		if elements > math.MaxInt/elemSize {
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

func identifier(s string) bool {
	if s == "" || !letter(s[0]) {
		return false
	}
	for i := 1; i < len(s); i++ {
		if !letter(s[i]) && (s[i] < '0' || s[i] > '9') {
			return false
		}
	}
	return true
}

func letter(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

func validateOperator(n *Node) error {
	same := func(a, b *Node) bool {
		if a == nil || b == nil || a.Type != b.Type || len(a.Shape) != len(b.Shape) {
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
		if len(n.Inputs) != 2 || !same(n, n.Inputs[0]) || !same(n, n.Inputs[1]) {
			return fmt.Errorf("binary operand shape or type mismatch")
		}
	case KindMatMul:
		if len(n.Inputs) != 2 || len(n.Inputs[0].Shape) != 2 || len(n.Inputs[1].Shape) != 2 {
			return fmt.Errorf("matmul requires rank-2 operands")
		}
		a, b := n.Inputs[0], n.Inputs[1]
		if a.Type != b.Type || n.Type != a.Type || a.Shape[1] != b.Shape[0] || len(n.Shape) != 2 || n.Shape[0] != a.Shape[0] || n.Shape[1] != b.Shape[1] {
			return fmt.Errorf("matmul shape or type mismatch")
		}
	case KindReduce:
		if len(n.Inputs) != 1 || n.Type != n.Inputs[0].Type || len(n.Inputs[0].Shape) == 0 || len(n.Shape) != len(n.Inputs[0].Shape)-1 && !(len(n.Inputs[0].Shape) == 1 && len(n.Shape) == 1 && n.Shape[0] == 1) {
			return fmt.Errorf("reduce operand or type mismatch")
		}
	default:
		return fmt.Errorf("invalid kind %d", n.Kind)
	}
	return nil
}
