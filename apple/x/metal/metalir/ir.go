package metalir

import (
	"fmt"
	"math"

	"github.com/tmc/lib/apple/x/metal/metalshape"
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

// ScalarAttributes contains scalar parameters that affect an operation's
// semantics. Presence flags distinguish an omitted parameter from its zero
// value; constructors validate parameters before setting these fields.
type ScalarAttributes struct {
	HasEpsilon bool
	Epsilon    float32
	HasTheta   bool
	Theta      float32
}

// SourceLocation captures host Go source attribution for an IR node.
type SourceLocation struct {
	File     string
	Line     int
	Function string
}

// Node represents an operation or tensor in the IR graph. Shape contains
// concrete dimensions when all dimensions are known. ShapeExpr is authoritative
// when present and may contain unresolved symbols.
type Node struct {
	ID              int
	Name            string
	Kind            NodeKind
	Op              string
	Inputs          []*Node
	Shape           []int
	ShapeExpr       []metalshape.Expr
	Type            DataType
	Attributes      ScalarAttributes
	SourceLoc       SourceLocation
	ValidationError error
	shapeConcrete   bool
}

// NumElements returns the element count for a concrete shape. It returns zero
// when the shape is symbolic, invalid, or overflows int. Use
// NumElementsChecked to distinguish these cases and resolve symbols.
func (n *Node) NumElements() int {
	count, err := n.NumElementsChecked(nil)
	if err != nil {
		return 0
	}
	return count
}

// NumElementsChecked resolves ShapeExpr with values and returns its element
// count. A nil values map is sufficient for concrete shapes. Scalar shapes
// contain zero dimensions and have one element.
func (n *Node) NumElementsChecked(values map[string]int64) (int, error) {
	if n == nil {
		return 0, fmt.Errorf("metalir: nil node")
	}
	if n.ValidationError != nil {
		return 0, n.ValidationError
	}
	dimensions, err := n.ResolveShape(values)
	if err != nil {
		return 0, err
	}
	dims := make([]int64, len(dimensions))
	for i, dim := range dimensions {
		dims[i] = int64(dim)
	}
	count := int64(1)
	for i, dim := range dims {
		if dim < 0 || (dim != 0 && count > math.MaxInt64/dim) {
			return 0, fmt.Errorf("metalir: element count overflows at dimension %d", i)
		}
		count *= dim
	}
	maxInt := int64(^uint(0) >> 1)
	if count > maxInt {
		return 0, fmt.Errorf("metalir: element count does not fit int")
	}
	return int(count), nil
}

// ResolveShape evaluates the node's dimensions using exact symbol values.
// It rejects unresolved symbols, invalid nodes, and disagreement between the
// symbolic and concrete representations.
func (n *Node) ResolveShape(values map[string]int64) ([]int, error) {
	if n == nil {
		return nil, fmt.Errorf("metalir: nil node")
	}
	if n.ValidationError != nil {
		return nil, n.ValidationError
	}
	if n.ShapeExpr == nil {
		shape := append([]int(nil), n.Shape...)
		for i, dim := range shape {
			if dim < 0 {
				return nil, fmt.Errorf("metalir: dimension %d is negative", i)
			}
		}
		return shape, nil
	}
	shape := make([]int, len(n.ShapeExpr))
	maxInt := int64(^uint(0) >> 1)
	for i, expr := range n.ShapeExpr {
		dim, err := expr.Eval(values)
		if err != nil {
			return nil, fmt.Errorf("metalir: resolve dimension %d: %w", i, err)
		}
		if dim > maxInt {
			return nil, fmt.Errorf("metalir: dimension %d does not fit int", i)
		}
		shape[i] = int(dim)
	}
	if n.shapeConcrete {
		if len(shape) != len(n.Shape) {
			return nil, fmt.Errorf("metalir: concrete and symbolic shape ranks differ")
		}
		for i := range shape {
			if shape[i] != n.Shape[i] {
				return nil, fmt.Errorf("metalir: concrete and symbolic dimension %d differ", i)
			}
		}
	}
	return shape, nil
}

// SizeBytes returns the total buffer size for a concrete shape, or zero when
// the shape cannot be resolved or multiplication overflows. Use
// SizeBytesChecked for error details.
func (n *Node) SizeBytes() int {
	size, err := n.SizeBytesChecked(nil)
	if err != nil {
		return 0
	}
	return size
}

// SizeBytesChecked resolves the shape and checks the byte-size multiplication.
func (n *Node) SizeBytesChecked(values map[string]int64) (int, error) {
	if n == nil {
		return 0, fmt.Errorf("metalir: nil node")
	}
	if n.Type < Float32 || n.Type > Uint8 {
		return 0, fmt.Errorf("metalir: unknown data type %d", n.Type)
	}
	elements, err := n.NumElementsChecked(values)
	if err != nil {
		return 0, err
	}
	unit := n.Type.SizeBytes()
	maxInt := int(^uint(0) >> 1)
	if unit < 0 || (unit != 0 && elements > maxInt/unit) {
		return 0, fmt.Errorf("metalir: byte size overflows int")
	}
	return elements * unit, nil
}

// Graph represents a directed acyclic graph (DAG) of tensor operations.
type Graph struct {
	Nodes []*Node
}

// NewGraph creates an empty symbolic computation Graph.
func NewGraph() *Graph { return &Graph{Nodes: make([]*Node, 0)} }

// AddInput creates an input placeholder node with concrete dimensions.
func (g *Graph) AddInput(name string, shape []int, dtype DataType) *Node {
	copied := append([]int(nil), shape...)
	exprs, err := expressionsFromInts(copied)
	n := &Node{ID: len(g.Nodes), Name: name, Kind: KindInput, Op: "input", Shape: copied, Type: dtype, shapeConcrete: true}
	if err == nil {
		n.ShapeExpr = exprs
	} else {
		n.ValidationError = fmt.Errorf("metalir: invalid input shape: %w", err)
	}
	if dtype < Float32 || dtype > Uint8 {
		n.ValidationError = fmt.Errorf("metalir: unknown data type %d", dtype)
	}
	g.Nodes = append(g.Nodes, n)
	return n
}

// AddInputWithShapeExpr creates an input with symbolic or concrete dimensions.
func (g *Graph) AddInputWithShapeExpr(name string, shape []metalshape.Expr, dtype DataType) (*Node, error) {
	if g == nil {
		return nil, fmt.Errorf("metalir: nil graph")
	}

	if dtype < Float32 || dtype > Uint8 {
		return nil, fmt.Errorf("metalir: unknown data type %d", dtype)
	}
	copied := append([]metalshape.Expr(nil), shape...)
	concrete, ok := concreteShape(copied)
	n := &Node{ID: len(g.Nodes), Name: name, Kind: KindInput, Op: "input", ShapeExpr: copied, Type: dtype, shapeConcrete: ok}
	if ok {
		n.Shape = concrete
	}
	g.Nodes = append(g.Nodes, n)
	return n, nil
}

// AddUnary creates a unary operation node and carries its shape expressions.
func (g *Graph) AddUnary(op string, in *Node) *Node {
	if err := validateInputNode(in); err != nil {
		return g.addInvalid(op, KindUnary, []*Node{in}, fmt.Errorf("metalir: unary input: %w", err))
	}
	n := &Node{ID: len(g.Nodes), Name: fmt.Sprintf("%s_%d", op, len(g.Nodes)), Kind: KindUnary, Op: op, Shape: append([]int(nil), in.Shape...), ShapeExpr: append([]metalshape.Expr(nil), in.ShapeExpr...), Type: in.Type, shapeConcrete: in.shapeConcrete, Inputs: []*Node{in}}
	g.Nodes = append(g.Nodes, n)
	return n
}

// AddBinary creates an elementwise operation. Use AddBinaryChecked to receive
// validation failures directly; invalid legacy constructions carry
// ValidationError and cannot be sized.
func (g *Graph) AddBinary(op string, a, b *Node) *Node {
	n, err := g.AddBinaryChecked(op, a, b)
	if err == nil {
		return n
	}
	return g.addInvalid(op, KindBinary, []*Node{a, b}, err)
}

// AddBinaryChecked creates an elementwise operation with equal input shapes.
func (g *Graph) AddBinaryChecked(op string, a, b *Node) (*Node, error) {
	if g == nil {
		return nil, fmt.Errorf("metalir: nil graph")
	}

	if err := validateInputPair(a, b); err != nil {
		return nil, fmt.Errorf("metalir: binary inputs: %w", err)
	}
	if !sameShapeExpr(a, b) {
		return nil, fmt.Errorf("metalir: %s inputs have different shapes", op)
	}
	n := &Node{ID: len(g.Nodes), Name: fmt.Sprintf("%s_%d", op, len(g.Nodes)), Kind: KindBinary, Op: op, Shape: append([]int(nil), a.Shape...), ShapeExpr: append([]metalshape.Expr(nil), a.ShapeExpr...), Type: a.Type, shapeConcrete: a.shapeConcrete, Inputs: []*Node{a, b}}
	g.Nodes = append(g.Nodes, n)
	return n, nil
}

// AddRMSNorm creates an RMSNorm node with its epsilon parameter. Invalid
// construction returns a node carrying ValidationError, matching AddBinary's
// fail-closed legacy behavior.
func (g *Graph) AddRMSNorm(input, weight *Node, epsilon float32) *Node {
	n, err := g.AddRMSNormChecked(input, weight, epsilon)
	if err == nil {
		return n
	}
	return g.addInvalid("rmsnorm", KindBinary, []*Node{input, weight}, err)
}

// AddRMSNormChecked creates an RMSNorm node after validating its scalar
// parameter and input shapes.
func (g *Graph) AddRMSNormChecked(input, weight *Node, epsilon float32) (*Node, error) {
	if !positiveFinite(float64(epsilon)) {
		return nil, fmt.Errorf("metalir: rmsnorm epsilon must be finite and positive")
	}
	n, err := g.AddBinaryChecked("rmsnorm", input, weight)
	if err != nil {
		return nil, err
	}
	n.Attributes = ScalarAttributes{HasEpsilon: true, Epsilon: epsilon}
	return n, nil
}

// AddRoPE creates a RoPE node with its theta parameter. Invalid construction
// returns a node carrying ValidationError.
func (g *Graph) AddRoPE(input, position *Node, theta float32) *Node {
	n, err := g.AddRoPEChecked(input, position, theta)
	if err == nil {
		return n
	}
	return g.addInvalid("rope", KindBinary, []*Node{input, position}, err)
}

// AddRoPEChecked creates a RoPE node after validating its scalar parameter
// and data input shape. Kernels pair the first and second halves of the final
// dimension; generated lowering requires that dimension to be even.
func (g *Graph) AddRoPEChecked(input, position *Node, theta float32) (*Node, error) {
	if g == nil {
		return nil, fmt.Errorf("metalir: nil graph")
	}

	if !positiveFinite(float64(theta)) {
		return nil, fmt.Errorf("metalir: rope theta must be finite and positive")
	}
	if err := validateInputNode(input); err != nil {
		return nil, fmt.Errorf("metalir: rope input: %w", err)
	}
	if err := validateInputNode(position); err != nil {
		return nil, fmt.Errorf("metalir: rope position: %w", err)
	}
	n := &Node{ID: len(g.Nodes), Name: fmt.Sprintf("rope_%d", len(g.Nodes)), Kind: KindBinary, Op: "rope", Shape: append([]int(nil), input.Shape...), ShapeExpr: append([]metalshape.Expr(nil), input.ShapeExpr...), Type: input.Type, Inputs: []*Node{input, position}, shapeConcrete: input.shapeConcrete, Attributes: ScalarAttributes{HasTheta: true, Theta: theta}}
	g.Nodes = append(g.Nodes, n)
	return n, nil
}

func positiveFinite(value float64) bool {
	return value > 0 && !math.IsInf(value, 0) && !math.IsNaN(value)
}

// AddMatMul creates a rank-2 matrix multiplication. Invalid legacy
// constructions carry ValidationError; use AddMatMulChecked for the error.
func (g *Graph) AddMatMul(a, b *Node) *Node {
	n, err := g.AddMatMulChecked(a, b)
	if err == nil {
		return n
	}
	return g.addInvalid("matmul", KindMatMul, []*Node{a, b}, err)
}

// AddMatMulChecked creates a rank-2 matrix multiplication whose inner
// dimensions are structurally equal.
func (g *Graph) AddMatMulChecked(a, b *Node) (*Node, error) {
	if g == nil {
		return nil, fmt.Errorf("metalir: nil graph")
	}

	if err := validateInputPair(a, b); err != nil {
		return nil, fmt.Errorf("metalir: matmul inputs: %w", err)
	}
	aShape, aOK := nodeShapeExpr(a)
	bShape, bOK := nodeShapeExpr(b)
	if !aOK || !bOK || len(aShape) != 2 || len(bShape) != 2 {
		return nil, fmt.Errorf("metalir: matmul inputs must have rank 2")
	}
	if !aShape[1].Equal(bShape[0]) {
		return nil, fmt.Errorf("metalir: matmul inner dimensions differ")
	}
	shape := []metalshape.Expr{aShape[0], bShape[1]}
	concrete, ok := concreteShape(shape)
	n := &Node{ID: len(g.Nodes), Name: fmt.Sprintf("matmul_%d", len(g.Nodes)), Kind: KindMatMul, Op: "matmul", ShapeExpr: shape, Type: a.Type, Inputs: []*Node{a, b}, shapeConcrete: ok}
	if ok {
		n.Shape = concrete
	}
	g.Nodes = append(g.Nodes, n)
	return n, nil
}

func (g *Graph) addInvalid(op string, kind NodeKind, inputs []*Node, err error) *Node {
	n := &Node{ID: len(g.Nodes), Name: fmt.Sprintf("%s_%d", op, len(g.Nodes)), Kind: kind, Op: op, Inputs: inputs, ValidationError: err}
	g.Nodes = append(g.Nodes, n)
	return n
}

func expressionsFromInts(shape []int) ([]metalshape.Expr, error) {
	exprs := make([]metalshape.Expr, len(shape))
	for i, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("metalir: dimension %d is negative", i)
		}
		expr, err := metalshape.DimensionConstant(int64(dim))
		if err != nil {
			return nil, err
		}
		exprs[i] = expr
	}
	return exprs, nil
}

func concreteShape(shape []metalshape.Expr) ([]int, bool) {
	out := make([]int, len(shape))
	maxInt := int64(^uint(0) >> 1)
	for i, expr := range shape {
		dim, err := expr.Eval(nil)
		if err != nil || dim > maxInt {
			return nil, false
		}
		out[i] = int(dim)
	}
	return out, true
}

func nodeShapeExpr(n *Node) ([]metalshape.Expr, bool) {
	if n.ShapeExpr != nil {
		return n.ShapeExpr, true
	}
	exprs, err := expressionsFromInts(n.Shape)
	return exprs, err == nil
}

func sameShapeExpr(a, b *Node) bool {
	as, aOK := nodeShapeExpr(a)
	bs, bOK := nodeShapeExpr(b)
	if aOK && bOK {
		if len(as) != len(bs) {
			return false
		}
		for i := range as {
			if !as[i].Equal(bs[i]) {
				return false
			}
		}
		return true
	}
	if len(a.Shape) != len(b.Shape) {
		return false
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return false
		}
	}
	return true
}

func validateInputNode(n *Node) error {
	if n == nil {
		return fmt.Errorf("nil node")
	}
	if n.ValidationError != nil {
		return n.ValidationError
	}
	if n.Type < Float32 || n.Type > Uint8 {
		return fmt.Errorf("unknown data type %d", n.Type)
	}
	if _, ok := nodeShapeExpr(n); !ok {
		return fmt.Errorf("invalid shape")
	}
	if n.shapeConcrete {
		if _, err := n.ResolveShape(nil); err != nil {
			return err
		}
	}
	return nil
}

func validateInputPair(a, b *Node) error {
	if err := validateInputNode(a); err != nil {
		return fmt.Errorf("left input: %w", err)
	}
	if err := validateInputNode(b); err != nil {
		return fmt.Errorf("right input: %w", err)
	}
	if a.Type != b.Type {
		return fmt.Errorf("input data types differ")
	}
	return nil
}
