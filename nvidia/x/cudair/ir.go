package cudair

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudashape"
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
	Int8
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
	case Uint8, Int8:
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
	case Int8:
		return "int8"
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
	KindReshape
	KindTranspose
	KindSlice
	KindConcat
)

func (k NodeKind) String() string {
	switch k {
	case KindInput:
		return "input"
	case KindConstant:
		return "constant"
	case KindUnary:
		return "unary"
	case KindBinary:
		return "binary"
	case KindMatMul:
		return "matmul"
	case KindReduce:
		return "reduce"
	case KindReshape:
		return "reshape"
	case KindTranspose:
		return "transpose"
	case KindSlice:
		return "slice"
	case KindConcat:
		return "concat"
	default:
		return "unknown"
	}
}

// StorageEffect indicates the memory effect of an operation on its buffers.
type StorageEffect int

const (
	// EffectReadOnly denotes an operation that reads inputs and produces a distinct output buffer.
	EffectReadOnly StorageEffect = iota
	// EffectMutating denotes an operation that mutates its target buffer in-place.
	EffectMutating
	// EffectInPlaceAlias denotes an operation that aliases an existing input buffer without allocation.
	EffectInPlaceAlias
)

func (e StorageEffect) String() string {
	switch e {
	case EffectReadOnly:
		return "read-only"
	case EffectMutating:
		return "mutating"
	case EffectInPlaceAlias:
		return "in-place-alias"
	default:
		return "unknown"
	}
}

// SourceLocation captures host Go source attribution for an IR node.
//
// File, Line and Function name the statement that built the node. Callers
// holds the frames above that statement, nearest first, so a node built
// inside a helper can also be charged to the code that called the helper.
// Callers is capped by whoever records the location and may be empty.
type SourceLocation struct {
	File     string
	Line     int
	Function string
	Callers  []Frame
}

// A Frame is one caller of the statement a SourceLocation names.
type Frame struct {
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

	// Layout and memory metadata
	Layout       cudashape.Layout
	Strides      []int
	Effect       StorageEffect
	StorageAlias *Node

	// Operator specific parameters
	ConstValues []float32
	Axes        []int
	Perm        []int
	SliceStarts []int
	SliceEnds   []int
	SliceSteps  []int
	Axis        int

	graph            *Graph
	snapshotOriginal *Node
}

// Graph returns the graph to which this node belongs.
func (n *Node) Graph() *Graph {
	return n.graph
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
	nodes          []*Node
	outputs        []*Node
	snapshotSource *Graph
	err            error
}

// NewGraph creates an empty IR Graph.
func NewGraph() *Graph {
	return &Graph{}
}

// Nodes returns all nodes in the graph in topological insertion order.
func (g *Graph) Nodes() []*Node {
	return g.nodes
}

// Outputs returns the explicit graph outputs, or if none were explicitly set,
// all terminal non-input nodes.
func (g *Graph) Outputs() []*Node {
	if len(g.outputs) > 0 {
		return append([]*Node(nil), g.outputs...)
	}
	// Fallback to terminal nodes
	consumerCount := make(map[*Node]int)
	for _, n := range g.nodes {
		for _, inp := range n.Inputs {
			consumerCount[inp]++
		}
	}
	var outs []*Node
	for _, n := range g.nodes {
		if n.Kind != KindInput && consumerCount[n] == 0 {
			outs = append(outs, n)
		}
	}
	return outs
}

// SetOutputs configures explicit graph outputs. Each output node must belong to the graph.
func (g *Graph) SetOutputs(nodes ...*Node) error {
	for _, n := range nodes {
		if n == nil || n.graph != g {
			err := fmt.Errorf("%w: node does not belong to this graph", ErrInvalidOutput)
			g.recordErr(err)
			return err
		}
	}
	g.outputs = append([]*Node(nil), nodes...)
	return nil
}

// AddOutput adds an explicit output node to the graph.
func (g *Graph) AddOutput(n *Node) error {
	if n == nil || n.graph != g {
		err := fmt.Errorf("%w: node does not belong to this graph", ErrInvalidOutput)
		g.recordErr(err)
		return err
	}
	g.outputs = append(g.outputs, n)
	return nil
}

// Err returns any sticky error recorded during graph construction.
func (g *Graph) Err() error {
	return g.err
}

func (g *Graph) recordErr(err error) {
	if g.err == nil {
		g.err = err
	}
}

func (g *Graph) addNode(n *Node) *Node {
	n.ID = len(g.nodes)
	n.graph = g
	if len(n.Strides) == 0 && len(n.Shape) > 0 {
		if strides, err := cudashape.RowMajorStrides(n.Shape); err == nil {
			n.Strides = strides
			n.Layout = cudashape.LayoutRowMajor
		}
	}
	g.nodes = append(g.nodes, n)
	return n
}

// AddInput creates an external input node.
func (g *Graph) AddInput(name string, shape []int, dt DataType) *Node {
	n, err := g.AddInputChecked(name, shape, dt)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddInputChecked creates an external input node with explicit validation.
func (g *Graph) AddInputChecked(name string, shape []int, dt DataType) (*Node, error) {
	if _, err := cudashape.New(shape...); err != nil {
		return nil, fmt.Errorf("AddInput %q: %w", name, err)
	}
	return g.addNode(&Node{
		Name:   name,
		Kind:   KindInput,
		Op:     "input",
		Shape:  append([]int(nil), shape...),
		Type:   dt,
		Effect: EffectReadOnly,
	}), nil
}

// AddConstant creates a constant tensor node.
func (g *Graph) AddConstant(name string, shape []int, dt DataType, values []float32) *Node {
	n, err := g.AddConstantChecked(name, shape, dt, values)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddConstantChecked creates a constant tensor node with explicit validation.
func (g *Graph) AddConstantChecked(name string, shape []int, dt DataType, values []float32) (*Node, error) {
	s, err := cudashape.New(shape...)
	if err != nil {
		return nil, fmt.Errorf("AddConstant %q: %w", name, err)
	}
	numElems, err := s.NumElements()
	if err != nil {
		return nil, err
	}
	if len(values) > 0 && int64(len(values)) != numElems {
		return nil, fmt.Errorf("%w: constant %q values len %d != shape elements %d", ErrShapeMismatch, name, len(values), numElems)
	}
	return g.addNode(&Node{
		Name:        name,
		Kind:        KindConstant,
		Op:          "constant",
		Shape:       append([]int(nil), shape...),
		Type:        dt,
		ConstValues: append([]float32(nil), values...),
		Effect:      EffectReadOnly,
	}), nil
}

// AddUnary creates an elementwise unary operation (e.g. "relu", "gelu", "exp", "log", "sqrt", "neg").
func (g *Graph) AddUnary(op string, in *Node) *Node {
	n, err := g.AddUnaryChecked(op, in)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddUnaryChecked creates an elementwise unary operation with explicit validation.
func (g *Graph) AddUnaryChecked(op string, in *Node) (*Node, error) {
	if in == nil {
		return nil, errors.New("cudair: nil input to AddUnary")
	}
	if in.graph != g {
		return nil, fmt.Errorf("%w: input node belongs to another graph", ErrMixedGraphOwnership)
	}
	normOp := strings.ToLower(op)
	return g.addNode(&Node{
		Name:   fmt.Sprintf("%s_%d", normOp, len(g.nodes)),
		Kind:   KindUnary,
		Op:     normOp,
		Inputs: []*Node{in},
		Shape:  append([]int(nil), in.Shape...),
		Type:   in.Type,
		Effect: EffectReadOnly,
	}), nil
}

// AddBinary creates an elementwise binary operation (e.g. "add", "sub", "mul", "div").
func (g *Graph) AddBinary(op string, a, b *Node) *Node {
	n, err := g.AddBinaryChecked(op, a, b)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddBinaryChecked creates an elementwise binary operation with equal input shapes
// or a scalar (one-element) operand. Nonscalar broadcasting is not supported.
func (g *Graph) AddBinaryChecked(op string, a, b *Node) (*Node, error) {
	if a == nil || b == nil {
		return nil, errors.New("cudair: nil input to AddBinary")
	}
	if a.graph != g || b.graph != g {
		return nil, fmt.Errorf("%w: input nodes belong to different graphs", ErrMixedGraphOwnership)
	}
	if a.Type != b.Type {
		return nil, fmt.Errorf("%w: binary op %s has type mismatch (%v vs %v)", ErrTypeMismatch, op, a.Type, b.Type)
	}
	shape, err := binaryShape(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	normOp := strings.ToLower(op)
	if normOp == "multiply" {
		normOp = "mul"
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("%s_%d", normOp, len(g.nodes)),
		Kind:   KindBinary,
		Op:     normOp,
		Inputs: []*Node{a, b},
		Shape:  shape,
		Type:   a.Type,
		Effect: EffectReadOnly,
	}), nil
}

// AddMatMul creates a 2D matrix multiplication node (C = A x B).
func (g *Graph) AddMatMul(a, b *Node) *Node {
	n, err := g.AddMatMulChecked(a, b)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddMatMulChecked creates a matrix multiplication node with explicit validation.
func (g *Graph) AddMatMulChecked(a, b *Node) (*Node, error) {
	if a == nil || b == nil {
		return nil, errors.New("cudair: nil input to AddMatMul")
	}
	if a.graph != g || b.graph != g {
		return nil, fmt.Errorf("%w: inputs to AddMatMul belong to different graphs", ErrMixedGraphOwnership)
	}
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return nil, fmt.Errorf("%w: AddMatMul inputs must be 2D tensors, got %dD and %dD", ErrShapeMismatch, len(a.Shape), len(b.Shape))
	}
	if a.Shape[1] != b.Shape[0] {
		return nil, fmt.Errorf("%w: AddMatMul shape mismatch (%v x %v)", ErrShapeMismatch, a.Shape, b.Shape)
	}
	if a.Type != b.Type {
		return nil, fmt.Errorf("%w: AddMatMul type mismatch (%v vs %v)", ErrTypeMismatch, a.Type, b.Type)
	}
	outShape := []int{a.Shape[0], b.Shape[1]}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("matmul_%d", len(g.nodes)),
		Kind:   KindMatMul,
		Op:     "matmul",
		Inputs: []*Node{a, b},
		Shape:  outShape,
		Type:   a.Type,
		Effect: EffectReadOnly,
	}), nil
}

// AddReduce creates a reduction operation (e.g. "sum", "max") along specified axis.
func (g *Graph) AddReduce(op string, in *Node, axis int) *Node {
	n, err := g.AddReduceChecked(op, in, axis)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddReduceChecked creates a reduction operation with explicit validation.
func (g *Graph) AddReduceChecked(op string, in *Node, axis int) (*Node, error) {
	if in == nil {
		return nil, errors.New("cudair: nil input to AddReduce")
	}
	if in.graph != g {
		return nil, fmt.Errorf("%w: input to AddReduce belongs to another graph", ErrMixedGraphOwnership)
	}
	if axis < 0 || axis >= len(in.Shape) {
		return nil, fmt.Errorf("%w: axis %d out of bounds for shape %v", ErrInvalidAxis, axis, in.Shape)
	}
	normOp := strings.ToLower(op)
	if normOp == "reducesum" {
		normOp = "sum"
	} else if normOp == "reducemax" {
		normOp = "max"
	}
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
		Name:   fmt.Sprintf("%s_reduce_%d", normOp, len(g.nodes)),
		Kind:   KindReduce,
		Op:     normOp,
		Inputs: []*Node{in},
		Shape:  outShape,
		Type:   in.Type,
		Axes:   []int{axis},
		Effect: EffectReadOnly,
	}), nil
}

// AddReshape creates a tensor reshape operation.
func (g *Graph) AddReshape(in *Node, shape []int) *Node {
	n, err := g.AddReshapeChecked(in, shape)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddReshapeChecked creates a tensor reshape operation with explicit validation.
func (g *Graph) AddReshapeChecked(in *Node, shape []int) (*Node, error) {
	if in == nil {
		return nil, errors.New("cudair: nil input to AddReshape")
	}
	if in.graph != g {
		return nil, fmt.Errorf("%w: input to AddReshape belongs to another graph", ErrMixedGraphOwnership)
	}
	s, err := cudashape.New(shape...)
	if err != nil {
		return nil, fmt.Errorf("AddReshape: %w", err)
	}
	newElems, err := s.NumElements()
	if err != nil {
		return nil, err
	}
	inElems := int64(in.NumElements())
	if newElems != inElems {
		return nil, fmt.Errorf("%w: reshape from %v (%d elements) to %v (%d elements)",
			ErrShapeMismatch, in.Shape, inElems, shape, newElems)
	}
	return g.addNode(&Node{
		Name:         fmt.Sprintf("reshape_%d", len(g.nodes)),
		Kind:         KindReshape,
		Op:           "reshape",
		Inputs:       []*Node{in},
		Shape:        append([]int(nil), shape...),
		Type:         in.Type,
		Effect:       EffectInPlaceAlias,
		StorageAlias: in,
	}), nil
}

// AddTranspose creates a tensor transposition with axis permutation.
func (g *Graph) AddTranspose(in *Node, perm []int) *Node {
	n, err := g.AddTransposeChecked(in, perm)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddTransposeChecked creates a tensor transposition with explicit validation.
func (g *Graph) AddTransposeChecked(in *Node, perm []int) (*Node, error) {
	if in == nil {
		return nil, errors.New("cudair: nil input to AddTranspose")
	}
	if in.graph != g {
		return nil, fmt.Errorf("%w: input to AddTranspose belongs to another graph", ErrMixedGraphOwnership)
	}
	rank := len(in.Shape)
	if len(perm) != rank {
		return nil, fmt.Errorf("%w: permutation len %d != tensor rank %d", ErrInvalidAxis, len(perm), rank)
	}
	seen := make([]bool, rank)
	outShape := make([]int, rank)
	for i, p := range perm {
		if p < 0 || p >= rank {
			return nil, fmt.Errorf("%w: perm index %d out of bounds for rank %d", ErrInvalidAxis, p, rank)
		}
		if seen[p] {
			return nil, fmt.Errorf("%w: duplicate axis %d in permutation", ErrInvalidAxis, p)
		}
		seen[p] = true
		outShape[i] = in.Shape[p]
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("transpose_%d", len(g.nodes)),
		Kind:   KindTranspose,
		Op:     "transpose",
		Inputs: []*Node{in},
		Shape:  outShape,
		Type:   in.Type,
		Perm:   append([]int(nil), perm...),
		Effect: EffectReadOnly,
	}), nil
}

// AddSlice creates a tensor slice operation.
func (g *Graph) AddSlice(in *Node, starts, ends, steps []int) *Node {
	n, err := g.AddSliceChecked(in, starts, ends, steps)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddSliceChecked creates a tensor slice operation with explicit validation.
func (g *Graph) AddSliceChecked(in *Node, starts, ends, steps []int) (*Node, error) {
	if in == nil {
		return nil, errors.New("cudair: nil input to AddSlice")
	}
	if in.graph != g {
		return nil, fmt.Errorf("%w: input to AddSlice belongs to another graph", ErrMixedGraphOwnership)
	}
	rank := len(in.Shape)
	if len(starts) != rank || len(ends) != rank || len(steps) != rank {
		return nil, fmt.Errorf("%w: slice parameters length mismatch with rank %d", ErrInvalidAxis, rank)
	}
	outShape := make([]int, rank)
	for i := 0; i < rank; i++ {
		start, end, step := starts[i], ends[i], steps[i]
		if step <= 0 {
			return nil, fmt.Errorf("%w: non-positive step %d at axis %d", ErrInvalidAxis, step, i)
		}
		if start < 0 || start > in.Shape[i] || end < start || end > in.Shape[i] {
			return nil, fmt.Errorf("%w: slice range [%d, %d] invalid for extent %d at axis %d",
				ErrInvalidAxis, start, end, in.Shape[i], i)
		}
		outShape[i] = (end - start + step - 1) / step
	}
	return g.addNode(&Node{
		Name:        fmt.Sprintf("slice_%d", len(g.nodes)),
		Kind:        KindSlice,
		Op:          "slice",
		Inputs:      []*Node{in},
		Shape:       outShape,
		Type:        in.Type,
		SliceStarts: append([]int(nil), starts...),
		SliceEnds:   append([]int(nil), ends...),
		SliceSteps:  append([]int(nil), steps...),
		Effect:      EffectReadOnly,
	}), nil
}

// AddConcat creates a tensor concatenation along axis.
func (g *Graph) AddConcat(inputs []*Node, axis int) *Node {
	n, err := g.AddConcatChecked(inputs, axis)
	if err != nil {
		g.recordErr(err)
	}
	return n
}

// AddConcatChecked creates a tensor concatenation with explicit validation.
func (g *Graph) AddConcatChecked(inputs []*Node, axis int) (*Node, error) {
	if len(inputs) == 0 {
		return nil, errors.New("cudair: empty inputs to AddConcat")
	}
	first := inputs[0]
	if first == nil {
		return nil, errors.New("cudair: nil input to AddConcat")
	}
	if first.graph != g {
		return nil, fmt.Errorf("%w: input to AddConcat belongs to another graph", ErrMixedGraphOwnership)
	}
	rank := len(first.Shape)
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("%w: concat axis %d out of bounds for rank %d", ErrInvalidAxis, axis, rank)
	}
	outShape := append([]int(nil), first.Shape...)
	outShape[axis] = 0
	for idx, inp := range inputs {
		if inp == nil {
			return nil, fmt.Errorf("cudair: nil input at index %d to AddConcat", idx)
		}
		if inp.graph != g {
			return nil, fmt.Errorf("%w: input %s belongs to another graph", ErrMixedGraphOwnership, inp.Name)
		}
		if len(inp.Shape) != rank {
			return nil, fmt.Errorf("%w: input %s rank %d != first rank %d", ErrShapeMismatch, inp.Name, len(inp.Shape), rank)
		}
		if inp.Type != first.Type {
			return nil, fmt.Errorf("%w: input %s type %v != first type %v", ErrTypeMismatch, inp.Name, inp.Type, first.Type)
		}
		for d := 0; d < rank; d++ {
			if d != axis && inp.Shape[d] != first.Shape[d] {
				return nil, fmt.Errorf("%w: input %s shape %v incompatible with %v along non-concat axis %d",
					ErrShapeMismatch, inp.Name, inp.Shape, first.Shape, d)
			}
		}
		outShape[axis] += inp.Shape[axis]
	}
	return g.addNode(&Node{
		Name:   fmt.Sprintf("concat_%d", len(g.nodes)),
		Kind:   KindConcat,
		Op:     "concat",
		Inputs: append([]*Node(nil), inputs...),
		Shape:  outShape,
		Type:   first.Type,
		Axis:   axis,
		Effect: EffectReadOnly,
	}), nil
}
