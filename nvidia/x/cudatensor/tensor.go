package cudatensor

import (
	"errors"
	"fmt"
	"runtime"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// Tensor represents a deferred symbolic or device-backed tensor node in a computation DAG.
type Tensor struct {
	graph *cudair.Graph
	node  *cudair.Node
	shape []int
	dtype cudair.DataType
	err   error
}

// pkgPrefix is this package's symbol prefix, read from a function in the
// package rather than written as a literal so a move or a rename cannot leave
// a stale string behind.
var pkgPrefix = func() string {
	pc, _, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	name := runtime.FuncForPC(pc).Name() // ".../cudatensor.glob..func1"
	slash := strings.LastIndex(name, "/")
	dot := strings.Index(name[slash+1:], ".")
	if dot < 0 {
		return ""
	}
	return name[:slash+1+dot+1]
}()

// captureSource reports where in the caller's code this graph node was built.
//
// It returns the first frame outside this package rather than a frame at a
// fixed depth. The depth is not uniform: MatMul calls this directly while Add
// reaches it through withBinary, so a fixed skip is right for some operations
// and wrong for others, and being wrong here is silent -- it attributes the
// caller's computation to a line in this file, which reads like an answer.
//
// A location that cannot be determined is returned as the zero value. Callers
// treat that as absent, which is the safe failure: no attribution is a gap a
// reader can see, and a confident wrong one is not.
//
// The frames above the caller, up to maxCallers of them and stopping at the
// runtime, are kept in Callers so a graph built through helpers can be read
// from the helper's callers as well.
func captureSource() cudair.SourceLocation {
	if pkgPrefix == "" {
		return cudair.SourceLocation{}
	}
	var pcs [2 + maxCallers + 16]uintptr
	n := runtime.Callers(2, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	var loc cudair.SourceLocation
	for {
		f, more := frames.Next()
		switch {
		case f.Function == "" || strings.HasPrefix(f.Function, pkgPrefix):
		case strings.HasPrefix(f.Function, "runtime."):
			return loc
		case loc.File == "":
			loc = cudair.SourceLocation{File: f.File, Line: f.Line, Function: f.Function}
		case len(loc.Callers) < maxCallers:
			loc.Callers = append(loc.Callers, cudair.Frame{File: f.File, Line: f.Line, Function: f.Function})
		default:
			return loc
		}
		if !more {
			return loc
		}
	}
}

// maxCallers bounds the frames kept above a captured location.
const maxCallers = 8

// NewInput creates a new input tensor within the given graph.
func NewInput(g *cudair.Graph, name string, shape []int, dtype cudair.DataType) Tensor {
	if g == nil {
		return Tensor{err: errors.New("cudatensor: nil graph")}
	}
	if _, err := cudashape.New(shape...); err != nil {
		return Tensor{graph: g, err: fmt.Errorf("cudatensor.NewInput %q: %w", name, err)}
	}
	n := g.AddInput(name, shape, dtype)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: g,
		node:  n,
		shape: append([]int(nil), shape...),
		dtype: dtype,
	}
}

// NewConstant creates a constant tensor with fixed numerical values.
func NewConstant(g *cudair.Graph, name string, shape []int, dtype cudair.DataType, values []float32) Tensor {
	if g == nil {
		return Tensor{err: errors.New("cudatensor: nil graph")}
	}
	n, err := g.AddConstantChecked(name, shape, dtype, values)
	if err != nil {
		return Tensor{graph: g, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: g,
		node:  n,
		shape: append([]int(nil), shape...),
		dtype: dtype,
	}
}

// Graph returns the underlying computation graph.
func (t Tensor) Graph() *cudair.Graph {
	return t.graph
}

// Node returns the underlying IR node.
func (t Tensor) Node() *cudair.Node {
	return t.node
}

// Shape returns the tensor dimensions.
func (t Tensor) Shape() []int {
	return append([]int(nil), t.shape...)
}

// DataType returns the tensor data type.
func (t Tensor) DataType() cudair.DataType {
	return t.dtype
}

// Err returns the first error encountered during fluent tensor construction.
func (t Tensor) Err() error {
	return t.err
}

// MarkOutput explicitly marks this tensor as a required graph output.
func (t Tensor) MarkOutput() Tensor {
	if t.err != nil {
		return t
	}
	if t.graph != nil && t.node != nil {
		if err := t.graph.AddOutput(t.node); err != nil {
			return Tensor{graph: t.graph, node: t.node, shape: t.shape, dtype: t.dtype, err: err}
		}
	}
	return t
}

func (t Tensor) checkUnary() error {
	if t.err != nil {
		return t.err
	}
	if t.graph == nil || t.node == nil {
		return errors.New("cudatensor: uninitialized tensor")
	}
	return nil
}

func (t Tensor) checkBinary(other Tensor) error {
	if t.err != nil {
		return t.err
	}
	if other.err != nil {
		return other.err
	}
	if t.graph == nil || t.node == nil || other.graph == nil || other.node == nil {
		return errors.New("cudatensor: uninitialized tensor operand")
	}
	if t.graph != other.graph {
		return errors.New("cudatensor: mixed graph operands")
	}
	if t.dtype != other.dtype {
		return fmt.Errorf("cudatensor: type mismatch (%v vs %v)", t.dtype, other.dtype)
	}
	return nil
}

func (t Tensor) withUnary(op string) Tensor {
	if err := t.checkUnary(); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n, err := t.graph.AddUnaryChecked(op, t.node)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), t.shape...),
		dtype: t.dtype,
	}
}

func (t Tensor) withBinary(op string, other Tensor) Tensor {
	if err := t.checkBinary(other); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n, err := t.graph.AddBinaryChecked(op, t.node, other.node)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), n.Shape...),
		dtype: t.dtype,
	}
}

// Add computes elementwise addition t + other.
func (t Tensor) Add(other Tensor) Tensor {
	return t.withBinary("add", other)
}

// Sub computes elementwise subtraction t - other.
func (t Tensor) Sub(other Tensor) Tensor {
	return t.withBinary("sub", other)
}

// Mul computes elementwise multiplication t * other.
func (t Tensor) Mul(other Tensor) Tensor {
	return t.withBinary("mul", other)
}

// Div computes elementwise division t / other.
func (t Tensor) Div(other Tensor) Tensor {
	return t.withBinary("div", other)
}

// Relu computes elementwise rectified linear activation max(0, t).
func (t Tensor) Relu() Tensor {
	return t.withUnary("relu")
}

// Gelu computes elementwise Gaussian Error Linear Unit activation.
func (t Tensor) Gelu() Tensor {
	return t.withUnary("gelu")
}

// Silu computes elementwise sigmoid linear unit (swish activation) x * sigmoid(x).
func (t Tensor) Silu() Tensor {
	return t.withUnary("silu")
}

// Exp computes elementwise exponential exp(t).
func (t Tensor) Exp() Tensor {
	return t.withUnary("exp")
}

// Log computes elementwise natural logarithm log(t).
func (t Tensor) Log() Tensor {
	return t.withUnary("log")
}

// Sqrt computes elementwise square root sqrt(t).
func (t Tensor) Sqrt() Tensor {
	return t.withUnary("sqrt")
}

// Neg computes elementwise negation -t.
func (t Tensor) Neg() Tensor {
	return t.withUnary("neg")
}

// MatMul computes matrix multiplication t x w.
func (t Tensor) MatMul(w Tensor) Tensor {
	if err := t.checkBinary(w); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	if len(t.shape) != 2 || len(w.shape) != 2 {
		return Tensor{graph: t.graph, err: fmt.Errorf("cudatensor: MatMul requires 2D tensors, got %dD and %dD", len(t.shape), len(w.shape))}
	}
	if t.shape[1] != w.shape[0] {
		return Tensor{graph: t.graph, err: fmt.Errorf("cudatensor: MatMul shape mismatch (%v x %v)", t.shape, w.shape)}
	}
	n, err := t.graph.AddMatMulChecked(t.node, w.node)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: []int{t.shape[0], w.shape[1]},
		dtype: t.dtype,
	}
}

// ReduceSum computes reduction sum along axis.
func (t Tensor) ReduceSum(axis int) Tensor {
	if err := t.checkUnary(); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n, err := t.graph.AddReduceChecked("sum", t.node, axis)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), n.Shape...),
		dtype: t.dtype,
	}
}

// ReduceMax computes reduction maximum along axis.
func (t Tensor) ReduceMax(axis int) Tensor {
	if err := t.checkUnary(); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n, err := t.graph.AddReduceChecked("max", t.node, axis)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), n.Shape...),
		dtype: t.dtype,
	}
}

// Reshape changes the tensor extents without changing total elements.
func (t Tensor) Reshape(newShape []int) Tensor {
	if err := t.checkUnary(); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n, err := t.graph.AddReshapeChecked(t.node, newShape)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), newShape...),
		dtype: t.dtype,
	}
}

// Transpose permutes tensor axes.
func (t Tensor) Transpose(perm []int) Tensor {
	if err := t.checkUnary(); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n, err := t.graph.AddTransposeChecked(t.node, perm)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), n.Shape...),
		dtype: t.dtype,
	}
}

// Slice extracts a hyper-rectangular sub-tensor.
func (t Tensor) Slice(starts, ends, steps []int) Tensor {
	if err := t.checkUnary(); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n, err := t.graph.AddSliceChecked(t.node, starts, ends, steps)
	if err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), n.Shape...),
		dtype: t.dtype,
	}
}

// RMSNorm computes Root Mean Square layer normalization x * w / sqrt(mean(x^2) + eps).
func (t Tensor) RMSNorm(weight Tensor, eps float32) Tensor {
	if err := t.checkBinary(weight); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n := t.graph.AddUnary(fmt.Sprintf("rmsnorm_eps=%g", eps), t.node)
	n.SourceLoc = captureSource()
	normed := Tensor{graph: t.graph, node: n, shape: append([]int(nil), t.shape...), dtype: t.dtype}
	return normed.Mul(weight)
}

// RoPE computes Rotary Position Embedding on query/key tensors.
func (t Tensor) RoPE(pos Tensor, theta float32) Tensor {
	if err := t.checkBinary(pos); err != nil {
		return Tensor{graph: t.graph, err: err}
	}
	n := t.graph.AddBinary(fmt.Sprintf("rope_theta=%g", theta), t.node, pos.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: append([]int(nil), t.shape...),
		dtype: t.dtype,
	}
}

// Compile generates a single fused CUDA mega-kernel source for the graph terminating at t.
func (t Tensor) Compile(kernelName string) (string, error) {
	if t.err != nil {
		return "", fmt.Errorf("cudatensor.Compile: %w", t.err)
	}
	return cudafuse.Fuse(t.graph, kernelName)
}
