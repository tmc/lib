package cudatensor

import (
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// Tensor represents a deferred symbolic or device-backed tensor node in a computation DAG.
type Tensor struct {
	graph *cudair.Graph
	node  *cudair.Node
	shape []int
	dtype cudair.DataType
}

func captureSource() cudair.SourceLocation {
	var loc cudair.SourceLocation
	if pc, file, line, ok := runtime.Caller(2); ok {
		loc.File = file
		loc.Line = line
		if fn := runtime.FuncForPC(pc); fn != nil {
			loc.Function = fn.Name()
		}
	}
	return loc
}

// NewInput creates a new input tensor within the given graph.
func NewInput(g *cudair.Graph, name string, shape []int, dtype cudair.DataType) Tensor {
	n := g.AddInput(name, shape, dtype)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: g,
		node:  n,
		shape: shape,
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
	return t.shape
}

// DataType returns the tensor data type.
func (t Tensor) DataType() cudair.DataType {
	return t.dtype
}

// Add computes elementwise addition t + other.
func (t Tensor) Add(other Tensor) Tensor {
	n := t.graph.AddBinary("add", t.node, other.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Sub computes elementwise subtraction t - other.
func (t Tensor) Sub(other Tensor) Tensor {
	n := t.graph.AddBinary("sub", t.node, other.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Mul computes elementwise multiplication t * other.
func (t Tensor) Mul(other Tensor) Tensor {
	n := t.graph.AddBinary("mul", t.node, other.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Div computes elementwise division t / other.
func (t Tensor) Div(other Tensor) Tensor {
	n := t.graph.AddBinary("div", t.node, other.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Relu computes elementwise rectified linear activation max(0, t).
func (t Tensor) Relu() Tensor {
	n := t.graph.AddUnary("relu", t.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Silu computes elementwise sigmoid linear unit (swish activation) x * sigmoid(x).
func (t Tensor) Silu() Tensor {
	n := t.graph.AddUnary("silu", t.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Exp computes elementwise exponential exp(t).
func (t Tensor) Exp() Tensor {
	n := t.graph.AddUnary("exp", t.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Sqrt computes elementwise square root sqrt(t).
func (t Tensor) Sqrt() Tensor {
	n := t.graph.AddUnary("sqrt", t.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Neg computes elementwise negation -t.
func (t Tensor) Neg() Tensor {
	n := t.graph.AddUnary("neg", t.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// MatMul computes matrix multiplication t x w.
func (t Tensor) MatMul(w Tensor) Tensor {
	var outShape []int
	if len(t.shape) == 2 && len(w.shape) == 2 {
		outShape = []int{t.shape[0], w.shape[1]}
	}
	n := t.graph.AddMatMul(t.node, w.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: outShape,
		dtype: t.dtype,
	}
}

// RMSNorm computes Root Mean Square layer normalization x * w / sqrt(mean(x^2) + eps).
func (t Tensor) RMSNorm(weight Tensor, eps float32) Tensor {
	n := t.graph.AddUnary(fmt.Sprintf("rmsnorm_eps=%g", eps), t.node)
	n.SourceLoc = captureSource()
	normed := Tensor{graph: t.graph, node: n, shape: t.shape, dtype: t.dtype}
	return normed.Mul(weight)
}

// RoPE computes Rotary Position Embedding on query/key tensors.
func (t Tensor) RoPE(pos Tensor, theta float32) Tensor {
	n := t.graph.AddBinary(fmt.Sprintf("rope_theta=%g", theta), t.node, pos.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Compile generates a single fused CUDA mega-kernel source for the graph terminating at t.
func (t Tensor) Compile(kernelName string) (string, error) {
	return cudafuse.Fuse(t.graph, kernelName)
}

