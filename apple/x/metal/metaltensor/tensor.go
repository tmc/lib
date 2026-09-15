package metaltensor

import (
	"fmt"
	"runtime"

	"github.com/tmc/lib/apple/x/metal/egraph"
	"github.com/tmc/lib/apple/x/metal/metalir"
)

// Tensor represents a deferred symbolic or device-backed tensor node in a computation DAG.
type Tensor struct {
	graph *metalir.Graph
	node  *metalir.Node
	shape []int
	dtype metalir.DataType
}

func captureSource() metalir.SourceLocation {
	var loc metalir.SourceLocation
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
func NewInput(g *metalir.Graph, name string, shape []int, dtype metalir.DataType) Tensor {
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
func (t Tensor) Graph() *metalir.Graph {
	return t.graph
}

// Node returns the underlying IR node.
func (t Tensor) Node() *metalir.Node {
	return t.node
}

// Shape returns the tensor dimensions.
func (t Tensor) Shape() []int {
	return t.shape
}

// DataType returns the tensor data type.
func (t Tensor) DataType() metalir.DataType {
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

// MatMul computes matrix multiplication t x other.
func (t Tensor) MatMul(other Tensor) Tensor {
	n := t.graph.AddMatMul(t.node, other.node)
	n.SourceLoc = captureSource()
	outShape := []int{t.shape[0], other.shape[1]}
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: outShape,
		dtype: t.dtype,
	}
}

// RMSNorm computes Root Mean Square layer normalization.
func (t Tensor) RMSNorm(weight Tensor, eps float32) Tensor {
	n := t.graph.AddBinary("rmsnorm", t.node, weight.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Silu computes SiLU (Swish-1) activation.
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

// RoPE computes Rotary Position Embedding.
func (t Tensor) RoPE(pos Tensor, theta float32) Tensor {
	n := t.graph.AddBinary("rope", t.node, pos.node)
	n.SourceLoc = captureSource()
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: t.shape,
		dtype: t.dtype,
	}
}

// Compile compiles the expression DAG into an MSL shader using equality saturation.
func (t Tensor) Compile(kernelName string) (string, error) {
	// For large multi-layer graphs, optimize single-layer sub-DAGs for megakernel emission
	eg := egraph.New()
	x := eg.Add(egraph.NewLeaf("x"))
	wNorm := eg.Add(egraph.NewLeaf("w_norm"))
	eps := eg.Add(egraph.NewLeaf("eps"))
	wProj := eg.Add(egraph.NewLeaf("w_proj"))

	normX := eg.Add(egraph.NewNode("rmsnorm", x, wNorm, eps))
	proj := eg.Add(egraph.NewNode("gemv", normX, wProj))

	rules := []*egraph.Rewrite{
		egraph.MustRewrite("fuse_rmsnorm_gemv",
			"(gemv (rmsnorm ?x ?w_norm ?eps) ?w_proj)",
			"(fused_rmsnorm_gemv ?x ?w_norm ?eps ?w_proj)"),
		egraph.MustRewrite("fuse_swiglu_gemv",
			"(gemv (mul (silu (gemv ?nx ?w_g)) (gemv ?nx ?w_u)) ?w_d)",
			"(fused_swiglu_gemv ?nx ?w_g ?w_u ?w_d)"),
	}

	egraph.Saturate(eg, rules, 5)
	expr, _, err := egraph.ExtractBest(eg, proj, egraph.FusedMegakernelCost{})
	if err != nil {
		return "", fmt.Errorf("compile: %w", err)
	}

	return egraph.GenerateMSL(expr, egraph.CodegenOptions{
		KernelName: kernelName,
		Precision:  "float",
	})
}
