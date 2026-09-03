package metaltensor

import (
	"fmt"
	"runtime"
	"sort"
	"strings"
	"unicode"

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
	return Tensor{
		graph: t.graph,
		node:  n,
		shape: n.Shape,
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

// Compile emits an elementwise MSL kernel for the expression ending at t.
// MatMul, RMSNorm, and RoPE lowering is not yet supported.
func (t Tensor) Compile(kernelName string) (string, error) {
	if t.graph == nil || t.node == nil {
		return "", fmt.Errorf("compile: invalid tensor")
	}
	if !identifier(kernelName) {
		return "", fmt.Errorf("compile: invalid kernel name %q", kernelName)
	}
	if err := t.graph.Validate(); err != nil {
		return "", fmt.Errorf("compile: %w", err)
	}
	reachable := make(map[*metalir.Node]bool)
	var visit func(*metalir.Node)
	visit = func(n *metalir.Node) {
		if reachable[n] {
			return
		}
		for _, in := range n.Inputs {
			visit(in)
		}
		reachable[n] = true
	}
	visit(t.node)
	var nodes []*metalir.Node
	for _, n := range t.graph.Nodes {
		if reachable[n] {
			nodes = append(nodes, n)
		}
	}
	var inputs []*metalir.Node
	for _, n := range nodes {
		if n.Type != metalir.Float32 {
			return "", fmt.Errorf("compile: node %d: data type %s unsupported", n.ID, n.Type)
		}
		if n.Kind == metalir.KindInput {
			inputs = append(inputs, n)
			continue
		}
		if n.Kind != metalir.KindUnary && n.Kind != metalir.KindBinary {
			return "", fmt.Errorf("compile: node %d: operation %q unsupported", n.ID, n.Op)
		}
		if n.Op != "add" && n.Op != "mul" && n.Op != "silu" {
			return "", fmt.Errorf("compile: node %d: operation %q unsupported", n.ID, n.Op)
		}
	}
	sort.Slice(inputs, func(i, j int) bool { return inputs[i].ID < inputs[j].ID })
	var b strings.Builder
	b.WriteString("#include <metal_stdlib>\nusing namespace metal;\n\nkernel void ")
	b.WriteString(kernelName)
	b.WriteString("(\n")
	for i, n := range inputs {
		fmt.Fprintf(&b, "    device const float* in%d [[buffer(%d)]],\n", n.ID, i)
	}
	fmt.Fprintf(&b, "    device float* out [[buffer(%d)]],\n    uint i [[thread_position_in_grid]]) {\n", len(inputs))
	for _, n := range nodes {
		switch n.Kind {
		case metalir.KindInput:
			fmt.Fprintf(&b, "    float v%d = in%d[i];\n", n.ID, n.ID)
		case metalir.KindUnary:
			fmt.Fprintf(&b, "    float v%d = v%d / (1.0f + exp(-v%d));\n", n.ID, n.Inputs[0].ID, n.Inputs[0].ID)
		case metalir.KindBinary:
			op := "+"
			if n.Op == "mul" {
				op = "*"
			}
			fmt.Fprintf(&b, "    float v%d = v%d %s v%d;\n", n.ID, n.Inputs[0].ID, op, n.Inputs[1].ID)
		}
	}
	fmt.Fprintf(&b, "    out[i] = v%d;\n}\n", t.node.ID)
	return b.String(), nil
}

func identifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if !(r == '_' || unicode.IsLetter(r) || i > 0 && unicode.IsDigit(r)) {
			return false
		}
	}
	return true
}
