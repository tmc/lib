// Command llama-ir demonstrates construction and validation of a symbolic
// decoder-layer graph. The current tensor compiler does not lower attention.
package main

import (
	"fmt"
	"log"

	"github.com/tmc/lib/apple/x/metal/metalir"
	"github.com/tmc/lib/apple/x/metal/metaltensor"
)

const (
	sequence  = 8
	dimension = 64
	hidden    = 128
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	g, out := decoderGraph()
	if err := g.Validate(); err != nil {
		return fmt.Errorf("validate decoder graph: %w", err)
	}
	plan, err := metalir.PlanMemory(g, 256)
	if err != nil {
		return fmt.Errorf("plan decoder graph: %w", err)
	}
	_, lowerErr := out.Compile("DecoderLayer")
	if lowerErr == nil {
		return fmt.Errorf("lower decoder graph: unexpectedly supported")
	}
	fmt.Printf("validated symbolic decoder graph: nodes=%d output=%v scratch=%d bytes\n", len(g.Nodes), out.Shape(), plan.TotalBytes)
	fmt.Printf("lowering unsupported: %v\n", lowerErr)
	return nil
}

// decoderGraph builds a shape-correct single-head attention and SwiGLU layer.
// K is supplied transposed as [D,S]; the IR has no transpose operation yet.
// Softmax and causal masking are also absent, so scores are not executable
// attention probabilities. The graph exists to demonstrate shape validation.
func decoderGraph() (*metalir.Graph, metaltensor.Tensor) {
	g := metalir.NewGraph()
	x := metaltensor.NewInput(g, "x", []int{sequence, dimension}, metalir.Float32)
	norm := metaltensor.NewInput(g, "attention_norm", []int{sequence, dimension}, metalir.Float32)
	q := x.RMSNorm(norm, 1e-5).MatMul(metaltensor.NewInput(g, "wq", []int{dimension, dimension}, metalir.Float32))
	kt := metaltensor.NewInput(g, "k_transpose", []int{dimension, sequence}, metalir.Float32)
	v := metaltensor.NewInput(g, "v", []int{sequence, dimension}, metalir.Float32)
	context := q.MatMul(kt).MatMul(v)
	attn := context.MatMul(metaltensor.NewInput(g, "wo", []int{dimension, dimension}, metalir.Float32))
	h := x.Add(attn)
	fn := h.RMSNorm(metaltensor.NewInput(g, "ffn_norm", []int{sequence, dimension}, metalir.Float32), 1e-5)
	gate := fn.MatMul(metaltensor.NewInput(g, "w_gate", []int{dimension, hidden}, metalir.Float32)).Silu()
	up := fn.MatMul(metaltensor.NewInput(g, "w_up", []int{dimension, hidden}, metalir.Float32))
	down := gate.Mul(up).MatMul(metaltensor.NewInput(g, "w_down", []int{hidden, dimension}, metalir.Float32))
	return g, h.Add(down)
}
