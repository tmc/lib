package main

import (
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

// Each kernel is written as ordinary Go. Every line that contributes an
// operation is a line the disassembler should be able to name.

func gate(g *cudair.Graph) cudatensor.Tensor {
	x := cudatensor.NewInput(g, "x", shape, cudair.Float32)
	w := cudatensor.NewInput(g, "w", shape, cudair.Float32)
	act := x.Silu()
	return act.Mul(w)
}

func residual(g *cudair.Graph) cudatensor.Tensor {
	h := cudatensor.NewInput(g, "h", shape, cudair.Float32)
	skip := cudatensor.NewInput(g, "skip", shape, cudair.Float32)
	sum := h.Add(skip)
	return sum.Relu()
}

func rescale(g *cudair.Graph) cudatensor.Tensor {
	v := cudatensor.NewInput(g, "v", shape, cudair.Float32)
	d := cudatensor.NewInput(g, "d", shape, cudair.Float32)
	inv := d.Sqrt()
	scaled := v.Div(inv)
	return scaled.Neg()
}

// layered builds its graph through helpers, so the lines that own the
// operations are inside activate and normalize while the lines a reader
// looks at are the two calls below. The roll-up charges each helper's
// instructions to its caller as well.
func layered(g *cudair.Graph) cudatensor.Tensor {
	x := cudatensor.NewInput(g, "x", shape, cudair.Float32)
	w := cudatensor.NewInput(g, "w", shape, cudair.Float32)
	act := activate(x)
	return normalize(act, w)
}

func activate(t cudatensor.Tensor) cudatensor.Tensor {
	return t.Silu()
}

func normalize(t, w cudatensor.Tensor) cudatensor.Tensor {
	scale := w.Sqrt()
	return t.Div(scale)
}

var shape = []int{4096}

// A kernel is a name and the Go that builds its graph.
type kernel struct {
	name  string
	build func(*cudair.Graph) cudatensor.Tensor
}

// kernels is the set the driver compiles, in source order.
var kernels = []kernel{
	{"fused_gate", gate},
	{"fused_residual", residual},
	{"fused_rescale", rescale},
	{"fused_layered", layered},
}
