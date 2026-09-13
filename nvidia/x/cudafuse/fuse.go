package cudafuse

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
)


// FusedKernel represents a synthesized CUDA C++ kernel with line-level source provenance.
type FusedKernel struct {
	Name    string
	Source  string
	LineMap map[int]cudair.SourceLocation // Maps generated C++ line number -> Go source location
	Inputs  []string
	Outputs []string
}

// Fuse generates a single fused CUDA C++ mega-kernel source for the given IR graph.
func Fuse(g *cudair.Graph, kernelName string) (string, error) {
	fk, err := FuseWithSourceMap(g, kernelName)
	if err != nil {
		return "", err
	}
	return fk.Source, nil
}

// FuseWithSourceMap generates a fused kernel and records exact source mappings from generated lines to Go callsites.
func FuseWithSourceMap(g *cudair.Graph, kernelName string) (FusedKernel, error) {
	var fk FusedKernel
	fk.Name = kernelName
	fk.LineMap = make(map[int]cudair.SourceLocation)

	nodes := g.Nodes()
	if len(nodes) == 0 {
		return fk, fmt.Errorf("cudafuse: empty graph")
	}


	var inputs []*cudair.Node
	var outputs []*cudair.Node

	// Find inputs (KindInput) and terminal output nodes (nodes with no consumers)
	consumerCount := make(map[*cudair.Node]int)
	for _, n := range nodes {
		if n.Kind == cudair.KindInput {
			inputs = append(inputs, n)
		}
		for _, inp := range n.Inputs {
			consumerCount[inp]++
		}
	}

	for _, n := range nodes {
		if n.Kind != cudair.KindInput && consumerCount[n] == 0 {
			outputs = append(outputs, n)
		}
	}

	for _, in := range inputs {
		fk.Inputs = append(fk.Inputs, in.Name)
	}
	for _, out := range outputs {
		fk.Outputs = append(fk.Outputs, out.Name)
	}

	var lines []string
	addLine := func(s string, loc *cudair.SourceLocation) {
		lines = append(lines, s)
		if loc != nil && (loc.File != "" || loc.Line > 0) {
			fk.LineMap[len(lines)] = *loc
		}
	}

	addLine(`#include <cuda_runtime.h>`, nil)
	addLine(``, nil)
	addLine(`extern "C" __global__ void `+kernelName+`(`, nil)

	// Input parameters
	for _, in := range inputs {
		addLine(fmt.Sprintf("    const float* __restrict__ %s,", in.Name), &in.SourceLoc)
	}
	// Output parameters
	for i, out := range outputs {
		comma := ","
		if i == len(outputs)-1 {
			comma = ","
		}
		addLine(fmt.Sprintf("    float* __restrict__ out_%s%s", out.Name, comma), &out.SourceLoc)
	}
	addLine("    int n\n) {", nil)

	// Grid-stride loop
	addLine("    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {", nil)

	// Load inputs into local variables
	for _, in := range inputs {
		addLine(fmt.Sprintf("        float v_%s = %s[idx];", in.Name, in.Name), &in.SourceLoc)
	}

	// Compute intermediate expressions
	for _, n := range nodes {
		if n.Kind == cudair.KindInput {
			continue
		}

		var expr string
		switch {
		case n.Op == "relu":
			expr = fmt.Sprintf("(v_%s > 0.0f) ? v_%s : 0.0f", n.Inputs[0].Name, n.Inputs[0].Name)
		case n.Op == "silu":
			expr = fmt.Sprintf("v_%s / (1.0f + expf(-v_%s))", n.Inputs[0].Name, n.Inputs[0].Name)
		case n.Op == "exp":
			expr = fmt.Sprintf("expf(v_%s)", n.Inputs[0].Name)
		case n.Op == "sqrt":
			expr = fmt.Sprintf("sqrtf(v_%s)", n.Inputs[0].Name)
		case n.Op == "neg":
			expr = fmt.Sprintf("-v_%s", n.Inputs[0].Name)
		case n.Op == "add":
			expr = fmt.Sprintf("v_%s + v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case n.Op == "sub":
			expr = fmt.Sprintf("v_%s - v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case n.Op == "mul":
			expr = fmt.Sprintf("v_%s * v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case n.Op == "div":
			expr = fmt.Sprintf("v_%s / v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		default:
			return fk, fmt.Errorf("cudafuse: unsupported operator for elementwise fusion: %s", n.Op)
		}

		addLine(fmt.Sprintf("        float v_%s = %s;", n.Name, expr), &n.SourceLoc)
	}

	// Store outputs
	for _, out := range outputs {
		addLine(fmt.Sprintf("        out_%s[idx] = v_%s;", out.Name, out.Name), &out.SourceLoc)
	}

	addLine("    }", nil)
	addLine("}", nil)

	fk.Source = strings.Join(lines, "\n")
	return fk, nil
}

