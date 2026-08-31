package cudafuse

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// Fuse generates a single fused CUDA C++ mega-kernel source for the given IR graph.
func Fuse(g *cudair.Graph, kernelName string) (string, error) {
	nodes := g.Nodes()
	if len(nodes) == 0 {
		return "", fmt.Errorf("cudafuse: empty graph")
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

	var buf bytes.Buffer
	buf.WriteString(`#include <cuda_runtime.h>` + "\n\n")
	buf.WriteString(`extern "C" __global__ void ` + kernelName + "(\n")

	// Input parameters
	for _, in := range inputs {
		buf.WriteString(fmt.Sprintf("    const float* __restrict__ %s,\n", in.Name))
	}
	// Output parameters
	for i, out := range outputs {
		buf.WriteString(fmt.Sprintf("    float* __restrict__ out_%s", out.Name))
		if i < len(outputs)-1 || true {
			buf.WriteString(",\n")
		}
	}
	buf.WriteString("    int n\n) {\n")

	// Grid-stride loop
	buf.WriteString("    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {\n")

	// Load inputs into local variables
	for _, in := range inputs {
		buf.WriteString(fmt.Sprintf("        float v_%s = %s[idx];\n", in.Name, in.Name))
	}

	// Compute intermediate expressions
	for _, n := range nodes {
		if n.Kind == cudair.KindInput {
			continue
		}

		var expr string
		switch n.Op {
		case "relu":
			expr = fmt.Sprintf("(v_%s > 0.0f) ? v_%s : 0.0f", n.Inputs[0].Name, n.Inputs[0].Name)
		case "exp":
			expr = fmt.Sprintf("expf(v_%s)", n.Inputs[0].Name)
		case "sqrt":
			expr = fmt.Sprintf("sqrtf(v_%s)", n.Inputs[0].Name)
		case "neg":
			expr = fmt.Sprintf("-v_%s", n.Inputs[0].Name)
		case "add":
			expr = fmt.Sprintf("v_%s + v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case "sub":
			expr = fmt.Sprintf("v_%s - v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case "mul":
			expr = fmt.Sprintf("v_%s * v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		case "div":
			expr = fmt.Sprintf("v_%s / v_%s", n.Inputs[0].Name, n.Inputs[1].Name)
		default:
			return "", fmt.Errorf("cudafuse: unsupported operator for fusion: %s", n.Op)
		}

		buf.WriteString(fmt.Sprintf("        float v_%s = %s;\n", n.Name, expr))
	}

	// Store outputs
	for _, out := range outputs {
		buf.WriteString(fmt.Sprintf("        out_%s[idx] = v_%s;\n", out.Name, out.Name))
	}

	buf.WriteString("    }\n}\n")
	return strings.TrimSpace(buf.String()), nil
}
