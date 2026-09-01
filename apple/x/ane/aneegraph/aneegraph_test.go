//go:build darwin

package aneegraph

import (
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal/egraph"
)

func TestGenerateMILSwiGLU(t *testing.T) {
	expr := &egraph.Expr{
		Op: "fused_swiglu_gemv",
		Children: []*egraph.Expr{
			{Op: "x"},
			{Op: "w_gate"},
			{Op: "w_up"},
			{Op: "w_down"},
		},
	}

	opts := ANEOptions{
		SpatialDim:       1,
		HiddenDim:        64,
		IntermediateDim:  128,
		WeightBlobOffset: 64,
	}

	milText, err := GenerateMIL(expr, opts)
	if err != nil {
		t.Fatalf("GenerateMIL failed: %v", err)
	}

	if !strings.Contains(milText, "program(1.3)") {
		t.Errorf("expected program(1.3), got:\n%s", milText)
	}
	if !strings.Contains(milText, "conv_gate") || !strings.Contains(milText, "conv_up") || !strings.Contains(milText, "swiglu_product") {
		t.Errorf("expected fused SwiGLU ops in generated MIL, got:\n%s", milText)
	}
}
