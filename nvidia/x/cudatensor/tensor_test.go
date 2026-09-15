package cudatensor_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudatensor"
)

func TestFluentTensorMath(t *testing.T) {
	g := cudair.NewGraph()

	a := cudatensor.NewInput(g, "a", []int{1024}, cudair.Float32)
	b := cudatensor.NewInput(g, "b", []int{1024}, cudair.Float32)
	c := cudatensor.NewInput(g, "c", []int{1024}, cudair.Float32)

	// (a + b) * c -> relu -> exp
	out := a.Add(b).Mul(c).Relu().Exp()

	if out.Graph() != g {
		t.Error("expected matching graph reference")
	}

	src, err := out.Compile("FusedMathKernel")
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}

	if !strings.Contains(src, "FusedMathKernel") {
		t.Error("expected kernel name in compiled source")
	}
	if !strings.Contains(src, "expf(") {
		t.Error("expected expf in compiled source")
	}
}
