package cudafuse

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestCompileRegionKernelReturnsFuseError(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{4}, cudair.Float32)
	unsupported := g.AddUnary("rmsnorm", x)
	if _, err := compileRegionKernel(g, "bad_fusion", unsupported); err == nil || !errors.Is(err, ErrUnsupportedOperator) {
		t.Fatalf("compileRegionKernel error = %v, want ErrUnsupportedOperator", err)
	}
}

func TestCompileRegionKernelReturnsSetOutputsError(t *testing.T) {
	g := cudair.NewGraph()
	foreign := cudair.NewGraph().AddInput("foreign", []int{4}, cudair.Float32)
	if _, err := compileRegionKernel(g, "foreign_output", foreign); !errors.Is(err, cudair.ErrInvalidOutput) {
		t.Fatalf("compileRegionKernel error = %v, want ErrInvalidOutput", err)
	}
}
