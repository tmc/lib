package cudafuse_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestFuse(t *testing.T) {
	g := cudair.NewGraph()

	a := g.AddInput("a", []int{1024}, cudair.Float32)
	b := g.AddInput("b", []int{1024}, cudair.Float32)
	c := g.AddInput("c", []int{1024}, cudair.Float32)

	ab := g.AddBinary("add", a, b)
	abc := g.AddBinary("mul", ab, c)
	relu := g.AddUnary("relu", abc)
	exp := g.AddUnary("exp", relu)

	_ = exp

	src, err := cudafuse.Fuse(g, "FusedElementwiseOp")
	if err != nil {
		t.Fatalf("Fuse failed: %v", err)
	}

	if !strings.Contains(src, "FusedElementwiseOp") {
		t.Error("expected kernel name in generated source")
	}

	if !strings.Contains(src, "expf(") || !strings.Contains(src, "v_a + v_b") {
		t.Errorf("expected fused arithmetic in source, got:\n%s", src)
	}
}
