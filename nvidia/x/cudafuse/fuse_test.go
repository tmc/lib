package cudafuse_test

import (
	"errors"
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

func TestFuseWithSourceMap(t *testing.T) {
	g := cudair.NewGraph()

	a := g.AddInput("x", []int{512}, cudair.Float32)
	a.SourceLoc = cudair.SourceLocation{File: "model.go", Line: 42, Function: "Forward"}

	b := g.AddUnary("silu", a)
	b.SourceLoc = cudair.SourceLocation{File: "model.go", Line: 43, Function: "Forward"}

	fk, err := cudafuse.FuseWithSourceMap(g, "FusedSilu")
	if err != nil {
		t.Fatalf("FuseWithSourceMap failed: %v", err)
	}

	if len(fk.LineMap) == 0 {
		t.Fatal("expected non-empty source LineMap")
	}

	foundSilu := false
	for _, loc := range fk.LineMap {
		if loc.File == "model.go" && loc.Line == 43 {
			foundSilu = true
			break
		}
	}
	if !foundSilu {
		t.Errorf("expected to find model.go:43 in LineMap, got: %+v", fk.LineMap)
	}
}

func TestFuse_UnsupportedOperators(t *testing.T) {
	tests := []struct {
		name      string
		build     func(g *cudair.Graph)
		errSubstr string
	}{
		{
			name: "matmul",
			build: func(g *cudair.Graph) {
				a := g.AddInput("a", []int{128, 128}, cudair.Float32)
				b := g.AddInput("b", []int{128, 128}, cudair.Float32)
				g.AddBinary("matmul", a, b)
			},
			errSubstr: "matmul cannot be lowered as elementwise",
		},
		{
			name: "rmsnorm",
			build: func(g *cudair.Graph) {
				a := g.AddInput("a", []int{128}, cudair.Float32)
				g.AddUnary("rmsnorm", a)
			},
			errSubstr: "RMSNorm reduction not implemented",
		},
		{
			name: "rope",
			build: func(g *cudair.Graph) {
				a := g.AddInput("a", []int{128}, cudair.Float32)
				g.AddUnary("rope", a)
			},
			errSubstr: "RoPE transformation not implemented",
		},
		{
			name: "unknown_op",
			build: func(g *cudair.Graph) {
				a := g.AddInput("a", []int{128}, cudair.Float32)
				g.AddUnary("unsupported_custom_op", a)
			},
			errSubstr: "unsupported_custom_op",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g := cudair.NewGraph()
			tc.build(g)
			_, err := cudafuse.Fuse(g, "TestKernel")
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
			if !errors.Is(err, cudafuse.ErrUnsupportedOperator) {
				t.Errorf("expected ErrUnsupportedOperator, got %v", err)
			}
			if !strings.Contains(err.Error(), tc.errSubstr) {
				t.Errorf("expected error message to contain %q, got %q", tc.errSubstr, err.Error())
			}
		})
	}
}
