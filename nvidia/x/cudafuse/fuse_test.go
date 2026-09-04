package cudafuse_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestFuseNilGraph(t *testing.T) {
	for _, test := range []struct {
		name string
		fuse func() error
	}{
		{"source", func() error { _, err := cudafuse.Fuse(nil, "kernel"); return err }},
		{"source map", func() error { _, err := cudafuse.FuseWithSourceMap(nil, "kernel"); return err }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.fuse(); err == nil || err.Error() != "cudafuse: nil graph" {
				t.Fatalf("error = %v, want cudafuse: nil graph", err)
			}
		})
	}
}

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

func TestFuseRejectsInvalidGraph(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2}, cudair.Float32)
	b := g.AddInput("b", []int{3}, cudair.Float32)
	g.AddBinary("add", a, b)
	if _, err := cudafuse.Fuse(g, "invalid"); err == nil {
		t.Fatal("Fuse succeeded")
	}
}

func TestFuseRejectsMatMul(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 3}, cudair.Float32)
	b := g.AddInput("b", []int{3, 4}, cudair.Float32)
	g.AddMatMul(a, b)
	if _, err := cudafuse.Fuse(g, "matmul"); err == nil {
		t.Fatal("Fuse succeeded")
	}
}

func TestFuseRejectsInvalidKernelName(t *testing.T) {
	g := cudair.NewGraph()
	g.AddInput("a", []int{2}, cudair.Float32)
	if _, err := cudafuse.Fuse(g, "bad() {}"); err == nil {
		t.Fatal("Fuse succeeded")
	}
}

func TestFuseRejectsUnsupportedSemantics(t *testing.T) {
	tests := []struct {
		name  string
		build func(*cudair.Graph)
	}{
		{"float16", func(g *cudair.Graph) { g.AddInput("a", []int{2}, cudair.Float16) }},
		{"rmsnorm", func(g *cudair.Graph) {
			a := g.AddInput("a", []int{2}, cudair.Float32)
			g.AddUnary("rmsnorm_eps=1e-05", a)
		}},
		{"rope", func(g *cudair.Graph) {
			a := g.AddInput("a", []int{2}, cudair.Float32)
			b := g.AddInput("b", []int{2}, cudair.Float32)
			g.AddBinary("rope_theta=10000", a, b)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := cudair.NewGraph()
			test.build(g)
			if _, err := cudafuse.Fuse(g, "kernel"); err == nil {
				t.Fatal("Fuse succeeded")
			}
		})
	}
}
