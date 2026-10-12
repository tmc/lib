package cudaautograd_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaautograd"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// TestDifferentiateDataTypePolicy checks that only float32 graphs are
// differentiated. Integer and reduced-precision losses, paths, and targets
// have no defined gradient or accumulation type and must be refused.
func TestDifferentiateDataTypePolicy(t *testing.T) {
	tests := []struct {
		name                 string
		build                func(g *cudair.Graph) (loss *cudair.Node, targets []*cudair.Node)
		ok                   bool
		snapshotTypeMismatch bool
	}{
		{"float32", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.Float32)
			return g.AddReduce("sum", g.AddBinary("mul", x, x), 0), []*cudair.Node{x}
		}, true, false},
		{"int32 loss", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.Int32)
			return g.AddReduce("sum", x, 0), []*cudair.Node{x}
		}, false, false},
		{"int8 loss", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.Int8)
			return g.AddReduce("sum", x, 0), []*cudair.Node{x}
		}, false, false},
		{"int8 matmul loss", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			a := g.AddInput("a", []int{2, 2}, cudair.Int8)
			b := g.AddInput("b", []int{2, 2}, cudair.Int8)
			r := g.AddReduce("sum", g.AddMatMul(a, b), 1)
			return g.AddReduce("sum", r, 0), []*cudair.Node{b}
		}, false, false},
		// Node.Type is an exported field. These two manually retyped outputs
		// are malformed IR and must be rejected by Snapshot before autodiff.
		{"int8 matmul path", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			a := g.AddInput("a", []int{2, 2}, cudair.Float32)
			b := g.AddInput("b", []int{2, 2}, cudair.Float32)
			m := g.AddMatMul(a, b)
			m.Type = cudair.Int8
			r := g.AddReduce("sum", m, 1)
			loss := g.AddReduce("sum", r, 0)
			r.Type, loss.Type = cudair.Float32, cudair.Float32
			return loss, []*cudair.Node{b}
		}, false, true},
		{"int32 relu path", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.Float32)
			u := g.AddUnary("relu", x)
			u.Type = cudair.Int32
			loss := g.AddReduce("sum", u, 0)
			loss.Type = cudair.Float32
			return loss, []*cudair.Node{x}
		}, false, true},
		{"float16 loss", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.Float16)
			return g.AddReduce("sum", g.AddUnary("exp", x), 0), []*cudair.Node{x}
		}, false, false},
		{"bfloat16 loss", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.BFloat16)
			return g.AddReduce("sum", x, 0), []*cudair.Node{x}
		}, false, false},
		{"int32 loss float32 target", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			w := g.AddInput("w", []int{4}, cudair.Float32)
			x := g.AddInput("x", []int{4}, cudair.Int32)
			return g.AddReduce("sum", g.AddUnary("neg", x), 0), []*cudair.Node{w}
		}, false, false},
		{"int32 target off path", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.Float32)
			idx := g.AddInput("idx", []int{4}, cudair.Int32)
			return g.AddReduce("sum", x, 0), []*cudair.Node{x, idx}
		}, false, false},
		{"float16 target off path", func(g *cudair.Graph) (*cudair.Node, []*cudair.Node) {
			x := g.AddInput("x", []int{4}, cudair.Float32)
			h := g.AddInput("h", []int{4}, cudair.Float16)
			return g.AddReduce("sum", x, 0), []*cudair.Node{h}
		}, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := cudair.NewGraph()
			loss, targets := tt.build(g)
			if err := g.SetOutputs(loss); err != nil {
				t.Fatal(err)
			}
			s, err := g.Snapshot()
			if tt.snapshotTypeMismatch {
				if !errors.Is(err, cudair.ErrTypeMismatch) {
					t.Fatalf("Snapshot() = %v, want ErrTypeMismatch", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			res, err := cudaautograd.Differentiate(s, loss, targets)
			if tt.ok {
				if err != nil {
					t.Fatalf("Differentiate: %v", err)
				}
				for _, target := range targets {
					grad, ok := res.TargetGradientNode(target)
					if !ok || grad.Type != cudair.Float32 {
						t.Fatalf("gradient for %s: node=%v ok=%v", target.Name, grad, ok)
					}
				}
				return
			}
			if !errors.Is(err, cudaautograd.ErrUnsupportedDerivative) {
				t.Fatalf("Differentiate = %v, want ErrUnsupportedDerivative", err)
			}
			if res != nil {
				t.Fatalf("refused graph returned a backward result")
			}
		})
	}
}
