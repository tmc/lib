package cudair_test

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestSnapshotRejectsInvalidDataType(t *testing.T) {
	g := cudair.NewGraph()
	g.AddInput("x", []int{1}, cudair.DataType(99))
	if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrTypeMismatch) {
		t.Fatalf("Snapshot() error = %v, want ErrTypeMismatch", err)
	}
}

func TestSnapshotChecksOutputTypeByKind(t *testing.T) {
	tests := []struct {
		name  string
		build func(*cudair.Graph) *cudair.Node
	}{
		{"unary", func(g *cudair.Graph) *cudair.Node {
			return g.AddUnary("neg", g.AddInput("x", []int{2}, cudair.Float32))
		}},
		{"binary", func(g *cudair.Graph) *cudair.Node {
			return g.AddBinary("add", g.AddInput("x", []int{2}, cudair.Float32), g.AddInput("y", []int{2}, cudair.Float32))
		}},
		{"matmul", func(g *cudair.Graph) *cudair.Node {
			return g.AddMatMul(g.AddInput("x", []int{2, 3}, cudair.Float32), g.AddInput("y", []int{3, 4}, cudair.Float32))
		}},
		{"reduce", func(g *cudair.Graph) *cudair.Node {
			return g.AddReduce("sum", g.AddInput("x", []int{2, 3}, cudair.Float32), 1)
		}},
		{"reshape", func(g *cudair.Graph) *cudair.Node {
			return g.AddReshape(g.AddInput("x", []int{2, 3}, cudair.Float32), []int{3, 2})
		}},
		{"transpose", func(g *cudair.Graph) *cudair.Node {
			return g.AddTranspose(g.AddInput("x", []int{2, 3}, cudair.Float32), []int{1, 0})
		}},
		{"slice", func(g *cudair.Graph) *cudair.Node {
			return g.AddSlice(g.AddInput("x", []int{3}, cudair.Float32), []int{0}, []int{2}, []int{1})
		}},
		{"concat", func(g *cudair.Graph) *cudair.Node {
			x := g.AddInput("x", []int{1, 2}, cudair.Float32)
			y := g.AddInput("y", []int{1, 3}, cudair.Float32)
			return g.AddConcat([]*cudair.Node{x, y}, 1)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := cudair.NewGraph()
			out := tt.build(g)
			if out == nil {
				t.Fatal("builder returned nil")
			}
			out.Type = cudair.Int8
			if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrTypeMismatch) {
				t.Fatalf("Snapshot() error = %v, want ErrTypeMismatch", err)
			}
		})
	}
}

func TestSnapshotRejectsMismatchedMatMulInputTypes(t *testing.T) {
	g := cudair.NewGraph()
	a := g.AddInput("a", []int{2, 3}, cudair.Float32)
	b := g.AddInput("b", []int{3, 2}, cudair.Float32)
	out := g.AddMatMul(a, b)
	b.Type = cudair.Float16
	if _, err := g.Snapshot(); !errors.Is(err, cudair.ErrTypeMismatch) {
		t.Fatalf("Snapshot() error = %v, want ErrTypeMismatch", err)
	}
	_ = out
}
