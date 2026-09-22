package cudafuse_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

func TestFusionAdmission(t *testing.T) {
	tests := []struct {
		name string
		edit func(*cudair.Graph, *cudair.Node, *cudair.Node)
		want string
	}{
		{"half", func(_ *cudair.Graph, x, _ *cudair.Node) { x.Type = cudair.Float16 }, "requires float32"},
		{"integer result", func(_ *cudair.Graph, _, y *cudair.Node) { y.Type = cudair.Int32 }, "requires float32"},
		{"column major", func(_ *cudair.Graph, x, _ *cudair.Node) { x.Layout = cudashape.LayoutColumnMajor }, "row-major"},
		{"strided", func(_ *cudair.Graph, x, _ *cudair.Node) { x.Strides = []int{8, 2} }, "row-major"},
		{"short input", func(_ *cudair.Graph, x, _ *cudair.Node) { x.Shape = []int{1, 4} }, "shape"},
		{"same extent different shape", func(_ *cudair.Graph, x, _ *cudair.Node) { x.Shape, x.Strides = []int{8}, []int{1} }, "shape"},
		{"missing input", func(_ *cudair.Graph, _, y *cudair.Node) { y.Inputs = nil }, "1 inputs"},
		{"extra input", func(_ *cudair.Graph, x, y *cudair.Node) { y.Inputs = []*cudair.Node{x, x} }, "1 inputs"},
		{"nil input", func(_ *cudair.Graph, _, y *cudair.Node) { y.Inputs[0] = nil }, "nil input"},
		{"wrong kind", func(_ *cudair.Graph, _, y *cudair.Node) { y.Kind = cudair.KindInput }, "requires kind"},
		{"mutation", func(_ *cudair.Graph, _, y *cudair.Node) { y.Effect = cudair.EffectMutating }, "storage effects"},
		{"alias", func(_ *cudair.Graph, x, y *cudair.Node) { y.StorageAlias = x }, "storage effects"},
		{"cycle", func(_ *cudair.Graph, _, y *cudair.Node) { y.Inputs[0] = y }, "cycle"},
		{"foreign input", func(_ *cudair.Graph, _, y *cudair.Node) {
			y.Inputs[0] = cudair.NewGraph().AddInput("foreign", []int{2, 4}, cudair.Float32)
		}, "different graph"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{2, 4}, cudair.Float32)
			y := g.AddUnary("neg", x)
			g.SetOutputs(y)
			tt.edit(g, x, y)
			got, err := cudafuse.FuseWithSourceMap(g, "admission")
			if err == nil || !strings.Contains(err.Error(), tt.want) || got.Source != "" {
				t.Fatalf("source=%q, err=%v; want refusal containing %q", got.Source, err, tt.want)
			}
		})
	}
}

func TestFusionDependencyOrder(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2, 4}, cudair.Float32)
	first := g.AddUnary("neg", x)
	later := g.AddUnary("exp", x)
	first.Inputs[0] = later
	g.SetOutputs(first)
	source, err := cudafuse.Fuse(g, "ordered")
	if err != nil {
		t.Fatal(err)
	}
	a := strings.Index(source, "float v_"+later.Name+" =")
	b := strings.Index(source, "float v_"+first.Name+" =")
	if a < 0 || b < 0 || a >= b {
		t.Fatalf("dependency must precede consumer:\n%s", source)
	}
}

func TestFusionNames(t *testing.T) {
	for _, name := range []string{"", "a-b", "x;return", "3x", "λ", "__cuda", "_private", "class", "n", "idx", "expf", "blockDim", "v_x", "out_x", "kernel"} {
		t.Run(name, func(t *testing.T) {
			g := cudair.NewGraph()
			x := g.AddInput("x", []int{2}, cudair.Float32)
			g.AddInput(name, []int{2}, cudair.Float32)
			g.SetOutputs(x)
			if _, err := cudafuse.Fuse(g, "kernel"); err == nil {
				t.Fatalf("accepted conflicting or invalid input %q", name)
			}
		})
	}
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	g.SetOutputs(x)
	for _, kernel := range []string{"", "class", "x(){}", "_reserved", "expf", "v_x", "out_x"} {
		if _, err := cudafuse.Fuse(g, kernel); err == nil {
			t.Errorf("accepted kernel name %q", kernel)
		}
	}
	y := g.AddUnary("neg", x)
	y.Name = x.Name
	if _, err := cudafuse.Fuse(g, "kernel"); err == nil {
		t.Fatal("accepted duplicate node name")
	}
}
