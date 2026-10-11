package egraph_test

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func emitGoldenExpressionFixture(fused bool) (egraph.Source, map[string][]float32, error) {
	g := egraph.New()
	a, b := g.Add(egraph.NewLeaf("a")), g.Add(egraph.NewLeaf("b"))
	root := g.Add(egraph.NewNode("add", g.Add(egraph.NewNode("silu", a)), g.Add(egraph.NewNode("mul", a, b))))
	parameters := []egraph.Parameter{{Name: "a", Shape: []int{3}}, {Name: "b", Shape: []int{3}}}
	inputs := map[string][]float32{"a": {-2, 0, 3}, "b": {4, 5, -1}}
	if fused {
		w, matrix := g.Add(egraph.NewLeaf("weight")), g.Add(egraph.NewLeaf("matrix"))
		eps := g.Add(egraph.ENode{Op: "constant", Data: float32(1e-5)})
		root = g.Add(egraph.NewNode("fused_rmsnorm_gemv", a, w, eps, matrix))
		parameters = []egraph.Parameter{{Name: "a", Shape: []int{3}}, {Name: "weight", Shape: []int{3}}, {Name: "matrix", Shape: []int{2, 3}}}
		inputs = map[string][]float32{"a": {-2, 1, 3}, "weight": {1, 2, .5}, "matrix": {1, 2, 3, -2, 1, 4}}
	}
	expr, cost, err := egraph.ExtractBest(g, root, egraph.AstSizeCost{})
	if err != nil {
		return egraph.Source{}, nil, err
	}
	source, err := egraph.EmitCUDA(expr, egraph.EmitOptions{KernelName: "chain", Parameters: parameters, Cost: cost, CostModel: "AstSizeCost"})
	return source, inputs, err
}

type emitGoldenCompileFixture struct {
	name   string
	source egraph.Source
	inputs map[string][]float32
}

func emitGoldenCompileFixtures() ([]emitGoldenCompileFixture, error) {
	var result []emitGoldenCompileFixture
	bar, inputs, err := emitGoldenExpressionFixture(false)
	if err != nil {
		return nil, err
	}
	result = append(result, emitGoldenCompileFixture{"bar", bar, inputs})
	for _, op := range []string{"add", "mul", "silu", "gelu", "swish", "rmsnorm", "gemv", "fused_rmsnorm_gemv"} {
		g := egraph.New()
		a, b := g.Add(egraph.NewLeaf("a")), g.Add(egraph.NewLeaf("b"))
		children := []egraph.Id{a}
		parameters := []egraph.Parameter{{Name: "a", Shape: []int{3}}}
		inputs := map[string][]float32{"a": {-2, 1, 3}}
		switch op {
		case "add", "mul":
			children = append(children, b)
			parameters = append(parameters, egraph.Parameter{Name: "b", Shape: []int{3}})
			inputs["b"] = []float32{4, 5, -1}
		case "rmsnorm", "fused_rmsnorm_gemv":
			w := g.Add(egraph.NewLeaf("weight"))
			eps := g.Add(egraph.ENode{Op: "constant", Data: float32(1e-5)})
			children = append(children, w, eps)
			parameters = append(parameters, egraph.Parameter{Name: "weight", Shape: []int{3}})
			inputs["weight"] = []float32{1, 2, .5}
		}
		if op == "gemv" || op == "fused_rmsnorm_gemv" {
			m := g.Add(egraph.NewLeaf("matrix"))
			children = append(children, m)
			parameters = append(parameters, egraph.Parameter{Name: "matrix", Shape: []int{2, 3}})
			inputs["matrix"] = []float32{1, 2, 3, -2, 1, 4}
		}
		root := g.Add(egraph.NewNode(op, children...))
		expr, cost, err := egraph.ExtractBest(g, root, egraph.AstSizeCost{})
		if err != nil {
			return nil, err
		}
		source, err := egraph.EmitCUDA(expr, egraph.EmitOptions{KernelName: "chain", Parameters: parameters, Cost: cost, CostModel: "AstSizeCost"})
		if err != nil {
			return nil, err
		}
		result = append(result, emitGoldenCompileFixture{op, source, inputs})
	}
	return result, nil
}

func TestEmitReadableGoldens(t *testing.T) {
	fixtures, err := emitGoldenCompileFixtures()
	if err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 9 {
		t.Fatalf("got %d examples, want 9", len(fixtures))
	}
	seen := make(map[string]bool)
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			if seen[fixture.name] {
				t.Fatal("duplicate example")
			}
			seen[fixture.name] = true
			path := filepath.Join("testdata", "emit", fixture.name+".cu")
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(want, []byte(fixture.source.CUDA)) {
				t.Fatalf("%s differs from emitted source", path)
			}
		})
	}
	again, err := emitGoldenCompileFixtures()
	if err != nil {
		t.Fatal(err)
	}
	for i := range fixtures {
		if fixtures[i].name != again[i].name || fixtures[i].source.CUDA != again[i].source.CUDA {
			t.Fatalf("example %d is unstable", i)
		}
	}
}
