package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func sidecarExpressionFixture(fused bool) (egraph.Source, map[string][]float32, error) {
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

type sidecarCompileFixture struct {
	name   string
	source egraph.Source
	inputs map[string][]float32
}

func sidecarCompileFixtures() ([]sidecarCompileFixture, error) {
	var result []sidecarCompileFixture
	bar, inputs, err := sidecarExpressionFixture(false)
	if err != nil {
		return nil, err
	}
	result = append(result, sidecarCompileFixture{"bar", bar, inputs})
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
		result = append(result, sidecarCompileFixture{op, source, inputs})
	}
	return result, nil
}

func TestSidecarConsumerGoldens(t *testing.T) {
	for _, fused := range []bool{false, true} {
		t.Run(fmt.Sprintf("fused_%t", fused), func(t *testing.T) {
			source, inputs, err := sidecarExpressionFixture(fused)
			if err != nil {
				t.Fatal(err)
			}
			dir := filepath.Join("testdata", "sidecar", fmt.Sprintf("fused-%t", fused))
			if os.Getenv("UPDATE_SIDECAR_GOLDENS") == "1" {
				if err := os.MkdirAll(dir, 0755); err != nil {
					t.Fatal(err)
				}
				if err := source.WriteFile(filepath.Join(dir, "chain.cu")); err != nil {
					t.Fatal(err)
				}
				data, err := json.MarshalIndent(inputs, "", "    ")
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, "inputs.json"), append(data, '\n'), 0644); err != nil {
					t.Fatal(err)
				}
			}
			code, err := os.ReadFile(filepath.Join(dir, "chain.cu"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(code, []byte(source.CUDA)) {
				t.Fatal("actual emitted consumer source differs from golden")
			}
			ref, err := loadSidecarReference(filepath.Join(dir, "chain.cu"), filepath.Join(dir, "chain.egraph.json"), filepath.Join(dir, "inputs.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := sidecarABI(ref, "chain"); err != nil {
				t.Fatal(err)
			}
			if len(ref.want) == 0 || len(ref.inputs) != len(source.Parameters) {
				t.Fatal("expression reference or pointer order missing")
			}
		})
	}
}
