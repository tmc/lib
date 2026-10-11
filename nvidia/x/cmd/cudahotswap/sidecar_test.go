package main

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSidecarReferenceSurvivesEditedSource(t *testing.T) {
	a := &egraph.Expr{Class: 1, Origin: "input", Op: "a", Data: "a"}
	b := &egraph.Expr{Class: 2, Origin: "input", Op: "b", Data: "b"}
	expr := &egraph.Expr{Class: 3, Origin: "input", Op: "add", Children: []*egraph.Expr{a, b}}
	source, err := egraph.EmitCUDA(expr, egraph.EmitOptions{KernelName: "chain", Parameters: []egraph.Parameter{{Name: "b", Shape: []int{2}}, {Name: "a", Shape: []int{2}}}})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "chain.cu")
	if err := source.WriteFile(path); err != nil {
		t.Fatal(err)
	}
	metadata, err := os.ReadFile(filepath.Join(filepath.Dir(path), "chain.egraph.json"))
	if err != nil {
		t.Fatal(err)
	}
	inputs := map[string][]float32{"a": {1, 3}, "b": {7, 11}}
	inputPath := filepath.Join(filepath.Dir(path), "inputs.json")
	if err := os.WriteFile(inputPath, []byte(`{"a":[1,3],"b":[7,11]}`), 0600); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadSidecarReference(path, filepath.Join(filepath.Dir(path), "chain.egraph.json"), inputPath)
	if err != nil || !reflect.DeepEqual(loaded.want, []float32{8, 14}) {
		t.Fatalf("load sidecar: %v, %v", loaded, err)
	}
	if err := os.WriteFile(inputPath, []byte(`{"a":[1,3],"b":[7,11]} {}`), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadSidecarReference(path, filepath.Join(filepath.Dir(path), "chain.egraph.json"), inputPath); err == nil {
		t.Fatal("trailing input object accepted")
	}
	ref, err := readSidecarReference(bytes.NewReader(metadata), "// edited\n"+source.CUDA, inputs)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ref.want, []float32{8, 14}) || len(ref.source.Symbols) != 0 {
		t.Fatal("edit changed expression oracle or retained stale attribution")
	}
	inputs["a"][0] = 999
	if ref.inputs[0][0] != 1 || ref.want[0] != 8 {
		t.Fatal("caller mutated retained inputs")
	}
	inputs["extra"] = []float32{1}
	if _, err := readSidecarReference(bytes.NewReader(metadata), source.CUDA, inputs); err == nil {
		t.Fatal("extra input accepted")
	}
}
