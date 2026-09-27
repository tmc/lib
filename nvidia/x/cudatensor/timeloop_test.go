package cudatensor

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestExportTimeloopWorkloadRequiresCanonicalGraph(t *testing.T) {
	var out strings.Builder
	if err := ExportTimeloopWorkload(&CanonicalGraph{}, &out); !errors.Is(err, ErrNotCanonicalized) {
		t.Fatalf("error = %v", err)
	}
	g := &Graph{Nodes: []EinsumNode{
		{ID: "a", OutputRank: 2}, {ID: "b", OutputRank: 2},
		{ID: "mm", Equation: "ij,jk->ik", Inputs: []string{"a", "b"}},
	}, Shape: map[string]string{"A": "[1, 64]", "B": "[1, 64]", "C": "[1, 64]"}}
	canonical, err := CanonicalizeRanks(g)
	if err != nil {
		t.Fatal(err)
	}
	if err := ExportTimeloopWorkload(canonical, &out); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Workload struct {
			Version string            `json:"version"`
			Shape   map[string]string `json:"shape"`
			Einsums []struct {
				Name     string `json:"name"`
				Accesses []struct {
					Name       string   `json:"name"`
					Projection []string `json:"projection"`
					Output     bool     `json:"output"`
				} `json:"tensor_accesses"`
			} `json:"einsums"`
		} `json:"workload"`
	}
	if err := json.Unmarshal([]byte(out.String()), &got); err != nil {
		t.Fatal(err)
	}
	if got.Workload.Version != "0.5" || len(got.Workload.Einsums) != 1 || len(got.Workload.Einsums[0].Accesses) != 3 || !got.Workload.Einsums[0].Accesses[2].Output {
		t.Fatalf("unexpected workload: %+v", got)
	}
}

func TestExportTimeloopWorkloadUsesExternalTensorNames(t *testing.T) {
	g := &Graph{Nodes: []EinsumNode{{
		ID: "consume", Equation: "A->A", Inputs: []string{""},
		InputTensors:  []TensorInfo{{Name: "model_input", Elements: 4}},
		OutputTensors: []TensorInfo{{Name: "model_output", Elements: 4}},
	}}}
	canonical, err := CanonicalizeRanks(g)
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	if err := ExportTimeloopWorkload(canonical, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"model_input"`) || !strings.Contains(out.String(), `"model_output"`) {
		t.Fatalf("tensor names missing from workload: %s", out.String())
	}
}
