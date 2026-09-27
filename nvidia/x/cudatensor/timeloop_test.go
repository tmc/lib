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
	}, Shape: map[string]string{"I": "[1, 64]", "J": "[1, 64]", "K": "[1, 64]"}}
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

func TestExportTimeloopWorkloadRequiresProjectedShapeBounds(t *testing.T) {
	base := &Graph{Nodes: []EinsumNode{
		{ID: "a", OutputRank: 2}, {ID: "b", OutputRank: 2},
		{ID: "mm", Equation: "ij,jk->ik", Inputs: []string{"a", "b"}},
	}}
	for _, test := range []struct {
		name  string
		shape map[string]string
		want  string
	}{
		{"missing", map[string]string{"I": "M", "J": "K"}, `node "mm" rank "D"`},
		{"unused", map[string]string{"I": "M", "J": "K", "K": "N", "Z": "O"}, `shape rank "Z"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			g := *base
			g.Shape = test.shape
			canonical, err := CanonicalizeRanks(&g)
			if err != nil {
				t.Fatal(err)
			}
			if err := ExportTimeloopWorkload(canonical, &strings.Builder{}); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ExportTimeloopWorkload error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestExportTimeloopWorkloadUsesExternalTensorNames(t *testing.T) {
	g := &Graph{Shape: map[string]string{"A": "4"}, Nodes: []EinsumNode{{
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
