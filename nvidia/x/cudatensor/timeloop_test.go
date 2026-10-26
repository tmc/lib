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

func TestExportTimeloopWorkload_SchemaV05(t *testing.T) {
	type accessJSON struct {
		Name       string   `json:"name"`
		Projection []string `json:"projection"`
		Output     bool     `json:"output"`
	}
	type einsumRenamesJSON struct {
		Output string `json:"output"`
		Input  string `json:"input"`
		Weight string `json:"weight"`
	}
	type renameRuleJSON struct {
		Name          string `json:"name"`
		Source        string `json:"source"`
		ExpectedCount int    `json:"expected_count"`
	}
	type docJSON struct {
		Workload struct {
			Version string            `json:"version"`
			Shape   map[string]string `json:"shape"`
			Einsums []struct {
				Name     string             `json:"name"`
				Accesses []accessJSON       `json:"tensor_accesses"`
				Renames  *einsumRenamesJSON `json:"renames"`
			} `json:"einsums"`
		} `json:"workload"`
		Renames *struct {
			Einsums []struct {
				Name     string           `json:"name"`
				Accesses []renameRuleJSON `json:"tensor_accesses"`
			} `json:"einsums"`
		} `json:"renames"`
	}

	assertDefaultTopRenames := func(t *testing.T, doc docJSON) {
		t.Helper()
		if doc.Renames == nil || len(doc.Renames.Einsums) != 1 {
			t.Fatalf("top-level renames = %+v, want 1 default entry", doc.Renames)
		}
		def := doc.Renames.Einsums[0]
		if def.Name != "default" {
			t.Fatalf("top-level renames[0].name = %q, want %q", def.Name, "default")
		}
		wantRules := []renameRuleJSON{
			{Name: "input", Source: "Inputs() & Intermediates()", ExpectedCount: 1},
			{Name: "output", Source: "Outputs()", ExpectedCount: 1},
			{Name: "weight", Source: "~(input | output)", ExpectedCount: 1},
		}
		if len(def.Accesses) != len(wantRules) {
			t.Fatalf("top-level renames.tensor_accesses = %+v, want %+v", def.Accesses, wantRules)
		}
		for i, want := range wantRules {
			if def.Accesses[i] != want {
				t.Fatalf("top-level renames.tensor_accesses[%d] = %+v, want %+v", i, def.Accesses[i], want)
			}
		}
	}

	exportAndDecode := func(t *testing.T, g *Graph) (string, docJSON) {
		t.Helper()
		canonical, err := CanonicalizeRanks(g)
		if err != nil {
			t.Fatalf("CanonicalizeRanks: %v", err)
		}
		var first, second strings.Builder
		if err := ExportTimeloopWorkload(canonical, &first); err != nil {
			t.Fatalf("ExportTimeloopWorkload(first): %v", err)
		}
		if err := ExportTimeloopWorkload(canonical, &second); err != nil {
			t.Fatalf("ExportTimeloopWorkload(second): %v", err)
		}
		if first.String() != second.String() {
			t.Fatalf("non-deterministic export:\nfirst:\n%s\nsecond:\n%s", first.String(), second.String())
		}
		var doc docJSON
		if err := json.Unmarshal([]byte(first.String()), &doc); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		if doc.Workload.Version != "0.5" {
			t.Fatalf("workload.version = %q, want 0.5", doc.Workload.Version)
		}
		assertDefaultTopRenames(t, doc)
		return first.String(), doc
	}

	t.Run("TwoInputAndOneInputChain", func(t *testing.T) {
		g := &Graph{
			Nodes: []EinsumNode{
				{ID: "act", OutputRank: 2},
				{ID: "wt", OutputRank: 2},
				{ID: "mm", Equation: "ij,jk->ik", Inputs: []string{"act", "wt"}},
				{ID: "relu", Equation: "ik->ik", Inputs: []string{"mm"}},
			},
			Shape: map[string]string{
				"I": "[1, 32]",
				"J": "[1, 64]",
				"K": "[1, 128]",
			},
		}
		_, doc := exportAndDecode(t, g)
		if len(doc.Workload.Einsums) != 2 {
			t.Fatalf("len(einsums) = %d, want 2", len(doc.Workload.Einsums))
		}
		mm := doc.Workload.Einsums[0]
		if mm.Name != "mm" || mm.Renames == nil {
			t.Fatalf("mm einsum = %+v, want non-nil renames", mm)
		}
		wantMM := einsumRenamesJSON{Output: "mm", Input: "act", Weight: "wt"}
		if *mm.Renames != wantMM {
			t.Fatalf("mm renames = %+v, want %+v", *mm.Renames, wantMM)
		}
		relu := doc.Workload.Einsums[1]
		if relu.Name != "relu" || relu.Renames == nil {
			t.Fatalf("relu einsum = %+v, want non-nil renames", relu)
		}
		wantRelu := einsumRenamesJSON{Output: "relu", Input: "mm", Weight: "Nothing()"}
		if *relu.Renames != wantRelu {
			t.Fatalf("relu renames = %+v, want %+v", *relu.Renames, wantRelu)
		}
	})

	t.Run("ExternalTensorNamesInRenames", func(t *testing.T) {
		g := &Graph{
			Nodes: []EinsumNode{
				{ID: "n0", OutputRank: 2},
				{ID: "n1", OutputRank: 2},
				{
					ID:       "linear",
					Equation: "ij,jk->ik",
					Inputs:   []string{"n0", "n1"},
					InputTensors: []TensorInfo{
						{Name: "ext_activation", Elements: 16},
						{Name: "ext_filter", Elements: 16, Weight: true},
					},
					OutputTensors: []TensorInfo{
						{Name: "ext_logits", Elements: 16},
					},
				},
			},
			Shape: map[string]string{
				"I": "4",
				"J": "4",
				"K": "4",
			},
		}
		_, doc := exportAndDecode(t, g)
		if len(doc.Workload.Einsums) != 1 || doc.Workload.Einsums[0].Renames == nil {
			t.Fatalf("einsums = %+v, want 1 einsum with renames", doc.Workload.Einsums)
		}
		want := einsumRenamesJSON{
			Output: "ext_logits",
			Input:  "ext_activation",
			Weight: "ext_filter",
		}
		if *doc.Workload.Einsums[0].Renames != want {
			t.Fatalf("renames = %+v, want %+v", *doc.Workload.Einsums[0].Renames, want)
		}
	})

	t.Run("MultiInputAndMultiOutputSelectFirstRoles", func(t *testing.T) {
		g := &Graph{
			Nodes: []EinsumNode{
				{ID: "x0", OutputRank: 2},
				{ID: "x1", OutputRank: 2},
				{ID: "x2", OutputRank: 2},
				{
					ID:       "fused3",
					Equation: "ij,jk,kl->il",
					Inputs:   []string{"x0", "x1", "x2"},
					OutputTensors: []TensorInfo{
						{Name: "out_primary", Elements: 8},
						{Name: "out_aux", Elements: 8},
					},
				},
			},
			Shape: map[string]string{
				"I": "2",
				"J": "4",
				"K": "4",
				"L": "8",
			},
		}
		_, doc := exportAndDecode(t, g)
		if len(doc.Workload.Einsums) != 1 {
			t.Fatalf("len(einsums) = %d, want 1", len(doc.Workload.Einsums))
		}
		e := doc.Workload.Einsums[0]
		if len(e.Accesses) != 5 {
			t.Fatalf("len(tensor_accesses) = %d, want 5 (3 inputs + 2 outputs)", len(e.Accesses))
		}
		if e.Renames == nil {
			t.Fatalf("renames = nil, want populated")
		}
		want := einsumRenamesJSON{
			Output: "out_primary",
			Input:  "x0",
			Weight: "x1",
		}
		if *e.Renames != want {
			t.Fatalf("renames = %+v, want %+v", *e.Renames, want)
		}
	})
}
