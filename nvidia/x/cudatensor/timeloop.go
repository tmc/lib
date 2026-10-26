package cudatensor

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

// ExportTimeloopWorkload writes a Timeloop v0.5 workload as YAML-compatible
// JSON. It accepts only an unchanged graph returned by CanonicalizeRanks.
// The output includes SOLAR's per-einsum and top-level tensor-role renames.
func ExportTimeloopWorkload(g *CanonicalGraph, w io.Writer) error {
	if !canonicalGraphValid(g) {
		return ErrNotCanonicalized
	}
	if w == nil {
		return errors.New("cudatensor: nil Timeloop writer")
	}
	type access struct {
		Name       string   `json:"name"`
		Projection []string `json:"projection"`
		Output     bool     `json:"output,omitempty"`
	}
	type einsumRenames struct {
		Output string `json:"output"`
		Input  string `json:"input"`
		Weight string `json:"weight"`
	}
	type einsum struct {
		Name           string         `json:"name"`
		TensorAccesses []access       `json:"tensor_accesses"`
		Renames        *einsumRenames `json:"renames,omitempty"`
	}
	type renameAccessRule struct {
		Name          string `json:"name"`
		Source        string `json:"source"`
		ExpectedCount int    `json:"expected_count"`
	}
	type renameEinsumRule struct {
		Name           string             `json:"name"`
		TensorAccesses []renameAccessRule `json:"tensor_accesses"`
	}
	type workloadRenames struct {
		Einsums []renameEinsumRule `json:"einsums"`
	}
	workload := struct {
		Workload struct {
			Version string            `json:"version"`
			Shape   map[string]string `json:"shape"`
			Einsums []einsum          `json:"einsums"`
		} `json:"workload"`
		Renames *workloadRenames `json:"renames,omitempty"`
	}{}
	workload.Workload.Version = "0.5"
	workload.Workload.Shape = cloneStringMap(g.Shape)
	if workload.Workload.Shape == nil {
		workload.Workload.Shape = map[string]string{}
	}
	workload.Renames = &workloadRenames{
		Einsums: []renameEinsumRule{{
			Name: "default",
			TensorAccesses: []renameAccessRule{
				{Name: "input", Source: "Inputs() & Intermediates()", ExpectedCount: 1},
				{Name: "output", Source: "Outputs()", ExpectedCount: 1},
				{Name: "weight", Source: "~(input | output)", ExpectedCount: 1},
			},
		}},
	}
	projected := make(map[string]string)
	for _, n := range g.Nodes {
		if n.Start {
			continue
		}
		inputs, output, err := cudashape.ParseEinsumEquation(n.Equation)
		if err != nil {
			return fmt.Errorf("cudatensor: canonical node %q: %w", n.ID, err)
		}
		if len(inputs) != len(n.Inputs) {
			return fmt.Errorf("cudatensor: canonical node %q has %d operands for %d inputs", n.ID, len(inputs), len(n.Inputs))
		}
		item := einsum{Name: n.ID}
		for i, pred := range n.Inputs {
			name := pred
			if i < len(n.InputTensors) && n.InputTensors[i].Name != "" {
				name = n.InputTensors[i].Name
			}
			if name == "" {
				return fmt.Errorf("cudatensor: canonical node %q input %d has no tensor name", n.ID, i)
			}
			projection := make([]string, len(inputs[i]))
			for j, token := range inputs[i] {
				projection[j] = string(token)
				projected[string(token)] = n.ID
			}
			item.TensorAccesses = append(item.TensorAccesses, access{Name: name, Projection: projection})
		}
		projection := make([]string, len(output))
		for j, token := range output {
			projection[j] = string(token)
			projected[string(token)] = n.ID
		}
		if len(n.OutputTensors) == 0 {
			item.TensorAccesses = append(item.TensorAccesses, access{Name: n.ID, Projection: projection, Output: true})
		} else {
			for _, tensor := range n.OutputTensors {
				if tensor.Name == "" {
					return fmt.Errorf("cudatensor: canonical node %q has unnamed output", n.ID)
				}
				item.TensorAccesses = append(item.TensorAccesses, access{Name: tensor.Name, Projection: projection, Output: true})
			}
		}
		var inputNames, outputNames []string
		for _, a := range item.TensorAccesses {
			if a.Output {
				outputNames = append(outputNames, a.Name)
			} else {
				inputNames = append(inputNames, a.Name)
			}
		}
		if len(inputNames) > 0 && len(outputNames) > 0 {
			weight := "Nothing()"
			if len(inputNames) > 1 {
				weight = inputNames[1]
			}
			item.Renames = &einsumRenames{
				Output: outputNames[0],
				Input:  inputNames[0],
				Weight: weight,
			}
		}
		workload.Workload.Einsums = append(workload.Workload.Einsums, item)
	}
	projectedRanks := make([]string, 0, len(projected))
	for rank := range projected {
		projectedRanks = append(projectedRanks, rank)
	}
	sort.Strings(projectedRanks)
	for _, rank := range projectedRanks {
		node := projected[rank]
		if _, ok := workload.Workload.Shape[rank]; !ok {
			return fmt.Errorf("cudatensor: canonical node %q rank %q has no shape bound", node, rank)
		}
	}
	shapeRanks := make([]string, 0, len(workload.Workload.Shape))
	for rank := range workload.Workload.Shape {
		shapeRanks = append(shapeRanks, rank)
	}
	sort.Strings(shapeRanks)
	for _, rank := range shapeRanks {
		if _, ok := projected[rank]; !ok {
			return fmt.Errorf("cudatensor: shape rank %q is not projected by any einsum", rank)
		}
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(workload)
}
