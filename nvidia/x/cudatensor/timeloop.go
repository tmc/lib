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
// The output omits SOLAR's renames block; ranks are already canonical here.
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
	type einsum struct {
		Name           string   `json:"name"`
		TensorAccesses []access `json:"tensor_accesses"`
	}
	workload := struct {
		Workload struct {
			Version string            `json:"version"`
			Shape   map[string]string `json:"shape"`
			Einsums []einsum          `json:"einsums"`
		} `json:"workload"`
	}{}
	workload.Workload.Version = "0.5"
	workload.Workload.Shape = cloneStringMap(g.Shape)
	if workload.Workload.Shape == nil {
		workload.Workload.Shape = map[string]string{}
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
