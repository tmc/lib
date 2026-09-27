package cudatensor

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

// ExportTimeloopWorkload writes a Timeloop v0.5 workload as YAML-compatible
// JSON. It accepts only a graph returned by CanonicalizeRanks.
func ExportTimeloopWorkload(g *CanonicalGraph, w io.Writer) error {
	if g == nil || !g.canonicalized {
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
	for _, n := range g.Nodes {
		if n.Start {
			continue
		}
		inputs, output, err := cudashape.ParseEinsumEquation(n.Equation)
		if err != nil {
			return fmt.Errorf("cudatensor: canonical node %q: %w", n.ID, err)
		}
		item := einsum{Name: n.ID}
		for i, pred := range n.Inputs {
			projection := make([]string, len(inputs[i]))
			for j, token := range inputs[i] {
				projection[j] = string(token)
			}
			item.TensorAccesses = append(item.TensorAccesses, access{Name: pred, Projection: projection})
		}
		projection := make([]string, len(output))
		for j, token := range output {
			projection[j] = string(token)
		}
		item.TensorAccesses = append(item.TensorAccesses, access{Name: n.ID, Projection: projection, Output: true})
		workload.Workload.Einsums = append(workload.Workload.Einsums, item)
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(workload)
}
