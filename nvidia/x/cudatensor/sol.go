package cudatensor

import (
	"errors"
	"fmt"
	"math"
)

var (
	// ErrInvalidSOLCeiling reports missing or invalid roofline limits.
	ErrInvalidSOLCeiling = errors.New("cudatensor: invalid SOL ceiling")
	// ErrMissingSOLData reports graph metrics required to compute a SOL bound.
	ErrMissingSOLData = errors.New("cudatensor: missing SOL graph data")
	// ErrSOLOverflow reports a non-finite duration or byte total.
	ErrSOLOverflow = errors.New("cudatensor: SOL result overflow")
)

// Ceiling contains the per-cycle rates and element size used for SOL
// predictions. It matches SOLAR's architecture configuration units.
type Ceiling struct {
	MACsPerCycle    float64
	BytesPerCycle   float64
	FrequencyGHz    float64
	BytesPerElement float64
}

// SOLReport contains intended per-op bounds and the whole-graph values used
// by SOLAR. Durations are seconds; byte fields are decimal byte counts.
type SOLReport struct {
	Unfused              float64
	Fused                float64
	FusedPrefetched      float64
	UnfusedWholeGraph    float64
	FusedWholeGraph      float64
	UnfusedBytes         float64
	FusedBytes           float64
	FusedPrefetchedBytes float64
}

// ComputeSOLBounds computes intended per-op unfused/fused bounds and SOLAR's
// whole-graph values. Fused input and output tensors are deduplicated by name;
// intermediates are excluded. Shared external reads are charged to the
// consumer with the largest MAC count, breaking ties by node ID. Start nodes
// supply external inputs, not work.
func ComputeSOLBounds(g *CanonicalGraph, c Ceiling) (SOLReport, error) {
	if !canonicalGraphValid(g) {
		return SOLReport{}, ErrNotCanonicalized
	}
	if c.MACsPerCycle <= 0 || c.BytesPerCycle <= 0 || c.FrequencyGHz <= 0 || c.BytesPerElement <= 0 ||
		math.IsInf(c.MACsPerCycle, 0) || math.IsInf(c.BytesPerCycle, 0) || math.IsInf(c.FrequencyGHz, 0) || math.IsInf(c.BytesPerElement, 0) ||
		math.IsNaN(c.MACsPerCycle) || math.IsNaN(c.BytesPerCycle) || math.IsNaN(c.FrequencyGHz) || math.IsNaN(c.BytesPerElement) {
		return SOLReport{}, ErrInvalidSOLCeiling
	}
	byID := make(map[string]CanonicalNode, len(g.Nodes))
	produced := make(map[string]string)
	consumed := make(map[string]bool)
	tensorSizes := make(map[string]uint64)
	registerTensor := func(nodeID string, tensor TensorInfo) error {
		if tensor.Name == "" {
			return fmt.Errorf("%w: node %q has unnamed tensor", ErrMissingSOLData, nodeID)
		}
		if size, ok := tensorSizes[tensor.Name]; ok && size != tensor.Elements {
			return fmt.Errorf("cudatensor: tensor %q has inconsistent element counts %d and %d", tensor.Name, size, tensor.Elements)
		}
		tensorSizes[tensor.Name] = tensor.Elements
		return nil
	}
	for _, n := range g.Nodes {
		byID[n.ID] = n
		for _, tensor := range n.OutputTensors {
			if err := registerTensor(n.ID, tensor); err != nil {
				return SOLReport{}, err
			}
			if _, exists := produced[tensor.Name]; exists {
				return SOLReport{}, fmt.Errorf("cudatensor: tensor %q has multiple producers", tensor.Name)
			}
			produced[tensor.Name] = n.ID
		}
	}
	for _, n := range g.Nodes {
		if n.Start {
			continue
		}
		if len(n.Inputs) != len(n.InputTensors) {
			return SOLReport{}, fmt.Errorf("%w: node %q needs input tensor metadata for every operand", ErrMissingSOLData, n.ID)
		}
		if len(n.OutputTensors) == 0 {
			return SOLReport{}, fmt.Errorf("%w: node %q has no output tensor metadata", ErrMissingSOLData, n.ID)
		}
		for i, tensor := range n.InputTensors {
			if err := registerTensor(n.ID, tensor); err != nil {
				return SOLReport{}, fmt.Errorf("input %d: %w", i, err)
			}
			if source := n.Inputs[i]; source != "" {
				pred, ok := byID[source]
				if !ok {
					return SOLReport{}, fmt.Errorf("cudatensor: node %q has unknown producer %q", n.ID, source)
				}
				if !pred.Start {
					consumed[tensor.Name] = true
				}
				found := false
				for _, output := range pred.OutputTensors {
					if output.Name == tensor.Name && output.Elements == tensor.Elements {
						found = true
						break
					}
				}
				if !found {
					return SOLReport{}, fmt.Errorf("cudatensor: node %q input tensor %q does not match producer %q", n.ID, tensor.Name, source)
				}
			} else if producer, ok := produced[tensor.Name]; ok && !tensor.Weight {
				return SOLReport{}, fmt.Errorf("cudatensor: node %q input tensor %q omits producer %q", n.ID, tensor.Name, producer)
			}
		}
	}
	// Assign each external read to a stable consumer independent of node order.
	externalOwner := make(map[string]string)
	for _, n := range g.Nodes {
		if n.Start {
			continue
		}
		for i, input := range n.InputTensors {
			source := n.Inputs[i]
			external := input.Weight || source == "" || byID[source].Start
			if !external {
				continue
			}
			owner, ok := externalOwner[input.Name]
			if !ok || byID[owner].MACs < n.MACs || (byID[owner].MACs == n.MACs && n.ID < owner) {
				externalOwner[input.Name] = n.ID
			}
		}
	}
	seenExternal := make(map[string]bool)
	seenOutputs := make(map[string]bool)
	var report SOLReport
	var totalCompute, totalUnfusedBytes, totalFusedBytes float64
	for _, n := range g.Nodes {
		if n.Start {
			continue
		}
		compute := float64(n.MACs) / c.MACsPerCycle
		var unfusedBytes, fusedBytes float64
		for i, input := range n.InputTensors {
			bytes := float64(input.Elements) * c.BytesPerElement
			unfusedBytes += bytes
			source := n.Inputs[i]
			external := input.Weight || source == "" || byID[source].Start
			if external && externalOwner[input.Name] == n.ID && !seenExternal[input.Name] {
				fusedBytes += bytes
				seenExternal[input.Name] = true
			}
		}
		for _, output := range n.OutputTensors {
			bytes := float64(output.Elements) * c.BytesPerElement
			unfusedBytes += bytes
			if !consumed[output.Name] && !seenOutputs[output.Name] {
				fusedBytes += bytes
				seenOutputs[output.Name] = true
			}
		}
		memoryUnfused := unfusedBytes / c.BytesPerCycle
		memoryFused := fusedBytes / c.BytesPerCycle
		report.Unfused += math.Max(compute, memoryUnfused)
		report.Fused += math.Max(compute, memoryFused)
		totalCompute += compute
		totalUnfusedBytes += unfusedBytes
		totalFusedBytes += fusedBytes
	}
	wholeUnfused := totalUnfusedBytes / c.BytesPerCycle
	wholeFused := totalFusedBytes / c.BytesPerCycle
	cyclesPerSecond := c.FrequencyGHz * 1e9
	report.Unfused = report.Unfused / cyclesPerSecond
	report.Fused = report.Fused / cyclesPerSecond
	report.UnfusedWholeGraph = math.Max(totalCompute, wholeUnfused) / cyclesPerSecond
	report.FusedWholeGraph = math.Max(totalCompute, wholeFused) / cyclesPerSecond
	report.FusedPrefetched = report.FusedWholeGraph
	report.UnfusedBytes = totalUnfusedBytes
	report.FusedBytes = totalFusedBytes
	report.FusedPrefetchedBytes = totalFusedBytes
	for _, value := range []float64{report.Unfused, report.Fused, report.FusedPrefetched, report.UnfusedWholeGraph, report.FusedWholeGraph, report.UnfusedBytes, report.FusedBytes} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return SOLReport{}, ErrSOLOverflow
		}
	}
	return report, nil
}
