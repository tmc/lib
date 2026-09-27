package cudatensor

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func solTestGraph() *Graph {
	return &Graph{Nodes: []EinsumNode{
		{ID: "input", OutputRank: 1, OutputTensors: []TensorInfo{{Name: "x", Elements: 10}}},
		{ID: "first", Equation: "A->A", Inputs: []string{"input"}, MACs: 100,
			InputTensors:  []TensorInfo{{Name: "x", Elements: 10}},
			OutputTensors: []TensorInfo{{Name: "y", Elements: 10}}},
		{ID: "second", Equation: "A,B->A", Inputs: []string{"input", "first"}, MACs: 1,
			InputTensors:  []TensorInfo{{Name: "x", Elements: 10}, {Name: "y", Elements: 10}},
			OutputTensors: []TensorInfo{{Name: "z", Elements: 1000}}},
	}}
}

func TestComputeSOLBounds_FusedDeduplicatesExternalReadsAndDropsIntermediates(t *testing.T) {
	g, err := CanonicalizeRanks(solTestGraph())
	if err != nil {
		t.Fatal(err)
	}
	report, err := ComputeSOLBounds(g, Ceiling{MACsPerCycle: 1, BytesPerCycle: 100, FrequencyGHz: 1, BytesPerElement: 2})
	if err != nil {
		t.Fatal(err)
	}
	if report.UnfusedBytes != 2080 || report.FusedBytes != 2020 || report.FusedPrefetchedBytes != report.FusedBytes {
		t.Fatalf("bytes: unfused=%g fused=%g prefetched=%g", report.UnfusedBytes, report.FusedBytes, report.FusedPrefetchedBytes)
	}
}

func TestComputeSOLBounds_PerOpIntentAndWholeGraphDivergence(t *testing.T) {
	g, err := CanonicalizeRanks(solTestGraph())
	if err != nil {
		t.Fatal(err)
	}
	report, err := ComputeSOLBounds(g, Ceiling{MACsPerCycle: 1, BytesPerCycle: 100, FrequencyGHz: 1, BytesPerElement: 2})
	if err != nil {
		t.Fatal(err)
	}
	if report.Unfused < report.UnfusedWholeGraph || report.Fused < report.FusedWholeGraph {
		t.Fatalf("intended bounds below whole graph: %+v", report)
	}
	if report.Fused <= report.FusedWholeGraph || report.FusedPrefetched != report.FusedWholeGraph {
		t.Fatalf("expected visible per-op/whole-graph divergence: %+v", report)
	}
}

func TestComputeSOLBounds_FailsClosedWithoutTensorMetrics(t *testing.T) {
	canonical, err := CanonicalizeRanks(&Graph{Nodes: []EinsumNode{
		{ID: "x", OutputRank: 1}, {ID: "y", Equation: "A->A", Inputs: []string{"x"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = ComputeSOLBounds(canonical, Ceiling{MACsPerCycle: 1, BytesPerCycle: 1, FrequencyGHz: 1, BytesPerElement: 2})
	if !errors.Is(err, ErrMissingSOLData) {
		t.Fatalf("error = %v", err)
	}
}

func TestSOLDifferential_MatchesSOLARTextFixtures(t *testing.T) {
	read := func(name string, dst any) {
		t.Helper()
		data, err := os.ReadFile(filepath.Join("testdata", "solar-mini", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(data, dst); err != nil {
			t.Fatalf("decode %s: %v", name, err)
		}
	}
	type tensorSet struct {
		Inputs  []string `json:"inputs"`
		Outputs []string `json:"outputs"`
	}
	type layer struct {
		Type        string `json:"type"`
		Equation    string `json:"einsum_equation"`
		Connections struct {
			Inputs []string `json:"inputs"`
		} `json:"connections"`
		TensorNames  tensorSet `json:"tensor_names"`
		TensorShapes struct {
			Inputs  [][]int `json:"inputs"`
			Outputs [][]int `json:"outputs"`
		} `json:"tensor_shapes"`
		TensorTypes struct {
			Inputs []string `json:"inputs"`
		} `json:"tensor_types"`
	}
	var solarGraph struct {
		LayerOrder []string         `json:"layer_order"`
		Layers     map[string]layer `json:"layers"`
	}
	var analysis struct {
		Layers map[string]struct {
			MACs     uint64 `json:"macs"`
			OtherOps uint64 `json:"other_ops"`
		} `json:"layers"`
		Total struct {
			MACs                    uint64 `json:"macs"`
			UnfusedElements         uint64 `json:"unfused_elements"`
			FusedElements           uint64 `json:"fused_elements"`
			FusedPrefetchedElements uint64 `json:"fused_prefetched_elements"`
		} `json:"total"`
	}
	var perf struct {
		Arch struct {
			FrequencyGHz  float64 `json:"freq_GHz"`
			BytesPerCycle float64 `json:"DRAM_byte_per_cycle"`
			MACsPerCycle  float64 `json:"MAC_per_cycle"`
		} `json:"arch"`
		Unfused struct {
			RuntimeMS float64 `json:"runtime_ms"`
		} `json:"unfused"`
		Fused struct {
			RuntimeMS float64 `json:"runtime_ms"`
		} `json:"fused"`
	}
	read("einsum_graph.yaml", &solarGraph)
	read("analysis.yaml", &analysis)
	read("perf_FixtureArch.yaml", &perf)
	var arch struct {
		Name          string  `json:"name"`
		FrequencyGHz  float64 `json:"freq_GHz"`
		BytesPerCycle float64 `json:"DRAM_byte_per_cycle"`
		MACsPerCycle  float64 `json:"MAC_per_cycle_fp16_tc"`
	}
	read(filepath.Join("configs", "arch", "FixtureArch.yaml"), &arch)
	if perf.Arch.FrequencyGHz != arch.FrequencyGHz || perf.Arch.BytesPerCycle != arch.BytesPerCycle || perf.Arch.MACsPerCycle != arch.MACsPerCycle {
		t.Fatal("perf fixture does not match architecture fixture")
	}
	graph := &Graph{}
	product := func(shape []int) uint64 {
		n := uint64(1)
		for _, d := range shape {
			n *= uint64(d)
		}
		return n
	}
	for _, id := range solarGraph.LayerOrder {
		layer := solarGraph.Layers[id]
		if layer.Type == "start" {
			if len(layer.TensorNames.Outputs) != 1 || len(layer.TensorShapes.Outputs) != 1 {
				t.Fatalf("start node %q fixture must have one tensor", id)
			}
			graph.Nodes = append(graph.Nodes, EinsumNode{ID: id, OutputRank: len(layer.TensorShapes.Outputs[0]), OutputTensors: []TensorInfo{{Name: layer.TensorNames.Outputs[0], Elements: product(layer.TensorShapes.Outputs[0])}}})
			continue
		}
		metrics, ok := analysis.Layers[id]
		if !ok {
			t.Fatalf("analysis fixture has no layer %q", id)
		}
		node := EinsumNode{ID: id, Equation: layer.Equation, Inputs: append([]string(nil), layer.Connections.Inputs...), MACs: metrics.MACs, OtherOps: metrics.OtherOps}
		for i, name := range layer.TensorNames.Inputs {
			if i >= len(layer.TensorShapes.Inputs) {
				t.Fatalf("layer %q input %d has no shape", id, i)
			}
			weight := i < len(layer.TensorTypes.Inputs) && layer.TensorTypes.Inputs[i] == "weight"
			node.InputTensors = append(node.InputTensors, TensorInfo{Name: name, Elements: product(layer.TensorShapes.Inputs[i]), Weight: weight})
		}
		for i, name := range layer.TensorNames.Outputs {
			if i >= len(layer.TensorShapes.Outputs) {
				t.Fatalf("layer %q output %d has no shape", id, i)
			}
			node.OutputTensors = append(node.OutputTensors, TensorInfo{Name: name, Elements: product(layer.TensorShapes.Outputs[i])})
		}
		graph.Nodes = append(graph.Nodes, node)
	}
	canonical, err := CanonicalizeRanks(graph)
	if err != nil {
		t.Fatal(err)
	}
	report, err := ComputeSOLBounds(canonical, Ceiling{MACsPerCycle: arch.MACsPerCycle, BytesPerCycle: arch.BytesPerCycle, FrequencyGHz: arch.FrequencyGHz, BytesPerElement: 2})
	if err != nil {
		t.Fatal(err)
	}
	if report.UnfusedBytes/2 != float64(analysis.Total.UnfusedElements) || report.FusedBytes/2 != float64(analysis.Total.FusedElements) || report.FusedPrefetchedBytes/2 != float64(analysis.Total.FusedPrefetchedElements) {
		t.Fatalf("SOL element counts disagree with analysis fixture: report=%+v analysis=%+v", report, analysis.Total)
	}
	if float64(analysis.Total.MACs)/arch.MACsPerCycle != 100 {
		t.Fatal("analysis fixture compute cycles do not match perf input")
	}
	if got := report.UnfusedWholeGraph * 1000; math.Abs(got-perf.Unfused.RuntimeMS) > 1e-15 {
		t.Fatalf("unfused runtime %.9g ms, fixture %.9g ms", got, perf.Unfused.RuntimeMS)
	}
	if got := report.FusedWholeGraph * 1000; math.Abs(got-perf.Fused.RuntimeMS) > 1e-15 {
		t.Fatalf("fused runtime %.9g ms, fixture %.9g ms", got, perf.Fused.RuntimeMS)
	}
	if report.Unfused < report.UnfusedWholeGraph || report.Fused < report.FusedWholeGraph {
		t.Fatalf("intended bounds violate SOLAR inequality: %+v", report)
	}
}
