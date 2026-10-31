package main

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

// The fixture comes from an independent scalar translation of the pinned HF
// forward, not these helpers. It specifies scalar arithmetic, not GPU parity.
func TestAFMoEMoEFixture(t *testing.T) {
	type tensor struct {
		Shape []int
		Data  []float32
	}
	var fixture struct {
		Router struct {
			Logits      tensor
			Bias        tensor
			TopK        int `json:"top_k"`
			Epsilon     float32
			RouteScale  float32 `json:"route_scale"`
			SelectedIDs []int   `json:"selected_ids"`
			Weights     tensor
		}
		MoE struct {
			Shared      tensor
			Outputs     tensor `json:"expert_outputs"`
			SelectedIDs []int  `json:"selected_ids"`
			Weights     tensor
			Weighted    tensor `json:"weighted_contributions"`
			Sequential  tensor `json:"sequential_bf16_expected"`
			Wide        tensor `json:"f32_accumulation_expected"`
		} `json:"moe_combine"`
	}
	data, err := os.ReadFile("testdata/afmoe_moe.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	r := fixture.Router
	ids, weights, err := afmoeRoute(r.Logits.Data, r.Bias.Data, r.TopK, true, r.Epsilon, r.RouteScale)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, r.SelectedIDs) {
		t.Fatalf("selected experts %v, want %v", ids, r.SelectedIDs)
	}
	for i, weight := range weights {
		if math.Abs(float64(weight-r.Weights.Data[i])) > 1e-7 {
			t.Errorf("weight %d = %g, want %g", i, weight, r.Weights.Data[i])
		}
	}
	m := fixture.MoE
	n := m.Outputs.Shape[1]
	outputs := make([][]float32, len(m.SelectedIDs))
	for i, id := range m.SelectedIDs {
		outputs[i] = m.Outputs.Data[id*n : (id+1)*n]
	}
	weighted, err := afmoeWeightedExperts(outputs, m.Weights.Data)
	if err != nil {
		t.Fatal(err)
	}
	for i := range weighted {
		if !reflect.DeepEqual(weighted[i], m.Weighted.Data[i*n:(i+1)*n]) {
			t.Errorf("weighted route %d = %v, want %v", i, weighted[i], m.Weighted.Data[i*n:(i+1)*n])
		}
	}
	for _, test := range []struct {
		name string
		wide bool
		want []float32
	}{
		{"sequential BF16", false, m.Sequential.Data},
		{"wide float32", true, m.Wide.Data},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := afmoeCombineExperts(m.Shared.Data, weighted, m.SelectedIDs, test.wide)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("combined %v, want %v", got, test.want)
			}
		})
	}
	if reflect.DeepEqual(m.Sequential.Data, m.Wide.Data) {
		t.Fatal("fixture does not distinguish accumulation policies")
	}
}

func TestAFMoERoute(t *testing.T) {
	for _, test := range []struct {
		name      string
		logits    []float32
		bias      []float32
		k         int
		normalize bool
		want      []int
		weights   []float32
	}{
		{"bias selects only", []float32{0, 1}, []float32{1, 0}, 1, false, []int{0}, []float32{0.5}},
		{"all experts", make([]float32, 128), nil, 8, true, []int{0, 1, 2, 3, 4, 5, 6, 7}, []float32{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125}},
		{"highest expert addressable", []float32{0, 0, 0}, []float32{0, 0, 1}, 1, true, []int{2}, []float32{1}},
	} {
		t.Run(test.name, func(t *testing.T) {
			ids, weights, err := afmoeRoute(test.logits, test.bias, test.k, test.normalize, 1e-20, 1)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(ids, test.want) || !reflect.DeepEqual(weights, test.weights) {
				t.Errorf("route = %v/%v, want %v/%v", ids, weights, test.want, test.weights)
			}
		})
	}
	logits, bias := make([]float32, 128), make([]float32, 128)
	bias[127] = 1
	ids, _, err := afmoeRoute(logits, bias, 8, true, 1e-20, 2.826)
	if err != nil || ids[0] != 127 {
		t.Fatalf("expert127 = %v, %v", ids, err)
	}
}

func TestAFMoEMoERejectsInvalidInputs(t *testing.T) {
	nan := float32(math.NaN())
	for _, test := range []struct {
		name   string
		logits []float32
		bias   []float32
		k      int
		eps    float32
		scale  float32
	}{
		{"empty", nil, nil, 1, 1e-20, 1},
		{"top-k zero", []float32{0}, nil, 0, 1e-20, 1},
		{"top-k too large", []float32{0}, nil, 2, 1e-20, 1},
		{"bias length", []float32{0}, []float32{}, 1, 1e-20, 1},
		{"non-finite logit", []float32{nan}, nil, 1, 1e-20, 1},
		{"non-finite bias", []float32{0}, []float32{nan}, 1, 1e-20, 1},
		{"negative epsilon", []float32{0}, nil, 1, -1, 1},
		{"non-finite scale", []float32{0}, nil, 1, 1e-20, nan},
		{"zero denominator", []float32{-1000}, nil, 1, 0, 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := afmoeRoute(test.logits, test.bias, test.k, true, test.eps, test.scale); err == nil {
				t.Fatal("accepted invalid routing input")
			}
		})
	}
	if _, err := afmoeWeightedExperts([][]float32{{1}, {1, 2}}, []float32{1, 1}); err == nil {
		t.Error("accepted mismatched expert widths")
	}
	if _, err := afmoeWeightedExperts([][]float32{{nan}}, []float32{1}); err == nil {
		t.Error("accepted non-finite expert output")
	}
	if _, err := afmoeCombineExperts([]float32{1}, [][]float32{{1}, {2}}, []int{0, 0}, false); err == nil {
		t.Error("accepted repeated expert ID")
	}
	if _, err := afmoeCombineExperts([]float32{1}, [][]float32{{1, 2}}, []int{0}, false); err == nil {
		t.Error("accepted mismatched contribution width")
	}
}
