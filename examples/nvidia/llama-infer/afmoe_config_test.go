package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
)

func afmoeMiniFixture(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/afmoe-mini/config.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestAfmoeMiniContract(t *testing.T) {
	c, err := parseAfmoeConfig(afmoeMiniFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if c.HeadDim()*c.NumAttentionHeads != 4096 || c.HiddenSize != 2048 {
		t.Fatal("Mini query width must not be inferred from hidden size")
	}
	var pin struct {
		TensorCount       int    `json:"tensor_count"`
		SortedNamesSHA256 string `json:"sorted_names_sha256"`
		MetadataTotalSize int64  `json:"metadata_total_size"`
	}
	b, err := os.ReadFile("testdata/afmoe-mini/index-contract.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &pin); err != nil {
		t.Fatal(err)
	}
	var names []string
	var bytes int64
	for _, tensor := range c.tensors() {
		n := int64(1)
		for _, d := range tensor.Shape {
			if d < 1 {
				t.Fatalf("%s: invalid shape %v", tensor.Name, tensor.Shape)
			}
			n *= int64(d)
		}
		bytes += n * 2 // Mini BF16 expectation, not an observed header dtype.
		names = append(names, tensor.Name)
	}
	if len(names) != pin.TensorCount {
		t.Fatalf("tensor count %d, want %d", len(names), pin.TensorCount)
	}
	if !slices.IsSorted(names) {
		t.Fatal("tensor contract is not sorted")
	}
	hash := sha256.Sum256([]byte(strings.Join(names, "\n") + "\n"))
	if hex.EncodeToString(hash[:]) != pin.SortedNamesSHA256 || bytes != pin.MetadataTotalSize {
		t.Fatalf("contract mismatch: names sha=%x, BF16 bytes=%d", hash, bytes)
	}
	if err := checkArchitecture("afmoe", afmoeMiniFixture(t), c.MaxSeqLen); err == nil {
		t.Fatal("schema preparation must not open the runtime architecture guard")
	}
}

func TestAfmoeMiniRejectsUnsupported(t *testing.T) {
	tests := []struct {
		name string
		key  string
		val  any
	}{
		{"nano", "hidden_size", 1024},
		{"head_width", "head_dim", 64},
		{"route_scale", "route_scale", 1},
		{"route_score", "score_func", "softmax"},
		{"route_norm", "route_norm", false},
		{"expert_count", "num_experts", 64},
		{"shared_count", "num_shared_experts", 0},
		{"window", "sliding_window", 4096},
		{"mup", "mup_enabled", false},
		{"rope", "rope_scaling", map[string]any{"factor": 2}},
		{"quantized", "quantization_config", nil},
		{"bias", "attention_bias", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var obj map[string]any
			if err := json.Unmarshal(afmoeMiniFixture(t), &obj); err != nil {
				t.Fatal(err)
			}
			obj[tt.key] = tt.val
			b, err := json.Marshal(obj)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseAfmoeConfig(b); err == nil {
				t.Fatal("accepted unsupported semantics")
			}
		})
	}
}
