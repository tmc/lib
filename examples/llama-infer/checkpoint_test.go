package main

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func checkpointFixture() (*ModelConfig, map[string]checkpointTensor) {
	cfg := &ModelConfig{ModelType: "llama", HiddenSize: 4, IntermediateSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, ExplicitHeadDim: 2, VocabSize: 16, TieWordEmbeddings: true}
	found := make(map[string]checkpointTensor)
	for _, tensor := range expectedCheckpointTensors(cfg) {
		if tensor.name == "lm_head.weight" {
			continue
		}
		found[tensor.name] = checkpointTensor{DType: "BF16", Shape: append([]int(nil), tensor.shape...)}
	}
	return cfg, found
}

func TestAuditCheckpoint(t *testing.T) {
	cfg, found := checkpointFixture()
	receipt, err := auditCheckpoint(cfg, found, "F32")
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Verdict != "accepted" {
		t.Fatalf("verdict = %q", receipt.Verdict)
	}
	if got := receipt.Tensors[len(receipt.Tensors)-1].Name; got == "" {
		t.Fatal("empty tensor name")
	}
	first := append([]TensorConsumption(nil), receipt.Tensors...)
	again, err := auditCheckpoint(cfg, found, "F32")
	if err != nil || !reflect.DeepEqual(first, again.Tensors) {
		t.Fatal("receipt is not deterministic")
	}
}

func TestAuditCheckpointRejectsInvalidManifest(t *testing.T) {
	tests := []struct {
		name string
		edit func(map[string]checkpointTensor)
		want string
	}{
		{"missing", func(m map[string]checkpointTensor) { delete(m, "model.norm.weight") }, "model.norm.weight: missing"},
		{"extra", func(m map[string]checkpointTensor) {
			m["model.unused.weight"] = checkpointTensor{DType: "F32", Shape: []int{1}}
		}, "model.unused.weight: unused"},
		{"wrong shape", func(m map[string]checkpointTensor) {
			m["model.norm.weight"] = checkpointTensor{DType: "F32", Shape: []int{5}}
		}, "shape [5], want [4]"},
		{"wrong dtype", func(m map[string]checkpointTensor) {
			m["model.norm.weight"] = checkpointTensor{DType: "I32", Shape: []int{4}}
		}, "dtype I32"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg, found := checkpointFixture()
			test.edit(found)
			receipt, err := auditCheckpoint(cfg, found, "F32")
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
			if receipt.Verdict != "rejected" {
				t.Fatalf("verdict = %q", receipt.Verdict)
			}
			var checkpointErr *CheckpointError
			if !errors.As(err, &checkpointErr) || !reflect.DeepEqual(checkpointErr.Receipt, receipt) {
				t.Fatalf("error does not retain receipt: %v", err)
			}
		})
	}
}

func TestAuditCheckpointRequiresUntiedHead(t *testing.T) {
	cfg, found := checkpointFixture()
	cfg.TieWordEmbeddings = false
	_, err := auditCheckpoint(cfg, found, "F32")
	if err == nil || !strings.Contains(err.Error(), "lm_head.weight: missing") {
		t.Fatalf("error = %v", err)
	}
}
