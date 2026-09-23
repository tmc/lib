package main

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

type checkpointTensor struct {
	DType string
	Shape []int
	Shard string
}

// TensorConsumption records how one checkpoint tensor is used.
type TensorConsumption struct {
	Name       string `json:"name"`
	DType      string `json:"dtype,omitempty"`
	Shape      []int  `json:"shape,omitempty"`
	Shard      string `json:"shard,omitempty"`
	Action     string `json:"action"`
	TargetType string `json:"target_dtype,omitempty"`
	Source     string `json:"source,omitempty"`
	Reason     string `json:"reason,omitempty"`
}

// CheckpointReceipt is a deterministic account of every tensor in a checkpoint.
type CheckpointReceipt struct {
	SchemaVersion int                 `json:"schema_version"`
	ModelType     string              `json:"model_type"`
	TargetType    string              `json:"target_dtype"`
	Tensors       []TensorConsumption `json:"tensors"`
	Verdict       string              `json:"verdict"`
}

// CheckpointError reports a rejected checkpoint and retains its full receipt.
type CheckpointError struct {
	Receipt  CheckpointReceipt
	problems []string
}

func (e *CheckpointError) Error() string {
	return "checkpoint: " + strings.Join(e.problems, "; ")
}

type expectedTensor struct {
	name     string
	shape    []int
	required bool
}

func expectedCheckpointTensors(cfg *ModelConfig) []expectedTensor {
	headDim := cfg.HeadDim()
	qDim := cfg.NumAttentionHeads * headDim
	kvDim := cfg.NumKeyValueHeads * headDim
	want := []expectedTensor{
		{name: "model.embed_tokens.weight", shape: []int{cfg.VocabSize, cfg.HiddenSize}, required: true},
		{name: "model.norm.weight", shape: []int{cfg.HiddenSize}, required: true},
		{name: "lm_head.weight", shape: []int{cfg.VocabSize, cfg.HiddenSize}, required: !cfg.TieWordEmbeddings},
	}
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		want = append(want,
			expectedTensor{name: prefix + "input_layernorm.weight", shape: []int{cfg.HiddenSize}, required: true},
			expectedTensor{name: prefix + "post_attention_layernorm.weight", shape: []int{cfg.HiddenSize}, required: true},
			expectedTensor{name: prefix + "self_attn.q_proj.weight", shape: []int{qDim, cfg.HiddenSize}, required: true},
			expectedTensor{name: prefix + "self_attn.k_proj.weight", shape: []int{kvDim, cfg.HiddenSize}, required: true},
			expectedTensor{name: prefix + "self_attn.v_proj.weight", shape: []int{kvDim, cfg.HiddenSize}, required: true},
			expectedTensor{name: prefix + "self_attn.o_proj.weight", shape: []int{cfg.HiddenSize, qDim}, required: true},
			expectedTensor{name: prefix + "mlp.gate_proj.weight", shape: []int{cfg.IntermediateSize, cfg.HiddenSize}, required: true},
			expectedTensor{name: prefix + "mlp.up_proj.weight", shape: []int{cfg.IntermediateSize, cfg.HiddenSize}, required: true},
			expectedTensor{name: prefix + "mlp.down_proj.weight", shape: []int{cfg.HiddenSize, cfg.IntermediateSize}, required: true},
		)
		if strings.HasPrefix(cfg.ModelType, "qwen") {
			want = append(want,
				expectedTensor{name: prefix + "self_attn.q_proj.bias", shape: []int{qDim}, required: true},
				expectedTensor{name: prefix + "self_attn.k_proj.bias", shape: []int{kvDim}, required: true},
				expectedTensor{name: prefix + "self_attn.v_proj.bias", shape: []int{kvDim}, required: true},
			)
		}
	}
	sort.Slice(want, func(i, j int) bool { return want[i].name < want[j].name })
	return want
}

func auditCheckpoint(cfg *ModelConfig, found map[string]checkpointTensor, targetType string) (CheckpointReceipt, error) {
	receipt := CheckpointReceipt{SchemaVersion: 1, ModelType: cfg.ModelType, TargetType: targetType}
	want := expectedCheckpointTensors(cfg)
	expected := make(map[string]expectedTensor, len(want))
	var problems []string
	for _, tensor := range want {
		expected[tensor.name] = tensor
		got, ok := found[tensor.name]
		if !ok {
			if tensor.name == "lm_head.weight" && cfg.TieWordEmbeddings {
				receipt.Tensors = append(receipt.Tensors, TensorConsumption{Name: tensor.name, Action: "tied", TargetType: targetType, Source: "model.embed_tokens.weight", Reason: "config tie_word_embeddings=true"})
				continue
			}
			if tensor.required {
				receipt.Tensors = append(receipt.Tensors, TensorConsumption{Name: tensor.name, Shape: tensor.shape, Action: "missing", TargetType: targetType})
				problems = append(problems, tensor.name+": missing")
			}
			continue
		}
		entry := TensorConsumption{Name: tensor.name, DType: got.DType, Shape: append([]int(nil), got.Shape...), Shard: got.Shard, Action: "loaded", TargetType: targetType}
		if !slices.Equal(got.Shape, tensor.shape) {
			entry.Action = "wrong_shape"
			entry.Reason = fmt.Sprintf("want %v", tensor.shape)
			problems = append(problems, fmt.Sprintf("%s: shape %v, want %v", tensor.name, got.Shape, tensor.shape))
		} else if got.DType != "F16" && got.DType != "BF16" && got.DType != "F32" {
			entry.Action = "wrong_dtype"
			entry.Reason = "want F16, BF16, or F32"
			problems = append(problems, fmt.Sprintf("%s: dtype %s", tensor.name, got.DType))
		} else if (targetType == "F32" && got.DType != "F32") || (targetType == "F16" && got.DType != "F16") {
			entry.Action = "converted"
		}
		receipt.Tensors = append(receipt.Tensors, entry)
	}
	for name, got := range found {
		if _, ok := expected[name]; ok {
			continue
		}
		receipt.Tensors = append(receipt.Tensors, TensorConsumption{Name: name, DType: got.DType, Shape: append([]int(nil), got.Shape...), Shard: got.Shard, Action: "unused", Reason: "not consumed by the configured model"})
		problems = append(problems, name+": unused")
	}
	sort.Slice(receipt.Tensors, func(i, j int) bool { return receipt.Tensors[i].Name < receipt.Tensors[j].Name })
	if len(problems) != 0 {
		sort.Strings(problems)
		receipt.Verdict = "rejected"
		return receipt, &CheckpointError{Receipt: receipt, problems: problems}
	}
	receipt.Verdict = "accepted"
	return receipt, nil
}
