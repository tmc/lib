package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/safetensors"
)

// tinyCheckpoint writes a one-layer checkpoint to a new directory and returns
// it. edit may change the tensors before they are written.
func tinyCheckpoint(t *testing.T, edit func(map[string]safetensors.Tensor)) string {
	t.Helper()
	cfg := map[string]any{
		"hidden_size": 4, "intermediate_size": 6, "num_hidden_layers": 1,
		"num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 5,
	}
	const hidden, inter, qDim, kvDim, vocab = 4, 6, 4, 2, 5
	shapes := map[string]int{
		"model.embed_tokens.weight":                      vocab * hidden,
		"model.norm.weight":                              hidden,
		"model.layers.0.input_layernorm.weight":          hidden,
		"model.layers.0.post_attention_layernorm.weight": hidden,
		"model.layers.0.self_attn.q_proj.weight":         qDim * hidden,
		"model.layers.0.self_attn.k_proj.weight":         kvDim * hidden,
		"model.layers.0.self_attn.v_proj.weight":         kvDim * hidden,
		"model.layers.0.self_attn.o_proj.weight":         hidden * qDim,
		"model.layers.0.mlp.gate_proj.weight":            inter * hidden,
		"model.layers.0.mlp.up_proj.weight":              inter * hidden,
		"model.layers.0.mlp.down_proj.weight":            hidden * inter,
	}
	tensors := make(map[string]safetensors.Tensor)
	for name, n := range shapes {
		tensors[name] = bf16Tensor(t, name, n)
	}
	if edit != nil {
		edit(tensors)
	}
	dir := t.TempDir()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	var list []safetensors.Tensor
	for _, v := range tensors {
		list = append(list, v)
	}
	if err := safetensors.WriteFile(filepath.Join(dir, "model.safetensors"), list, nil); err != nil {
		t.Fatal(err)
	}
	return dir
}

func bf16Tensor(t *testing.T, name string, n int) safetensors.Tensor {
	t.Helper()
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i%7-3) / 8
	}
	data, err := safetensors.EncodeFloat32(safetensors.BF16, v)
	if err != nil {
		t.Fatal(err)
	}
	return safetensors.Tensor{Name: name, Dtype: safetensors.BF16, Shape: []int{n}, Data: data}
}

// setUntied rewrites dir's config.json with tie_word_embeddings false.
func setUntied(t *testing.T, dir string) {
	t.Helper()
	name := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg["tie_word_embeddings"] = false
	if data, err = json.Marshal(cfg); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(name, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

type weightLoader func(dir string) (*ModelWeights, *ModelConfig, error)

// loadFailures returns how load departs from the required behavior: the
// testing model only when no path is given, and an error naming the cause
// for every checkpoint that cannot be loaded.
func loadFailures(t *testing.T, load weightLoader) []string {
	var bad []string
	if w, cfg, err := load(""); err != nil || w == nil || cfg == nil {
		bad = append(bad, "no path: want testing model")
	}

	dir := tinyCheckpoint(t, nil)
	w, cfg, err := load(dir)
	switch {
	case err != nil:
		bad = append(bad, "valid checkpoint: "+err.Error())
	case cfg.HiddenSize != 4 || len(w.Layers) != 1:
		bad = append(bad, "valid checkpoint: loaded the testing model")
	case &w.LMHead[0] != &w.EmbedTokens[0]:
		bad = append(bad, "absent lm_head: not tied to embed_tokens")
	case w.Layers[0].QBias != nil:
		bad = append(bad, "absent bias: not nil")
	}

	// An explicit tie_word_embeddings false with an lm_head loads untied.
	untied := tinyCheckpoint(t, func(m map[string]safetensors.Tensor) {
		m["lm_head.weight"] = bf16Tensor(t, "lm_head.weight", 20)
	})
	setUntied(t, untied)
	if w, _, err := load(untied); err != nil {
		bad = append(bad, "untied checkpoint: "+err.Error())
	} else if &w.LMHead[0] == &w.EmbedTokens[0] {
		bad = append(bad, "untied checkpoint: lm_head tied")
	}
	untiedNoHead := tinyCheckpoint(t, nil)
	setUntied(t, untiedNoHead)

	truncated := tinyCheckpoint(t, nil)
	if err := os.Truncate(filepath.Join(truncated, "model.safetensors"), 6); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, dir, want string
	}{
		{"missing directory", filepath.Join(t.TempDir(), "absent"), "config.json"},
		{"untied without lm_head", untiedNoHead, "tie_word_embeddings is false"},
		{"truncated header", truncated, "safetensors"},
		{"missing layer tensor", tinyCheckpoint(t, func(m map[string]safetensors.Tensor) {
			delete(m, "model.layers.0.mlp.up_proj.weight")
		}), "missing tensor model.layers.0.mlp.up_proj.weight"},
		{"unknown dtype", tinyCheckpoint(t, func(m map[string]safetensors.Tensor) {
			x := m["model.norm.weight"]
			x.Dtype = safetensors.I16
			m["model.norm.weight"] = x
		}), `unsupported dtype "I16"`},
		{"wrong element count", tinyCheckpoint(t, func(m map[string]safetensors.Tensor) {
			m["model.layers.0.self_attn.k_proj.weight"] = bf16Tensor(t, "model.layers.0.self_attn.k_proj.weight", 7)
		}), "has 7 elements, want 8"},
		{"wrong bias count", tinyCheckpoint(t, func(m map[string]safetensors.Tensor) {
			m["model.layers.0.self_attn.v_proj.bias"] = bf16Tensor(t, "model.layers.0.self_attn.v_proj.bias", 3)
		}), "has 3 elements, want 2"},
	} {
		w, cfg, err := load(tt.dir)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			bad = append(bad, tt.name+": error does not name "+tt.want)
		}
		if w != nil || cfg != nil {
			bad = append(bad, tt.name+": returned a model with the error")
		}
	}
	return bad
}

func TestLoadWeights(t *testing.T) {
	for _, f := range loadFailures(t, loadWeights) {
		t.Error(f)
	}
}

// TestLoadWeightsRejectsFallback is a must-fail control: the behavior this
// change removed, falling back to the testing model when a checkpoint fails
// to load, must not satisfy loadFailures.
func TestLoadWeightsRejectsFallback(t *testing.T) {
	fallback := func(dir string) (*ModelWeights, *ModelConfig, error) {
		w, cfg, err := loadWeights(dir)
		if err != nil {
			return loadWeights("")
		}
		return w, cfg, nil
	}
	if bad := loadFailures(t, fallback); len(bad) == 0 {
		t.Fatal("fallback to the testing model not detected")
	}
}
