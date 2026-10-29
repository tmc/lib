package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadModelConfigArchitecture(t *testing.T) {
	const base = `"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
		"num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 256,
		"max_position_embeddings": 4096`
	tests := []struct {
		name    string
		extra   string
		wantErr string // "" for success
	}{
		{"llama", `"model_type": "llama", "hidden_act": "silu", "attention_bias": false, "mlp_bias": false`, ""},
		{"mistral v0.1 window covers context", `"model_type": "mistral", "sliding_window": 4096`, ""},
		{"mistral null window", `"model_type": "mistral", "sliding_window": null`, ""},
		{"qwen2 window off", `"model_type": "qwen2", "sliding_window": 32768, "use_sliding_window": false`, ""},
		{"no model_type", `"hidden_act": "silu"`, ""},
		{"qwen3", `"model_type": "qwen3"`, `model_type "qwen3"`},
		{"gelu", `"model_type": "llama", "hidden_act": "gelu"`, `hidden_act "gelu"`},
		{"attention bias", `"model_type": "llama", "attention_bias": true`, "attention_bias"},
		{"mlp bias", `"model_type": "llama", "mlp_bias": true`, "mlp_bias"},
		{"short window", `"model_type": "mistral", "sliding_window": 1024`, "sliding_window 1024"},
		{"text_config qwen3", `"model_type": "x", "text_config": {"model_type": "qwen3", ` + base + `}`, `model_type "qwen3"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte("{"+base+", "+tt.extra+"}"), 0o644); err != nil {
				t.Fatal(err)
			}
			_, _, err := loadModelConfig(dir)
			switch {
			case tt.wantErr == "" && err != nil:
				t.Errorf("loadModelConfig: %v", err)
			case tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)):
				t.Errorf("loadModelConfig: err = %v, want %q", err, tt.wantErr)
			}
		})
	}
}
