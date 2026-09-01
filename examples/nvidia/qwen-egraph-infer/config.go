package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// QwenConfig holds architecture hyperparameters for Qwen and LLaMA model families.
type QwenConfig struct {
	HiddenSize        int     `json:"hidden_size"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	VocabSize         int     `json:"vocab_size"`
	RmsNormEps        float32 `json:"rms_norm_eps"`
	RopeTheta         float32 `json:"rope_theta"`
	MaxSeqLen         int     `json:"max_position_embeddings"`
	HeadDimVal        int     `json:"head_dim"`

	// Nested text_config for Qwen3_5 / Qwen3.8 multimodal models
	TextConfig *QwenConfig `json:"text_config,omitempty"`
}

// LoadConfig reads and parses a model's config.json.
func LoadConfig(path string) (*QwenConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var raw QwenConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("unmarshal config: %w", err)
	}

	var cfg *QwenConfig
	if raw.TextConfig != nil && raw.TextConfig.HiddenSize > 0 {
		cfg = raw.TextConfig
	} else {
		cfg = &raw
	}

	if cfg.RmsNormEps == 0 {
		cfg.RmsNormEps = 1e-6
	}
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 1000000.0
	}
	if cfg.NumKeyValueHeads == 0 {
		cfg.NumKeyValueHeads = cfg.NumAttentionHeads
	}
	if cfg.MaxSeqLen == 0 || cfg.MaxSeqLen > 4096 {
		cfg.MaxSeqLen = 4096
	}

	return cfg, nil
}

// HeadDim returns the dimension per attention head.
func (cfg *QwenConfig) HeadDim() int {
	if cfg.HeadDimVal > 0 {
		return cfg.HeadDimVal
	}
	if cfg.NumAttentionHeads == 0 {
		return 128
	}
	return cfg.HiddenSize / cfg.NumAttentionHeads
}

// DefaultQwenConfig provides a compact testing configuration.
func DefaultQwenConfig() QwenConfig {
	return QwenConfig{
		HiddenSize:        1024,
		IntermediateSize:  3072,
		NumHiddenLayers:   2,
		NumAttentionHeads: 16,
		NumKeyValueHeads:  8,
		VocabSize:         151936,
		RmsNormEps:        1e-6,
		RopeTheta:         1000000.0,
		MaxSeqLen:         2048,
		HeadDimVal:        128,
	}
}
