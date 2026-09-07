package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
)

// pjrtCheckpoint keeps checkpoint-specific generation metadata separate from
// command-line defaults. Weights are widened explicitly to f32 after validation.
type pjrtCheckpoint struct {
	config  ModelConfig
	weights *ModelWeights
	bos     *int32
	theta   float64
	eos     []int32
}

// readPJRTConfig accepts the unscaled, full-attention LLaMA and Qwen2 variants.
// Unknown fields are rejected: silently ignoring a new architecture option can
// produce a valid graph for the wrong model. The listed metadata has no effect
// on this inference graph. Defaults follow Transformers v4.57.1 configurations.
func readPJRTConfig(data []byte) (*pjrtCheckpoint, error) {
	var raw struct {
		ModelConfig
		HiddenAct        *string         `json:"hidden_act"`
		RopeScaling      json.RawMessage `json:"rope_scaling"`
		AttentionBias    bool            `json:"attention_bias"`
		MLPBias          bool            `json:"mlp_bias"`
		UseMRoPE         bool            `json:"use_mrope"`
		Sliding          bool            `json:"use_sliding_window"`
		SlidingWindow    *int            `json:"sliding_window"`
		MaxWindowLayers  int             `json:"max_window_layers"`
		LayerTypes       []string        `json:"layer_types"`
		BOS              *int32          `json:"bos_token_id"`
		KVHeads          *int            `json:"num_key_value_heads"`
		Theta            *float64        `json:"rope_theta"`
		Epsilon          *float32        `json:"rms_norm_eps"`
		PretrainingTP    *int            `json:"pretraining_tp"`
		Architectures    []string        `json:"architectures"`
		Name             string          `json:"_name_or_path"`
		Version          string          `json:"transformers_version"`
		TorchDType       string          `json:"torch_dtype"`
		DType            string          `json:"dtype"`
		UseCache         bool            `json:"use_cache"`
		Dropout          float64         `json:"attention_dropout"`
		InitializerRange float64         `json:"initializer_range"`
		PadTokenID       *int32          `json:"pad_token_id"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&raw); err != nil {
		return nil, fmt.Errorf("pjrt config: %w", err)
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return nil, fmt.Errorf("pjrt config: trailing JSON")
	}
	c := raw.ModelConfig
	if c.ModelType != "llama" && c.ModelType != "qwen2" {
		return nil, fmt.Errorf("pjrt config: unsupported model_type %q", c.ModelType)
	}
	if c.TextConfig != nil {
		return nil, fmt.Errorf("pjrt config: nested text_config is unsupported")
	}
	if raw.HiddenAct != nil && *raw.HiddenAct != "silu" {
		return nil, fmt.Errorf("pjrt config: unsupported hidden_act %q", *raw.HiddenAct)
	}
	if len(raw.RopeScaling) > 0 && string(raw.RopeScaling) != "null" {
		return nil, fmt.Errorf("pjrt config: rope_scaling is unsupported")
	}
	if raw.UseMRoPE {
		return nil, fmt.Errorf("pjrt config: use_mrope is unsupported")
	}
	if raw.AttentionBias || raw.MLPBias {
		return nil, fmt.Errorf("pjrt config: output or MLP bias is unsupported")
	}
	if raw.Sliding || (c.ModelType == "llama" && raw.SlidingWindow != nil) {
		return nil, fmt.Errorf("pjrt config: sliding attention is unsupported")
	}
	if raw.LayerTypes != nil && len(raw.LayerTypes) != c.NumHiddenLayers {
		return nil, fmt.Errorf("pjrt config: layer_types count mismatch")
	}
	for _, kind := range raw.LayerTypes {
		if kind != "full_attention" {
			return nil, fmt.Errorf("pjrt config: unsupported layer type %q", kind)
		}
	}
	if raw.PretrainingTP != nil && *raw.PretrainingTP != 1 {
		return nil, fmt.Errorf("pjrt config: pretraining_tp must be one")
	}
	c.NumKeyValueHeads = c.NumAttentionHeads
	if raw.KVHeads != nil {
		c.NumKeyValueHeads = *raw.KVHeads
	} else if c.ModelType == "qwen2" {
		return nil, fmt.Errorf("pjrt config: qwen2 requires num_key_value_heads")
	}
	c.RmsNormEps = 1e-6
	if raw.Epsilon != nil {
		c.RmsNormEps = *raw.Epsilon
	}
	theta := 10000.0
	if raw.Theta != nil {
		theta = *raw.Theta
	}
	c.RopeTheta = float32(theta)

	for _, n := range []int{c.HiddenSize, c.IntermediateSize, c.NumHiddenLayers, c.NumAttentionHeads, c.NumKeyValueHeads, c.VocabSize, c.MaxSeqLen} {
		if n <= 0 {
			return nil, fmt.Errorf("pjrt config: dimensions must be positive")
		}
	}
	if int64(c.VocabSize)-1 > math.MaxInt32 {
		return nil, fmt.Errorf("pjrt config: vocabulary exceeds int32 token IDs")
	}
	if c.ExplicitHeadDim < 0 || (c.ExplicitHeadDim == 0 && c.HiddenSize%c.NumAttentionHeads != 0) || c.HeadDim()%2 != 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return nil, fmt.Errorf("pjrt config: incompatible attention heads")
	}
	if c.RmsNormEps <= 0 || math.IsInf(float64(c.RmsNormEps), 0) || math.IsNaN(float64(c.RmsNormEps)) || theta <= 0 || math.IsInf(theta, 0) || math.IsNaN(theta) {
		return nil, fmt.Errorf("pjrt config: epsilon and theta must be finite and positive")
	}
	// The metadata audit multiplies dimensions using int. Check each tensor's
	// f32 byte count before it constructs any expected tensor shapes.
	h, q, d, k := c.HiddenSize, c.NumAttentionHeads, c.HeadDim(), c.NumKeyValueHeads
	product := func(dims ...int) error {
		n := 4
		for _, v := range dims {
			if v > math.MaxInt/n {
				return fmt.Errorf("pjrt config: tensor size overflows")
			}
			n *= v
		}
		return nil
	}
	for _, shape := range [][]int{{c.VocabSize, h}, {q, d, h}, {k, d, h}, {h, c.IntermediateSize}, {c.NumHiddenLayers, 2, c.MaxSeqLen, k, d}} {
		if err := product(shape...); err != nil {
			return nil, err
		}
	}
	p := &pjrtCheckpoint{config: c, bos: raw.BOS, theta: theta}
	var err error
	p.eos, err = pjrtEOS(c.RawEOSTokenID, c.VocabSize)
	if err != nil {
		return nil, err
	}

	if p.bos != nil && (*p.bos < 0 || int64(*p.bos) >= int64(c.VocabSize)) {
		return nil, fmt.Errorf("pjrt config: BOS outside vocabulary")
	}
	p.config.EOSTokenIDs = append([]int32(nil), p.eos...)
	if len(p.eos) > 0 {
		p.config.EOSTokenID = p.eos[0]
	}
	return p, nil
}

func pjrtEOS(raw json.RawMessage, vocab int) ([]int32, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var ids []int32
	var id int32
	if err := json.Unmarshal(raw, &id); err == nil {
		ids = []int32{id}
	} else if err := json.Unmarshal(raw, &ids); err != nil {
		return nil, fmt.Errorf("pjrt config: invalid eos_token_id")
	}
	for _, id := range ids {
		if id < 0 || int64(id) >= int64(vocab) {
			return nil, fmt.Errorf("pjrt config: EOS outside vocabulary")
		}
	}
	return ids, nil
}

func loadPJRTCheckpoint(dir string, capacity int) (*pjrtCheckpoint, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("pjrt config: %w", err)
	}
	p, err := readPJRTConfig(data)
	if err != nil {
		return nil, err
	}
	if len(p.eos) == 0 || p.bos == nil {
		data, err := os.ReadFile(filepath.Join(dir, "generation_config.json"))
		if err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("pjrt generation config: %w", err)
		}
		if err == nil {
			var raw struct {
				EOS json.RawMessage `json:"eos_token_id"`
				BOS *int32          `json:"bos_token_id"`
			}
			if err := json.Unmarshal(data, &raw); err != nil {
				return nil, fmt.Errorf("pjrt generation config: %w", err)
			}
			if len(p.eos) == 0 {
				p.eos, err = pjrtEOS(raw.EOS, p.config.VocabSize)
				if err != nil {
					return nil, err
				}
			}
			if p.bos == nil {
				p.bos = raw.BOS
			}
			if p.bos != nil && (*p.bos < 0 || int64(*p.bos) >= int64(p.config.VocabSize)) {
				return nil, fmt.Errorf("pjrt generation config: BOS outside vocabulary")
			}
			p.config.EOSTokenIDs = append([]int32(nil), p.eos...)
			if len(p.eos) > 0 {
				p.config.EOSTokenID = p.eos[0]
			}
		}
	}
	if capacity < 0 {
		return nil, fmt.Errorf("pjrt: negative capacity")
	}
	if capacity > 0 && capacity < p.config.MaxSeqLen {
		p.config.MaxSeqLen = capacity
	}
	retained, _, err := loadRetainedModel(dir, &p.config)
	if err != nil {
		return nil, err
	}
	p.weights = retained.Float32()
	return p, nil
}

func preparePJRTModel(cfg Config) (*pjrtCheckpoint, Tokenizer, error) {
	if cfg.Precision != "" && cfg.Precision != "f32" {
		return nil, nil, fmt.Errorf("pjrt: only f32 execution is supported")
	}
	if cfg.ModelPath != "" {
		p, err := loadPJRTCheckpoint(cfg.ModelPath, cfg.MaxSeqLen)
		return p, nil, err
	}
	if !cfg.TestMode {
		return nil, nil, fmt.Errorf("pjrt: model path required unless TestMode is enabled")
	}
	c := ModelConfig{ModelType: "llama", HiddenSize: cfg.Dim, IntermediateSize: cfg.HiddenDim, NumHiddenLayers: cfg.NumLayers, NumAttentionHeads: cfg.NumHeads, NumKeyValueHeads: cfg.NumKVHeads, ExplicitHeadDim: cfg.HeadDim, VocabSize: cfg.VocabSize, RmsNormEps: cfg.Eps, RopeTheta: cfg.RopeTheta, MaxSeqLen: cfg.MaxSeqLen}
	data, err := json.Marshal(c)
	if err != nil {
		return nil, nil, err
	}
	p, err := readPJRTConfig(data)
	if err != nil {
		return nil, nil, err
	}
	if p.config.NumAttentionHeads*p.config.HeadDim() != p.config.HiddenSize {
		return nil, nil, fmt.Errorf("pjrt: synthetic weights require query width equal to hidden size")
	}
	if p.config.VocabSize != 256 {
		return nil, nil, fmt.Errorf("pjrt: byte tokenizer requires exactly 256 vocabulary entries")
	}
	p.weights = GenerateDeterministicWeights(&p.config)
	return p, NewByteTokenizer(), nil
}
