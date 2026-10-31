package main

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
)

// afmoeConfig describes the selected Trinity-Mini checkpoint. Parsing this
// contract does not enable the decoder: checkArchitecture still rejects afmoe.
type afmoeConfig struct {
	ModelConfig
	DType                  string   `json:"dtype"`
	LayerTypes             []string `json:"layer_types"`
	SlidingWindow          int      `json:"sliding_window"`
	GlobalAttnEveryNLayers int      `json:"global_attn_every_n_layers"`
	NumDenseLayers         int      `json:"num_dense_layers"`
	MoEIntermediateSize    int      `json:"moe_intermediate_size"`
	NumExperts             int      `json:"num_experts"`
	NumExpertsPerTok       int      `json:"num_experts_per_tok"`
	NumSharedExperts       int      `json:"num_shared_experts"`
	NumExpertGroups        int      `json:"num_expert_groups"`
	NumLimitedGroups       int      `json:"num_limited_groups"`
	NGroup                 int      `json:"n_group"`
	TopKGroup              int      `json:"topk_group"`
	MupEnabled             bool     `json:"mup_enabled"`
	RouteNorm              bool     `json:"route_norm"`
	RouteScale             float32  `json:"route_scale"`
	ScoreFunc              string   `json:"score_func"`
	HiddenAct              string   `json:"hidden_act"`
	TieWordEmbeddings      bool     `json:"tie_word_embeddings"`
	AttentionDropout       float32  `json:"attention_dropout"`
}

func parseAfmoeConfig(data []byte) (*afmoeConfig, error) {
	var c afmoeConfig
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("afmoe config: %w", err)
	}
	var extra map[string]json.RawMessage
	if err := json.Unmarshal(data, &extra); err != nil {
		return nil, fmt.Errorf("afmoe config: %w", err)
	}
	if c.ModelType != "afmoe" || c.DType != "bfloat16" || c.HiddenAct != "silu" || c.TieWordEmbeddings || c.AttentionDropout != 0 {
		return nil, fmt.Errorf("afmoe config: requires untied BF16 afmoe with silu and no dropout")
	}
	if _, ok := extra["quantization_config"]; ok {
		return nil, fmt.Errorf("afmoe config: quantized weights are not supported")
	}
	if raw, ok := extra["rope_scaling"]; ok && string(raw) != "null" {
		return nil, fmt.Errorf("afmoe config: scaled RoPE is not supported")
	}
	for _, key := range []string{"attention_bias", "mlp_bias"} {
		if raw, ok := extra[key]; ok && string(raw) != "false" {
			return nil, fmt.Errorf("afmoe config: unsupported %s", key)
		}
	}
	// Restrict the first adapter to the pinned Mini architecture. In
	// particular its query width is 4096, not hidden_size (2048).
	if c.HiddenSize != 2048 || c.IntermediateSize != 6144 || c.NumHiddenLayers != 32 || c.NumAttentionHeads != 32 || c.NumKeyValueHeads != 4 || c.ExplicitHeadDim != 128 || c.VocabSize != 200192 {
		return nil, fmt.Errorf("afmoe config: unsupported architecture (requires Trinity-Mini)")
	}
	if c.MaxSeqLen != 131072 || c.RopeTheta != 10000 || c.RmsNormEps != 1e-5 || !c.MupEnabled {
		return nil, fmt.Errorf("afmoe config: unsupported context, RoPE, norm or muP settings")
	}
	if c.NumDenseLayers != 2 || c.MoEIntermediateSize != 1024 || c.NumExperts != 128 || c.NumExpertsPerTok != 8 || c.NumSharedExperts != 1 {
		return nil, fmt.Errorf("afmoe config: unsupported dense or expert layout")
	}
	if c.ScoreFunc != "sigmoid" || !c.RouteNorm || c.RouteScale != float32(2.826) || c.NumExpertGroups != 1 || c.NumLimitedGroups != 1 || c.NGroup != 1 || c.TopKGroup != 1 {
		return nil, fmt.Errorf("afmoe config: unsupported routing settings")
	}
	if c.SlidingWindow != 2048 || c.GlobalAttnEveryNLayers != 4 || len(c.LayerTypes) != c.NumHiddenLayers {
		return nil, fmt.Errorf("afmoe config: unsupported attention layout")
	}
	for i, kind := range c.LayerTypes {
		want := "sliding_attention"
		if (i+1)%4 == 0 {
			want = "full_attention"
		}
		if kind != want {
			return nil, fmt.Errorf("afmoe config: layer %d is %q, want %q", i, kind, want)
		}
	}
	return &c, nil
}

// afmoeTensor is an expected name and row-major shape, not an observation
// of a checkpoint header's dtype or shape.
type afmoeTensor struct {
	Name  string
	Shape []int
}

func (c *afmoeConfig) tensors() []afmoeTensor {
	var out []afmoeTensor
	add := func(name string, shape ...int) { out = append(out, afmoeTensor{name, shape}) }
	h, q, kv := c.HiddenSize, c.NumAttentionHeads*c.HeadDim(), c.NumKeyValueHeads*c.HeadDim()
	add("model.embed_tokens.weight", c.VocabSize, h)
	add("lm_head.weight", c.VocabSize, h)
	add("model.norm.weight", h)
	for layer := range c.NumHiddenLayers {
		pre := fmt.Sprintf("model.layers.%d.", layer)
		for _, norm := range []string{"input_layernorm", "post_attention_layernorm", "pre_mlp_layernorm", "post_mlp_layernorm"} {
			add(pre+norm+".weight", h)
		}
		for _, p := range []struct {
			name string
			n, k int
		}{{"q_proj", q, h}, {"k_proj", kv, h}, {"v_proj", kv, h}, {"o_proj", h, q}, {"gate_proj", q, h}} {
			add(pre+"self_attn."+p.name+".weight", p.n, p.k)
		}
		add(pre+"self_attn.q_norm.weight", c.HeadDim())
		add(pre+"self_attn.k_norm.weight", c.HeadDim())
		mlp := func(prefix string, width int) {
			add(pre+"mlp."+prefix+"gate_proj.weight", width, h)
			add(pre+"mlp."+prefix+"up_proj.weight", width, h)
			add(pre+"mlp."+prefix+"down_proj.weight", h, width)
		}
		if layer < c.NumDenseLayers {
			mlp("", c.IntermediateSize)
			continue
		}
		add(pre+"mlp.router.gate.weight", c.NumExperts, h)
		add(pre+"mlp.expert_bias", c.NumExperts)
		mlp("shared_experts.", c.NumSharedExperts*c.MoEIntermediateSize)
		for expert := range c.NumExperts {
			mlp(fmt.Sprintf("experts.%d.", expert), c.MoEIntermediateSize)
		}
	}
	slices.SortFunc(out, func(a, b afmoeTensor) int {
		if a.Name < b.Name {
			return -1
		}
		if a.Name > b.Name {
			return 1
		}
		return 0
	})
	return out
}

func afmoeFinite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
