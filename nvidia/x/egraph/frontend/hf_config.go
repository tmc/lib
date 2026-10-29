package frontend

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// An ArchFamily is a HuggingFace architectures[0] value.
type ArchFamily string

const (
	ArchLlama      ArchFamily = "LlamaForCausalLM"
	ArchQwen3      ArchFamily = "Qwen3ForCausalLM"
	ArchDeepseekV3 ArchFamily = "DeepseekV3ForCausalLM"
	ArchNemotronH  ArchFamily = "NemotronHForCausalLM"
)

// An HFModelConfig holds the config.json fields the importer reads.
type HFModelConfig struct {
	Architectures     []string `json:"architectures"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	HeadDim           int      `json:"head_dim"`
	VocabSize         int      `json:"vocab_size"`
	RMSNormEps        float64  `json:"rms_norm_eps"`
	LayerNormEpsilon  float64  `json:"layer_norm_epsilon"` // Nemotron-H
	RoPETheta         float64  `json:"rope_theta"`

	// DeepSeek-V3 multi-head latent attention and MoE.
	QLoRARank           int `json:"q_lora_rank"`
	KVLoRARank          int `json:"kv_lora_rank"`
	QKNoPEHeadDim       int `json:"qk_nope_head_dim"`
	QKRoPEHeadDim       int `json:"qk_rope_head_dim"`
	VHeadDim            int `json:"v_head_dim"`
	NRoutedExperts      int `json:"n_routed_experts"`
	NSharedExperts      int `json:"n_shared_experts"`
	NumExpertsPerTok    int `json:"num_experts_per_tok"`
	MoEIntermediateSize int `json:"moe_intermediate_size"`
	FirstKDenseReplace  int `json:"first_k_dense_replace"`

	// Nemotron-H hybrid layout: M (Mamba-2), * (attention), - (MLP).
	HybridOverridePattern string `json:"hybrid_override_pattern"`
}

// Family returns the architecture family named by Architectures[0],
// matched without regard to case.
func (c HFModelConfig) Family() (ArchFamily, error) {
	if len(c.Architectures) == 0 {
		return "", fmt.Errorf("config has no architectures")
	}
	for _, f := range []ArchFamily{ArchLlama, ArchQwen3, ArchDeepseekV3, ArchNemotronH} {
		if strings.EqualFold(c.Architectures[0], string(f)) {
			return f, nil
		}
	}
	return "", fmt.Errorf("unsupported architecture %q", c.Architectures[0])
}

// ParseHFConfig parses and checks a config.json payload, filling
// head_dim, num_key_value_heads and rms_norm_eps defaults.
func ParseHFConfig(configJSON []byte) (HFModelConfig, error) {
	var c HFModelConfig
	if err := json.Unmarshal(configJSON, &c); err != nil {
		return c, fmt.Errorf("parse hf config: %w", err)
	}
	fam, err := c.Family()
	if err != nil {
		return c, fmt.Errorf("parse hf config: %w", err)
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 {
		return c, fmt.Errorf("parse hf config: hidden_size, num_hidden_layers and num_attention_heads must be positive")
	}
	if c.IntermediateSize <= 0 && fam != ArchDeepseekV3 {
		return c, fmt.Errorf("parse hf config: intermediate_size must be positive")
	}
	if c.NumKeyValueHeads == 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.NumKeyValueHeads < 0 || c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return c, fmt.Errorf("parse hf config: %d key-value heads do not divide %d heads", c.NumKeyValueHeads, c.NumAttentionHeads)
	}
	if c.HeadDim == 0 {
		if c.HiddenSize%c.NumAttentionHeads != 0 {
			return c, fmt.Errorf("parse hf config: hidden_size %d is not a multiple of %d heads", c.HiddenSize, c.NumAttentionHeads)
		}
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.RMSNormEps == 0 {
		c.RMSNormEps = c.LayerNormEpsilon
	}
	if c.RMSNormEps <= 0 {
		return c, fmt.Errorf("parse hf config: rms_norm_eps must be positive")
	}
	switch fam {
	case ArchDeepseekV3:
		if c.KVLoRARank <= 0 || c.QKNoPEHeadDim <= 0 || c.QKRoPEHeadDim <= 0 || c.VHeadDim <= 0 {
			return c, fmt.Errorf("parse hf config: DeepSeek-V3 needs kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim and v_head_dim")
		}
		if c.FirstKDenseReplace == 0 && (c.MoEIntermediateSize <= 0 || c.NSharedExperts <= 0) {
			return c, fmt.Errorf("parse hf config: MoE layer 0 needs moe_intermediate_size and n_shared_experts")
		}
		if c.FirstKDenseReplace > 0 && c.IntermediateSize <= 0 {
			return c, fmt.Errorf("parse hf config: dense layer 0 needs intermediate_size")
		}
	case ArchNemotronH:
		if !strings.Contains(c.HybridOverridePattern, "-") {
			return c, fmt.Errorf("parse hf config: hybrid_override_pattern %q has no MLP layer", c.HybridOverridePattern)
		}
		if len(c.HybridOverridePattern) != c.NumHiddenLayers {
			return c, fmt.Errorf("parse hf config: hybrid_override_pattern has %d layers, want %d", len(c.HybridOverridePattern), c.NumHiddenLayers)
		}
	}
	return c, nil
}

// ImportHFConfig parses a config.json payload and lowers one decoder
// layer's MLP sub-block for batchSize·seqLen tokens:
//
//	out = h + down(act(rmsnorm(h)·gate, rmsnorm(h)·up))
//
// with h the BF16 input "hidden_states" [batchSize·seqLen, hidden_size],
// act SwiGLU (silu(g)·u) for Llama, Qwen3 and DeepSeek-V3, and squared ReLU
// of up alone for Nemotron-H. The layer is 0, except for Nemotron-H where it
// is the first MLP layer of the hybrid pattern; for a DeepSeek-V3 layer 0
// that is MoE, the shared experts are lowered and the routed experts are
// listed as unlowered. WeightMap holds every weight of the layer.
func ImportHFConfig(configJSON []byte, batchSize, seqLen int) (HFModelConfig, ImportResult, error) {
	c, err := ParseHFConfig(configJSON)
	if err != nil {
		return c, ImportResult{}, err
	}
	if batchSize <= 0 || seqLen <= 0 {
		return c, ImportResult{}, fmt.Errorf("import hf config: batch %d and sequence %d must be positive", batchSize, seqLen)
	}
	fam, _ := c.Family()
	m, d := batchSize*seqLen, c.HiddenSize
	wm := make(map[string][]int)
	var unlowered []string
	layer, prefix := 0, "model.layers.0."
	norm, gate, up, down := prefix+"post_attention_layernorm.weight", prefix+"mlp.gate_proj.weight", prefix+"mlp.up_proj.weight", prefix+"mlp.down_proj.weight"
	inter := c.IntermediateSize
	switch fam {
	case ArchLlama, ArchQwen3:
		h, kv, hd := c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim
		wm[prefix+"input_layernorm.weight"] = []int{d}
		wm[prefix+"self_attn.q_proj.weight"] = []int{h * hd, d}
		wm[prefix+"self_attn.k_proj.weight"] = []int{kv * hd, d}
		wm[prefix+"self_attn.v_proj.weight"] = []int{kv * hd, d}
		wm[prefix+"self_attn.o_proj.weight"] = []int{d, h * hd}
		if fam == ArchQwen3 {
			wm[prefix+"self_attn.q_norm.weight"] = []int{hd}
			wm[prefix+"self_attn.k_norm.weight"] = []int{hd}
		}
		unlowered = append(unlowered, "self_attn: input_layernorm, q/k/v/o projections, RoPE and softmax(QKᵀ)V")
	case ArchDeepseekV3:
		h := c.NumAttentionHeads
		qk := c.QKNoPEHeadDim + c.QKRoPEHeadDim
		wm[prefix+"input_layernorm.weight"] = []int{d}
		if c.QLoRARank > 0 {
			wm[prefix+"self_attn.q_a_proj.weight"] = []int{c.QLoRARank, d}
			wm[prefix+"self_attn.q_a_layernorm.weight"] = []int{c.QLoRARank}
			wm[prefix+"self_attn.q_b_proj.weight"] = []int{h * qk, c.QLoRARank}
		} else {
			wm[prefix+"self_attn.q_proj.weight"] = []int{h * qk, d}
		}
		wm[prefix+"self_attn.kv_a_proj_with_mqa.weight"] = []int{c.KVLoRARank + c.QKRoPEHeadDim, d}
		wm[prefix+"self_attn.kv_a_layernorm.weight"] = []int{c.KVLoRARank}
		wm[prefix+"self_attn.kv_b_proj.weight"] = []int{h * (c.QKNoPEHeadDim + c.VHeadDim), c.KVLoRARank}
		wm[prefix+"self_attn.o_proj.weight"] = []int{d, h * c.VHeadDim}
		unlowered = append(unlowered, "self_attn: multi-head latent attention")
		if c.FirstKDenseReplace == 0 {
			inter = c.NSharedExperts * c.MoEIntermediateSize
			gate, up, down = prefix+"mlp.shared_experts.gate_proj.weight", prefix+"mlp.shared_experts.up_proj.weight", prefix+"mlp.shared_experts.down_proj.weight"
			wm[prefix+"mlp.gate.weight"] = []int{c.NRoutedExperts, d}
			for e := range c.NRoutedExperts {
				ep := fmt.Sprintf("%smlp.experts.%d.", prefix, e)
				wm[ep+"gate_proj.weight"] = []int{c.MoEIntermediateSize, d}
				wm[ep+"up_proj.weight"] = []int{c.MoEIntermediateSize, d}
				wm[ep+"down_proj.weight"] = []int{d, c.MoEIntermediateSize}
			}
			unlowered = append(unlowered, fmt.Sprintf("mlp: router and %d routed experts (top %d)", c.NRoutedExperts, c.NumExpertsPerTok))
		}
	case ArchNemotronH:
		layer = strings.Index(c.HybridOverridePattern, "-")
		prefix = fmt.Sprintf("backbone.layers.%d.", layer)
		norm, gate, up, down = prefix+"norm.weight", "", prefix+"mixer.up_proj.weight", prefix+"mixer.down_proj.weight"
	}
	wm[norm] = []int{d}
	if gate != "" {
		wm[gate] = []int{inter, d}
	}
	wm[up] = []int{inter, d}
	wm[down] = []int{d, inter}

	l := newLowerer()
	h := l.input("hidden_states", hopper.BF16, []int{m, d})
	var p value
	n, err := l.rmsNorm(h, l.weight(norm, hopper.BF16, []int{d}), float32(c.RMSNormEps))
	if err == nil {
		n, err = l.convert(n, hopper.BF16)
	}
	if err != nil {
		return c, ImportResult{}, fmt.Errorf("import hf config: %w", err)
	}
	uv, err := l.matmul(n, l.weight(up, hopper.BF16, []int{inter, d}), false)
	if err != nil {
		return c, ImportResult{}, fmt.Errorf("import hf config: %w", err)
	}
	if gate != "" {
		var g, s, a value
		g, err = l.matmul(n, l.weight(gate, hopper.BF16, []int{inter, d}), false)
		if err == nil {
			s, err = l.unary("sigmoid", g)
		}
		if err == nil {
			a, err = l.binary("mul", g, s)
		}
		if err == nil {
			p, err = l.binary("mul", a, uv)
		}
	} else {
		var r value
		r, err = l.binary("max", uv, l.constant(0, hopper.F32))
		if err == nil {
			p, err = l.binary("mul", r, r)
		}
	}
	var out value
	if err == nil {
		p, err = l.convert(p, hopper.BF16)
	}
	if err == nil {
		out, err = l.matmul(p, l.weight(down, hopper.BF16, []int{d, inter}), false)
	}
	if err == nil {
		var h32 value
		if h32, err = l.convert(h, hopper.F32); err == nil {
			out, err = l.binary("add", out, h32)
		}
	}
	if err != nil {
		return c, ImportResult{}, fmt.Errorf("import hf config: %w", err)
	}
	res, err := l.finish(out)
	if err != nil {
		return c, ImportResult{}, fmt.Errorf("import hf config: %w", err)
	}
	res.WeightMap = wm
	res.Unlowered = unlowered
	return c, res, nil
}
