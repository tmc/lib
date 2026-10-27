package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// egraphPlans are the decode-step kernels the e-graph generator
// (nvidia/x/egraph/hopper) builds for one token of a dense LLaMA model.
// Every layer shares the same plans; only the bound buffers change.
//
//	attnIn: qkv = cast_bf16(rmsnorm(x, attn_norm)) · w_qkvᵀ [+ b_qkv]
//	post:   h = cast_bf16(attn) · w_oᵀ + x
//	        out = swiglu(cast_bf16(rmsnorm(h, ffn_norm)) · w_guᵀ) · w_downᵀ + h
//	head:   logits = cast_bf16(rmsnorm(x, norm)) · lm_headᵀ
//
// RoPE, the KV-cache store, attention, the embedding lookup and argmax are
// not expressed as terms; llama-infer's own kernels run them.
type egraphPlans struct {
	attnIn, post, head *hopper.Plan
}

// egraphGenerated and egraphFallback name the ops each side computes.
var (
	egraphGenerated = []string{"rmsnorm", "qkv_proj(+bias)", "o_proj+residual", "rmsnorm", "gate_up_proj", "swiglu", "down_proj+residual", "final_rmsnorm", "lm_head"}
	egraphFallback  = []string{"embedding", "rope", "kv_store", "attention", "argmax"}
)

func buildEgraphPlans(cfg *ModelConfig, qkvBias bool) (*egraphPlans, error) {
	h := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
	inter := cfg.IntermediateSize
	eps := cfg.RmsNormEps

	compile := func(name string, b *hopper.Builder, root egraph.Id) (*hopper.Plan, error) {
		e, err := hopper.Optimize(b.G, root)
		if err != nil {
			return nil, fmt.Errorf("optimize %s: %w", name, err)
		}
		p, err := hopper.Compile(e, name, hopper.F32)
		if err != nil {
			return nil, err
		}
		return p, nil
	}

	var p egraphPlans
	var err error

	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, 1, h)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("attn_norm", hopper.F32, h), h, eps))
	qkv := b.MatMul(norm, b.Input("w_qkv", hopper.BF16, qDim+2*kvDim, h))
	if qkvBias {
		qkv = b.Map(hopper.MustLambda("(add $0 $1)"), qkv, b.Input("b_qkv", hopper.F32, qDim+2*kvDim))
	}
	if p.attnIn, err = compile("llama_attn_in", b, qkv); err != nil {
		return nil, err
	}

	b = hopper.NewBuilder()
	attn := b.Cast(hopper.BF16, b.Input("attn", hopper.F32, 1, qDim))
	x = b.Input("x", hopper.F32, 1, h)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(attn, b.Input("w_o", hopper.BF16, h, qDim)), x)
	norm = b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*inter, h)))
	out := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, h, inter)), hid)
	if p.post, err = compile("llama_post", b, out); err != nil {
		return nil, err
	}

	b = hopper.NewBuilder()
	x = b.Input("x", hopper.F32, 1, h)
	norm = b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("norm", hopper.F32, h), h, eps))
	logits := b.MatMul(norm, b.Input("lm_head", hopper.BF16, cfg.VocabSize, h))
	if p.head, err = compile("llama_head", b, logits); err != nil {
		return nil, err
	}
	return &p, nil
}
