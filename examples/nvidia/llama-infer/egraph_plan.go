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
// The embedding lookup and argmax are not expressed as terms; llama-infer's
// own kernels run them. Without attn, neither are RoPE, the KV-cache store
// and attention. With attn, the generator's direct emitters run them on a
// BF16 KV cache: [hopper.QKNormRoPEKVAppend] rotates q and k and appends k
// and v, qcast widens q to F32, and [hopper.SM120PagedAttention] attends.
// Those two kernels are emitted directly, not lowered from terms; the
// generator has no term for them.
type egraphPlans struct {
	attnIn, post, head *hopper.Plan

	attn  bool
	rope  hopper.QKNormRoPEKVAppend
	qcast *hopper.Plan
	sdpa  hopper.SM120PagedAttention
}

// egraphPageSize is the attention page size. The cache is contiguous: page
// i holds positions [16i, 16i+16).
const egraphPageSize = 16

// ops returns the ops the generator emits and the ops llama-infer's kernels
// still run.
func (p *egraphPlans) ops() (generated, fallback []string) {
	generated = []string{"rmsnorm", "qkv_proj(+bias)", "o_proj+residual", "rmsnorm", "gate_up_proj", "swiglu", "down_proj+residual", "final_rmsnorm", "lm_head"}
	if p.attn {
		generated = append(generated, "rope+kv_store(direct)", "q_cast", "attention(direct)")
		return generated, []string{"embedding", "argmax"}
	}
	return generated, []string{"embedding", "rope", "kv_store", "attention", "argmax"}
}

func buildEgraphPlans(cfg *ModelConfig, qkvBias, attn bool) (*egraphPlans, error) {
	h := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
	inter := cfg.IntermediateSize
	eps := cfg.RmsNormEps

	compile := func(name string, b *hopper.Builder, root egraph.Id, out hopper.DType) (*hopper.Plan, error) {
		e, err := hopper.Optimize(b.G, root)
		if err != nil {
			return nil, fmt.Errorf("optimize %s: %w", name, err)
		}
		p, err := hopper.Compile(e, name, out)
		if err != nil {
			return nil, err
		}
		return p, nil
	}

	p := egraphPlans{attn: attn}
	var err error
	qkvOut := hopper.F32
	if attn {
		// The rotary kernel reads the QKV projection and writes q and the
		// cache in one dtype, and the attention kernel reads a BF16 cache.
		qkvOut = hopper.BF16
	}

	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, 1, h)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("attn_norm", hopper.F32, h), h, eps))
	qkv := b.MatMul(norm, b.Input("w_qkv", hopper.BF16, qDim+2*kvDim, h))
	if qkvBias {
		qkv = b.Map(hopper.MustLambda("(add $0 $1)"), qkv, b.Input("b_qkv", hopper.F32, qDim+2*kvDim))
	}
	if p.attnIn, err = compile("llama_attn_in", b, qkv, qkvOut); err != nil {
		return nil, err
	}

	b = hopper.NewBuilder()
	ao := b.Cast(hopper.BF16, b.Input("attn", hopper.F32, 1, qDim))
	x = b.Input("x", hopper.F32, 1, h)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(ao, b.Input("w_o", hopper.BF16, h, qDim)), x)
	norm = b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*inter, h)))
	out := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, h, inter)), hid)
	if p.post, err = compile("llama_post", b, out, hopper.F32); err != nil {
		return nil, err
	}

	b = hopper.NewBuilder()
	x = b.Input("x", hopper.F32, 1, h)
	norm = b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("norm", hopper.F32, h), h, eps))
	logits := b.MatMul(norm, b.Input("lm_head", hopper.BF16, cfg.VocabSize, h))
	if p.head, err = compile("llama_head", b, logits, hopper.F32); err != nil {
		return nil, err
	}
	if !attn {
		return &p, nil
	}

	hd := cfg.HeadDim()
	p.rope = hopper.QKNormRoPEKVAppend{
		Name:     "llama_rope_kv",
		NumHeads: cfg.NumAttentionHeads,
		KVHeads:  cfg.NumKeyValueHeads,
		HeadDim:  hd,
		// The decoder offsets the cache pointers to the token's row, so
		// the kernel always writes slot 0 of page 0.
		PageSize: 1,
		MaxPages: 1,
		DType:    hopper.BF16,
	}
	b = hopper.NewBuilder()
	qf := b.Cast(hopper.F32, b.Input("q", hopper.BF16, 1, qDim))
	if p.qcast, err = compile("llama_qcast", b, qf, hopper.F32); err != nil {
		return nil, err
	}
	p.sdpa = hopper.SM120PagedAttention{
		Name:       "llama_sdpa",
		NumQHeads:  cfg.NumAttentionHeads,
		NumKVHeads: cfg.NumKeyValueHeads,
		HeadDim:    hd,
		PageSize:   egraphPageSize,
		MaxPages:   (cfg.MaxSeqLen + egraphPageSize - 1) / egraphPageSize,
		SeqLen:     cfg.MaxSeqLen,
		Target:     "sm_121a",
		KVFormat:   hopper.WeightBF16,
	}
	return &p, nil
}
