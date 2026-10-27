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

	pf *egraphPrefillPlans // with prefill
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

// egraphPrefillPlans are the plans of a batched prefill over up to rows
// prompt tokens. With more than 16 rows the generator lowers every
// projection to an mma.sync [hopper.TileGEMM]. The rotary kernel indexes
// the cache through the attention page table, so it writes each token's
// row without pointer offsets.
type egraphPrefillPlans struct {
	rows                int
	attnIn, post, qcast *hopper.Plan
	rope                hopper.QKNormRoPEKVAppend
}

// egraphPrefillRows is the most prompt tokens one prefill pass runs; a
// longer prompt runs in chunks.
const egraphPrefillRows = 128

func buildEgraphPlans(cfg *ModelConfig, qkvBias, attn, prefill bool) (*egraphPlans, error) {
	p := egraphPlans{attn: attn}
	var err error
	if p.attnIn, p.post, err = buildLayerPlans(cfg, qkvBias, attn, "llama", 1); err != nil {
		return nil, err
	}
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, 1, cfg.HiddenSize)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("norm", hopper.F32, cfg.HiddenSize), cfg.HiddenSize, cfg.RmsNormEps))
	logits := b.MatMul(norm, b.Input("lm_head", hopper.BF16, cfg.VocabSize, cfg.HiddenSize))
	if p.head, err = compilePlan("llama_head", b, logits, hopper.F32); err != nil {
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
	if p.qcast, err = buildQCastPlan(cfg, "llama_qcast", 1); err != nil {
		return nil, err
	}
	maxPages := (cfg.MaxSeqLen + egraphPageSize - 1) / egraphPageSize
	p.sdpa = hopper.SM120PagedAttention{
		Name:       "llama_sdpa",
		NumQHeads:  cfg.NumAttentionHeads,
		NumKVHeads: cfg.NumKeyValueHeads,
		HeadDim:    hd,
		PageSize:   egraphPageSize,
		MaxPages:   maxPages,
		SeqLen:     cfg.MaxSeqLen,
		Target:     "sm_121a",
		KVFormat:   hopper.WeightBF16,
	}
	if !prefill {
		return &p, nil
	}
	pf := egraphPrefillPlans{rows: egraphPrefillRows}
	if pf.attnIn, pf.post, err = buildLayerPlans(cfg, qkvBias, attn, "llama_pf", pf.rows); err != nil {
		return nil, err
	}
	// The default 128 × 128 tile with three stages needs 108 KiB of
	// shared memory, more than sm_121's 99 KiB per block; 64-row tiles
	// need 81 KiB and waste fewer rows on short prompts.
	for _, pl := range []*hopper.Plan{pf.attnIn, pf.post} {
		for _, k := range pl.Kernels {
			if k.Tile != nil {
				k.Tile.BM = 64
			}
		}
	}
	if pf.qcast, err = buildQCastPlan(cfg, "llama_pf_qcast", pf.rows); err != nil {
		return nil, err
	}
	pf.rope = p.rope
	pf.rope.Name = "llama_pf_rope_kv"
	pf.rope.PageSize, pf.rope.MaxPages = egraphPageSize, maxPages
	p.pf = &pf
	return &p, nil
}

func compilePlan(name string, b *hopper.Builder, root egraph.Id, out hopper.DType) (*hopper.Plan, error) {
	e, err := hopper.Optimize(b.G, root)
	if err != nil {
		return nil, fmt.Errorf("optimize %s: %w", name, err)
	}
	return hopper.Compile(e, name, out)
}

// buildLayerPlans builds a decoder layer's attnIn and post plans over m
// rows, named prefix_attn_in and prefix_post.
func buildLayerPlans(cfg *ModelConfig, qkvBias, attn bool, prefix string, m int) (attnIn, post *hopper.Plan, err error) {
	h := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
	inter := cfg.IntermediateSize
	eps := cfg.RmsNormEps

	qkvOut := hopper.F32
	if attn {
		// The rotary kernel reads the QKV projection and writes q and the
		// cache in one dtype, and the attention kernel reads a BF16 cache.
		qkvOut = hopper.BF16
	}
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, m, h)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("attn_norm", hopper.F32, h), h, eps))
	qkv := b.MatMul(norm, b.Input("w_qkv", hopper.BF16, qDim+2*kvDim, h))
	if qkvBias {
		qkv = b.Map(hopper.MustLambda("(add $0 $1)"), qkv, b.Input("b_qkv", hopper.F32, qDim+2*kvDim))
	}
	if attnIn, err = compilePlan(prefix+"_attn_in", b, qkv, qkvOut); err != nil {
		return nil, nil, err
	}

	b = hopper.NewBuilder()
	ao := b.Cast(hopper.BF16, b.Input("attn", hopper.F32, m, qDim))
	x = b.Input("x", hopper.F32, m, h)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(ao, b.Input("w_o", hopper.BF16, h, qDim)), x)
	norm = b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*inter, h)))
	out := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, h, inter)), hid)
	if post, err = compilePlan(prefix+"_post", b, out, hopper.F32); err != nil {
		return nil, nil, err
	}
	return attnIn, post, nil
}

// buildQCastPlan builds the plan widening m rows of BF16 q to F32.
func buildQCastPlan(cfg *ModelConfig, name string, m int) (*hopper.Plan, error) {
	b := hopper.NewBuilder()
	qf := b.Cast(hopper.F32, b.Input("q", hopper.BF16, m, cfg.NumAttentionHeads*cfg.HeadDim()))
	return compilePlan(name, b, qf, hopper.F32)
}
