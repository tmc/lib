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

// egraphOptions selects the e-graph decoder's variants.
type egraphOptions struct {
	Mega bool // run each decode plan as one persistent megakernel
	Attn bool // generate RoPE, the KV-cache store and attention

	// PrefillRows, with Attn, prefills in chunks of this many rows: 0
	// prefills token by token, < 0 picks the cache length.
	PrefillRows int

	// AttnSplits divides decode attention's keys among this many CTAs per
	// KV head, with a combine kernel; 0 runs one CTA per KV head.
	AttnSplits int

	// Target is the nvcc -arch of the GPU the projection plans are for,
	// passed to [hopper.CompileFor]; "" lowers them for GB10. For sm_100a
	// and sm_103a the projections become tcgen05 [hopper.UMMAGEMM]s, which
	// the decoder cannot launch yet.
	Target string
}

// egraphPageSize is the attention page size. The cache is contiguous: page
// i holds positions [16i, 16i+16).
const egraphPageSize = 16

// ops returns the ops the generator emits and the ops llama-infer's kernels
// still run.
func (p *egraphPlans) ops() (generated, fallback []string) {
	generated = []string{"rmsnorm", "qkv_proj(+bias)", "o_proj+residual", "rmsnorm", "gate_up_proj", "swiglu", "down_proj+residual", "final_rmsnorm", "lm_head"}
	if p.attn {
		a := "attention (generated, direct emitter)"
		if p.sdpa.Splits > 0 {
			a = fmt.Sprintf("attention (generated, direct emitter: split-KV, %d splits per KV head, + combine)", p.sdpa.Splits)
		}
		generated = append(generated, "rope+kv_store (generated, direct emitter)", "q_cast", a)
		return generated, []string{"embedding", "argmax"}
	}
	return generated, []string{"embedding", "rope", "kv_store", "attention", "argmax"}
}

// egraphPrefillPlans are the plans of a batched prefill over up to rows
// prompt tokens. With more than 16 rows the generator lowers every
// projection to an mma.sync [hopper.TileGEMM]. The rotary kernel indexes
// the cache through the attention page table, so it writes each token's
// row without pointer offsets. Attention is one
// [hopper.MTPPagedAttention] launch per chunk: each CTA serves qtok
// consecutive rows of one KV head, attending the cache before its first
// row and, causally, its own rows.
type egraphPrefillPlans struct {
	rows       int
	attnIn     *hopper.Plan
	post       *hopper.Plan
	rope       hopper.QKNormRoPEKVAppend
	attn       hopper.MTPPagedAttention
	qtok       int
	cacheSlots int // KV cache rows, MaxSeqLen padded for a last partial group
}

// sm121SharedPerBlock is the shared memory one sm_121 block may use (opt
// in above 48 KiB), and staticSharedLimit the static shared memory any
// kernel may declare.
const (
	sm121SharedPerBlock = 99 * 1024
	staticSharedLimit   = 48 * 1024
)

// prefillQTokens returns the most query rows per group the warp
// MTPPagedAttention kernel takes: G·qtok query vectors may not exceed 64,
// and its static shared memory (G·qtok query rows and 16-row K and V tiles
// of FP32, and a mask word per row) must fit in 48 KiB.
func prefillQTokens(g, headDim int) int {
	q := 64 / g
	for q > 1 && 4*headDim*(g*q+2*16)+8*q > staticSharedLimit {
		q--
	}
	return q
}

// buildEgraphPlans builds the decode plans and, with opt.PrefillRows != 0,
// the prefill plans.
func buildEgraphPlans(cfg *ModelConfig, qkvBias bool, opt egraphOptions) (*egraphPlans, error) {
	attn, prefillRows := opt.Attn, opt.PrefillRows
	p := egraphPlans{attn: attn}
	var err error
	if p.attnIn, p.post, err = buildLayerPlans(cfg, qkvBias, attn, "llama", 1, hopper.F32, opt.Target); err != nil {
		return nil, err
	}
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, 1, cfg.HiddenSize)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("norm", hopper.F32, cfg.HiddenSize), cfg.HiddenSize, cfg.RmsNormEps))
	logits := b.MatMul(norm, b.Input("lm_head", hopper.BF16, cfg.VocabSize, cfg.HiddenSize))
	if p.head, err = compilePlan("llama_head", b, logits, hopper.F32, opt.Target); err != nil {
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
		Splits:     opt.AttnSplits,
	}
	if prefillRows == 0 {
		return &p, nil
	}
	// The tile gemms' M rows set the chunk: by default the whole cache,
	// so a prompt runs in one chunk and streams the weights once, in
	// multiples of the largest tile height whose shared memory fits.
	if prefillRows <= 0 {
		prefillRows = cfg.MaxSeqLen
	}
	pf := egraphPrefillPlans{qtok: prefillQTokens(cfg.NumAttentionHeads/cfg.NumKeyValueHeads, hd)}
	for _, bm := range []int{128, 64} {
		pf.rows = (prefillRows + bm - 1) / bm * bm
		if pf.attnIn, pf.post, err = buildLayerPlans(cfg, qkvBias, attn, "llama_pf", pf.rows, hopper.BF16, opt.Target); err != nil {
			return nil, err
		}
		fits := true
		for _, pl := range []*hopper.Plan{pf.attnIn, pf.post} {
			for _, k := range pl.Kernels {
				if k.Tile != nil {
					k.Tile.BM = bm
					fits = fits && k.Tile.SharedBytes() <= sm121SharedPerBlock
				}
			}
		}
		if fits {
			break
		}
	}
	if err := overridePrefillTiles(&pf, prefillRows); err != nil {
		return nil, err
	}
	// A last group of fewer than qtok rows still covers qtok positions;
	// pad the cache so they exist.
	pf.cacheSlots = (cfg.MaxSeqLen + pf.qtok - 1 + egraphPageSize - 1) / egraphPageSize * egraphPageSize
	pf.rope = p.rope
	pf.rope.Name = "llama_pf_rope_kv"
	pf.rope.PageSize, pf.rope.MaxPages = egraphPageSize, pf.cacheSlots/egraphPageSize
	pf.attn = hopper.MTPPagedAttention{
		Name:          "llama_pf_attn",
		NumHeads:      cfg.NumAttentionHeads,
		KVHeads:       cfg.NumKeyValueHeads,
		HeadDim:       hd,
		PageSize:      egraphPageSize,
		MaxPages:      pf.cacheSlots / egraphPageSize,
		QTokensPerSeq: pf.qtok,
	}
	p.pf = &pf
	return &p, nil
}

func compilePlan(name string, b *hopper.Builder, root egraph.Id, out hopper.DType, target string) (*hopper.Plan, error) {
	e, err := hopper.Optimize(b.G, root)
	if err != nil {
		return nil, fmt.Errorf("optimize %s: %w", name, err)
	}
	return hopper.CompileFor(e, name, out, target)
}

// buildLayerPlans builds a decoder layer's attnIn and post plans over m
// rows, named prefix_attn_in and prefix_post. The post plan reads the
// attention output in attnDType. The gemms are lowered for target, an nvcc
// -arch value or "".
func buildLayerPlans(cfg *ModelConfig, qkvBias, attn bool, prefix string, m int, attnDType hopper.DType, target string) (attnIn, post *hopper.Plan, err error) {
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
	if attnIn, err = compilePlan(prefix+"_attn_in", b, qkv, qkvOut, target); err != nil {
		return nil, nil, err
	}

	b = hopper.NewBuilder()
	ao := b.Input("attn", attnDType, m, qDim)
	if attnDType != hopper.BF16 {
		ao = b.Cast(hopper.BF16, ao)
	}
	x = b.Input("x", hopper.F32, m, h)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(ao, b.Input("w_o", hopper.BF16, h, qDim)), x)
	norm = b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*inter, h)))
	out := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, h, inter)), hid)
	if post, err = compilePlan(prefix+"_post", b, out, hopper.F32, target); err != nil {
		return nil, nil, err
	}
	return attnIn, post, nil
}

// buildQCastPlan builds the plan widening m rows of BF16 q to F32.
func buildQCastPlan(cfg *ModelConfig, name string, m int) (*hopper.Plan, error) {
	b := hopper.NewBuilder()
	qf := b.Cast(hopper.F32, b.Input("q", hopper.BF16, m, cfg.NumAttentionHeads*cfg.HeadDim()))
	return compilePlan(name, b, qf, hopper.F32, "")
}
