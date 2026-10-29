package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// egraphPlans are the decode-step kernels the e-graph generator
// (nvidia/x/egraph/hopper) builds for one token of a dense LLaMA model.
// Every layer shares the same plans; only the bound buffers change.
//
//	norm:   xn = cast_bf16(rmsnorm(x, norm))
//	attnIn: qkv = xn · w_qkvᵀ [+ b_qkv]
//	post:   h = cast_bf16(attn) · w_oᵀ + x
//	        res = swiglu(cast_bf16(rmsnorm(h, ffn_norm)) · w_guᵀ) · w_downᵀ + h
//	        xn = cast_bf16(rmsnorm(res, next_norm))
//	head:   logits = xn · lm_headᵀ
//
// Each layer's post normalizes its residual res for the next layer's
// attnIn, with next_norm the next layer's attention norm, or the final
// norm after the last layer; res is a temporary of post (postRes) that the
// decoder binds to the residual buffer. norm normalizes the embedding for
// the first layer.
//
// The embedding lookup and argmax are not expressed as terms; llama-infer's
// own kernels run them. Without attn, neither are RoPE, the KV-cache store
// and attention. With attn, the generator's direct emitters run them on a
// BF16 KV cache: [hopper.QKNormRoPEKVAppend] rotates q and k and appends k
// and v, and [hopper.SM120PagedAttention] attends, reading q in BF16.
// Those two kernels are emitted directly, not lowered from terms; the
// generator has no term for them.
//
// On targets without tcgen05 every decode kernel but norm is a skinny
// gemm: the casts of attn and of the SwiGLU product fold into the gemms
// that read them ([hopper.SkinnyGEMM] X and GLU), and post's two RMSNorms
// run as the o_proj and down_proj gemms' tails ([hopper.Plan.FuseTails]).
type egraphPlans struct {
	norm, attnIn, post, head *hopper.Plan
	postRes                  string

	attn bool
	rope hopper.QKNormRoPEKVAppend
	sdpa hopper.SM120PagedAttention

	pf *egraphPrefillPlans // with prefill
}

// egraphOptions selects the e-graph decoder's variants.
type egraphOptions struct {
	// Profile is the GPU the plans are built for. Its Target is the PTX
	// target the kernels are emitted for, and its shared memory limits
	// and SM count choose tile heights and attention splits.
	Profile cudadevicetable.Profile

	Mega bool // run each decode plan as one persistent megakernel
	Attn bool // generate RoPE, the KV-cache store and attention

	// PDL launches each generated kernel as the programmatic dependent of
	// the kernel before it. The caller sets it only for sm_90 and newer.
	PDL bool

	// PrefillRows, with Attn, prefills in chunks of this many rows: 0
	// prefills token by token, < 0 picks the cache length.
	PrefillRows int

	// AttnSplits divides decode attention's keys among this many CTAs per
	// KV head, with a combine kernel; 0 runs one CTA per KV head, and < 0
	// picks Profile's SMs / KV heads, so the CTAs cover the SMs once.
	AttnSplits int

	// SMs is the device's multiprocessor count, which the prefill skinny
	// mmas split K to fill; 0 means hopper's default.
	SMs int
}

// egraphPageSize is the attention page size. The cache is contiguous: page
// i holds positions [16i, 16i+16).
const egraphPageSize = 16

// ops returns the ops the generator emits and the ops llama-infer's kernels
// still run.
func (p *egraphPlans) ops() (generated, fallback []string) {
	generated = []string{"rmsnorm", "qkv_proj(+bias)", "o_proj+residual", "rmsnorm", "gate_up_proj", "swiglu", "down_proj+residual", "final_rmsnorm", "lm_head"}
	if p.attn {
		a := "attention (generated, direct emitter, bf16 q)"
		if p.sdpa.Splits > 0 {
			a = fmt.Sprintf("attention (generated, direct emitter: split-KV, %d splits per KV head, + combine)", p.sdpa.Splits)
		}
		generated = append(generated, "rope+kv_store (generated, direct emitter)", a)
		return generated, []string{"embedding", "argmax"}
	}
	return generated, []string{"embedding", "rope", "kv_store", "attention", "argmax"}
}

// egraphPrefillPlans are the plans of a batched prefill over up to rows
// prompt tokens. With more than 16 rows the generator lowers every
// projection to an mma.sync [hopper.TileGEMM]. A chunk of at most narrow
// rows runs narrowIn and narrowPost instead, whose projections are
// [hopper.SkinnyMMA]s: with so few rows a projection costs what streaming
// its weights costs, and a skinny mma streams them as the decode GEMV
// does, where a 64-row tile gemm does not. The rotary kernel indexes
// the cache through the attention page table, so it writes each token's
// row without pointer offsets. Attention is one
// [hopper.MTPPagedAttention] launch per chunk: each CTA serves qtok
// consecutive rows of one KV head, attending the cache before its first
// row and, causally, its own rows. A narrow chunk's attention, narrowAttn,
// serves narrowQTok rows per CTA.
type egraphPrefillPlans struct {
	rows       int
	attnIn     *hopper.Plan
	post       *hopper.Plan
	narrow     int
	narrowIn   *hopper.Plan
	narrowPost *hopper.Plan
	rope       hopper.QKNormRoPEKVAppend
	attn       hopper.MTPPagedAttention
	qtok       int
	narrowAttn hopper.MTPPagedAttention
	narrowQTok int
	cacheSlots int // KV cache rows, MaxSeqLen padded for a last partial group
}

// prefillQTokens returns the most query rows per group the warp
// MTPPagedAttention kernel takes: G·qtok query vectors may not exceed 64,
// and its static shared memory (G·qtok query rows and 16-row K and V tiles
// of FP32, and a mask word per row) must fit in staticShared bytes.
func prefillQTokens(g, headDim, staticShared int) int {
	q := 64 / g
	for q > 1 && 4*headDim*(g*q+2*16)+8*q > staticShared {
		q--
	}
	return q
}

// buildEgraphPlans builds the decode plans and, with opt.PrefillRows != 0,
// the prefill plans, for opt.Profile.
func buildEgraphPlans(cfg *ModelConfig, qkvBias bool, opt egraphOptions) (*egraphPlans, error) {
	prof := opt.Profile
	if prof.Target == "" || prof.SMs.Value < 1 || prof.SharedPerBlock.Value < 1 || prof.SharedPerBlockOptin.Value < 1 {
		return nil, fmt.Errorf("e-graph plans need a target profile with SMs and shared memory limits, have %+v", prof)
	}
	attn, prefillRows := opt.Attn, opt.PrefillRows
	p := egraphPlans{attn: attn}
	var err error
	if p.attnIn, p.post, err = buildLayerPlans(cfg, qkvBias, attn, "llama", 1, hopper.F32, prof.Target, true); err != nil {
		return nil, err
	}
	if p.postRes, err = normInput(p.post); err != nil {
		return nil, err
	}
	h := cfg.HiddenSize
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, 1, h)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("norm", hopper.F32, h), h, cfg.RmsNormEps))
	if p.norm, err = compilePlan("llama_norm", b, norm, hopper.BF16, prof.Target); err != nil {
		return nil, err
	}
	b = hopper.NewBuilder()
	logits := b.MatMul(b.Input("xn", hopper.BF16, 1, h), b.Input("lm_head", hopper.BF16, cfg.VocabSize, h))
	if p.head, err = compilePlan("llama_head", b, logits, hopper.F32, prof.Target); err != nil {
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
	maxPages := (cfg.MaxSeqLen + egraphPageSize - 1) / egraphPageSize
	p.sdpa = hopper.SM120PagedAttention{
		Name:       "llama_sdpa",
		NumQHeads:  cfg.NumAttentionHeads,
		NumKVHeads: cfg.NumKeyValueHeads,
		HeadDim:    hd,
		PageSize:   egraphPageSize,
		MaxPages:   maxPages,
		SeqLen:     cfg.MaxSeqLen,
		Target:     prof.Target,
		KVFormat:   hopper.WeightBF16,
		Splits:     opt.AttnSplits,
		Q:          hopper.BF16,
	}
	if p.sdpa.Splits < 0 {
		p.sdpa.Splits = attnSplits(prof, cfg.NumKeyValueHeads)
	}
	if prefillRows == 0 {
		return &p, nil
	}
	// The tile gemms' M rows set the chunk: by default the whole cache,
	// so a prompt runs in one chunk and streams the weights once, in
	// multiples of the tallest tile, at the deepest pipeline, whose shared
	// memory fits the profile.
	if prefillRows <= 0 {
		prefillRows = cfg.MaxSeqLen
	}
	smem := int(prof.SharedPerBlockOptin.Value)
	pf := egraphPrefillPlans{qtok: prefillQTokens(cfg.NumAttentionHeads/cfg.NumKeyValueHeads, hd, int(prof.SharedPerBlock.Value))}
	fits := false
	for _, tile := range prefillTiles {
		pf.rows = (prefillRows + tile.bm - 1) / tile.bm * tile.bm
		if pf.attnIn, pf.post, err = buildLayerPlans(cfg, qkvBias, attn, "llama_pf", pf.rows, hopper.BF16, prof.Target, false); err != nil {
			return nil, err
		}
		fits = true
		for _, pl := range []*hopper.Plan{pf.attnIn, pf.post} {
			for _, k := range pl.Kernels {
				if k.Tile != nil {
					k.Tile.BM, k.Tile.Stages = tile.bm, tile.stages
					if tile.swizzle {
						k.Tile.Swizzle128B = true
					}
					fits = fits && k.Tile.SharedBytes() <= smem
				}
			}
		}
		if fits {
			break
		}
	}
	if !fits {
		return nil, fmt.Errorf("no prefill tile fits %s's %d bytes of shared memory per block", prof.Product, smem)
	}
	if err := overridePrefillTiles(&pf, prefillRows, smem); err != nil {
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
	if err := pf.planNarrow(cfg, qkvBias, opt.SMs); err != nil {
		return nil, err
	}
	p.pf = &pf
	return &p, nil
}

// prefillTiles are the prefill tile heights, pipeline depths and shared
// memory layouts, in order of preference. Stages 3 is [hopper.TileGEMM]'s
// default. On GB10, whose 99 KB of shared memory per block holds no padded
// 3-stage 128-row tile, the unpadded swizzled layout
// ([hopper.TileGEMM.Swizzle128B]) holds one. A 128-row tile at 2 stages
// comes before a 64-row one at 3: it cuts Llama-3-8B prefill by a third
// (128 to 1980 tokens), reading the weights once per 128 rows instead of
// per 64, with twice the mma per ldmatrix.
var prefillTiles = []struct {
	bm, stages int
	swizzle    bool
}{
	{128, 3, false},
	{128, 3, true},
	{128, 2, false},
	{64, 3, false},
	{64, 2, false},
}

// attnSplits returns the decode attention splits per KV head that make
// kvHeads × splits CTAs cover p's SMs once.
func attnSplits(p cudadevicetable.Profile, kvHeads int) int {
	return max(1, int(p.SMs.Value)/kvHeads)
}

// compilePlan optimizes and lowers root, running single-CTA row kernels as
// the tails of the gemms before them ([hopper.Plan.FuseTails]).
func compilePlan(name string, b *hopper.Builder, root egraph.Id, out hopper.DType, target string) (*hopper.Plan, error) {
	e, err := hopper.Optimize(b.G, root)
	if err != nil {
		return nil, fmt.Errorf("optimize %s: %w", name, err)
	}
	p, err := hopper.CompileFor(e, name, out, target)
	if err != nil {
		return nil, err
	}
	p.FuseTails()
	return p, nil
}

// buildLayerPlans builds a decoder layer's attnIn and post plans over m
// rows, named prefix_attn_in and prefix_post. The post plan reads the
// attention output in attnDType. With chain, attnIn reads the normalized
// input xn and post writes the next layer's xn (see [egraphPlans]);
// otherwise attnIn normalizes x and post writes the residual. The gemms
// are lowered for target, an nvcc -arch value or "".
func buildLayerPlans(cfg *ModelConfig, qkvBias, attn bool, prefix string, m int, attnDType hopper.DType, target string, chain bool) (attnIn, post *hopper.Plan, err error) {
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
	var norm egraph.Id
	if chain {
		norm = b.Input("xn", hopper.BF16, m, h)
	} else {
		x := b.Input("x", hopper.F32, m, h)
		norm = b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("attn_norm", hopper.F32, h), h, eps))
	}
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
	x := b.Input("x", hopper.F32, m, h)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(ao, b.Input("w_o", hopper.BF16, h, qDim)), x)
	norm = b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*inter, h)))
	out := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, h, inter)), hid)
	outDType := hopper.F32
	if chain {
		out = b.Cast(hopper.BF16, b.RMSNorm(out, b.Input("next_norm", hopper.F32, h), h, eps))
		outDType = hopper.BF16
	}
	if post, err = compilePlan(prefix+"_post", b, out, outDType, target); err != nil {
		return nil, nil, err
	}
	return attnIn, post, nil
}

// normInput returns the temporary that p's output RMSNorm reads: the row
// kernel writing p's output, or the gemm whose tail it is, reads it.
func normInput(p *hopper.Plan) (string, error) {
	for _, k := range p.Kernels {
		r := k.Row
		if k.GEMM != nil {
			r = k.GEMM.Tail
		}
		if r == nil || len(r.Outputs) != 1 || r.Outputs[0].Name != p.Output.Name {
			continue
		}
		for _, in := range r.Inputs {
			for _, t := range p.Temps {
				if in.Buffer == t.Name || in.Name == t.Name {
					return t.Name, nil
				}
			}
		}
	}
	return "", fmt.Errorf("%s: no row kernel normalizes a temporary into %s", p.Name, p.Output.Name)
}
