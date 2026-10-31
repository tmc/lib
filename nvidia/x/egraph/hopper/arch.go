package hopper

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/model"
)

// NVFP4 is the DType alias for 4-bit E2M1 block-scaled weights.
const NVFP4 = E2M1

// Execution mode names selected by [CompileLayer].
const (
	ModePersistentMegakernel = model.ModePersistentMegakernel
	ModeStreamKWarpSpec      = model.ModeStreamKWarpSpec
	ModeMultiLaunch          = model.ModeMultiLaunch
)

// ArchConfig describes a transformer decoder architecture (dense GQA or
// routed MoE, optional gated attention and QK norm, BF16/E4M3/NVFP4 weights).
// It unmarshals directly from standard HuggingFace config.json bytes via
// [ParseArchConfig].
type ArchConfig struct {
	Architectures       []string `json:"architectures,omitempty"`
	HiddenSize          int      `json:"hidden_size"`
	IntermediateSize    int      `json:"intermediate_size"`
	MoEIntermediateSize int      `json:"moe_intermediate_size,omitempty"`
	NumHiddenLayers     int      `json:"num_hidden_layers"`
	NumAttentionHeads   int      `json:"num_attention_heads"`
	NumKeyValueHeads    int      `json:"num_key_value_heads"`
	HeadDim             int      `json:"head_dim,omitempty"`
	NumLocalExperts     int      `json:"num_local_experts,omitempty"`
	NumExperts          int      `json:"num_experts,omitempty"`
	NumExpertsPerTok    int      `json:"num_experts_per_tok,omitempty"`
	RMSNormEps          float32  `json:"rms_norm_eps,omitempty"`
	QKNorm              bool     `json:"qk_norm,omitempty"`
	AttentionGate       bool     `json:"attention_gate,omitempty"`
	WeightDType         DType    `json:"-"` // BF16, E4M3, NVFP4 (inferred from quantization_config or explicit)
	KVDType             DType    `json:"-"` // BF16 or E4M3
}

func (c ArchConfig) toModelArch() model.ArchConfig {
	return model.ArchConfig{
		Architectures:       c.Architectures,
		HiddenSize:          c.HiddenSize,
		IntermediateSize:    c.IntermediateSize,
		MoEIntermediateSize: c.MoEIntermediateSize,
		NumHiddenLayers:     c.NumHiddenLayers,
		NumAttentionHeads:   c.NumAttentionHeads,
		NumKeyValueHeads:    c.NumKeyValueHeads,
		HeadDim:             c.HeadDim,
		NumLocalExperts:     c.NumLocalExperts,
		NumExperts:          c.NumExperts,
		NumExpertsPerTok:    c.NumExpertsPerTok,
		RMSNormEps:          c.RMSNormEps,
		QKNorm:              c.QKNorm,
		AttentionGate:       c.AttentionGate,
		WeightDType:         model.DType(c.WeightDType),
		KVDType:             model.DType(c.KVDType),
	}
}

func fromModelArch(mc model.ArchConfig) ArchConfig {
	return ArchConfig{
		Architectures:       mc.Architectures,
		HiddenSize:          mc.HiddenSize,
		IntermediateSize:    mc.IntermediateSize,
		MoEIntermediateSize: mc.MoEIntermediateSize,
		NumHiddenLayers:     mc.NumHiddenLayers,
		NumAttentionHeads:   mc.NumAttentionHeads,
		NumKeyValueHeads:    mc.NumKeyValueHeads,
		HeadDim:             mc.HeadDim,
		NumLocalExperts:     mc.NumLocalExperts,
		NumExperts:          mc.NumExperts,
		NumExpertsPerTok:    mc.NumExpertsPerTok,
		RMSNormEps:          mc.RMSNormEps,
		QKNorm:              mc.QKNorm,
		AttentionGate:       mc.AttentionGate,
		WeightDType:         DType(mc.WeightDType),
		KVDType:             DType(mc.KVDType),
	}
}

// ParseArchConfig unmarshals a HuggingFace-style config.json payload into an
// [ArchConfig], inferring WeightDType and KVDType from quantization_config when
// present and filling defaults.
func ParseArchConfig(data []byte) (ArchConfig, error) {
	mc, err := model.ParseArchConfig(data)
	if err != nil {
		return ArchConfig{}, err
	}
	return fromModelArch(mc), nil
}

// IsMoE reports whether c configures a routed Mixture-of-Experts layer.
func (c ArchConfig) IsMoE() bool {
	return c.toModelArch().IsMoE()
}

// WeightFormat returns the [WeightFormat] corresponding to c.WeightDType.
func (c ArchConfig) WeightFormat() WeightFormat {
	return WeightFormat(c.toModelArch().WeightFormat())
}

func (c ArchConfig) norm() (ArchConfig, error) {
	mc, err := model.ValidateArchConfig(c.toModelArch())
	if err != nil {
		return c, err
	}
	return fromModelArch(mc), nil
}

// RMSNorm adds a per-row RMSNorm term over the last axis of width dim scaled
// by weight [dim]:
//
//	(x * rsqrt(mean(x^2) + eps)) * weight
func (b *Builder) RMSNorm(x, weight egraph.Id, dim int, eps float32) egraph.Id {
	if eps <= 0 {
		eps = 1e-5
	}
	ss := b.ReduceSum(b.Map(MustLambda("(mul $0 $0)"), x))
	fn := MustLambda(fmt.Sprintf("(mul (mul $0 (rsqrt (add (div $1 %d) %s))) $2)",
		dim, strconv.FormatFloat(float64(eps), 'g', -1, 32)))
	return b.Map(fn, x, ss, weight)
}

// ResidualRMSNorm adds x + residual followed by per-row [Builder.RMSNorm] over
// width dim with weight [dim].
func (b *Builder) ResidualRMSNorm(x, residual, weight egraph.Id, dim int, eps float32) egraph.Id {
	res := b.Map(MustLambda("(add $0 $1)"), x, residual)
	return b.RMSNorm(res, weight, dim, eps)
}

// AttentionWithGate adds a [Builder.PagedAttention] term followed by
// [Builder.RMSNormGate] so equality saturation ([Optimize]) rewrites the pair
// into a single gated [PagedAttention] launch (PagedAttention{Gate: true}).
func (b *Builder) AttentionWithGate(cfg PagedAttention, q, kCache, vCache, pageTable, seqLens, kScale, vScale, gate egraph.Id, eps float32) egraph.Id {
	attn := b.PagedAttention(cfg, q, kCache, vCache, pageTable, seqLens, kScale, vScale)
	return b.RMSNormGate(attn, gate, eps)
}

// MoEChain adds the routed Mixture-of-Experts operator sequence:
//
//	RowTopK -> ExpertAlign -> GroupedMatMul(w13) -> SwiGLU -> GroupedMatMul(w2) -> PostExpertNorm -> WeightedTopKSum
//
// which equality saturation ([Optimize]) rewrites into:
//
//	RowTopK -> ExpertAlign -> TileGEMM(GLU) -> TileGEMM(SumSq) -> ExpertSum
func (b *Builder) MoEChain(routeCfg RowTopK, alignCfg ExpertAlign, x, logits, bias, localStart, w13, w2, postW egraph.Id, eps float32) egraph.Id {
	ids, weights := b.RowTopK(routeCfg, logits, bias)
	offsets, rows, slots := b.ExpertAlign(alignCfg, ids, localStart)
	mm13 := b.GroupedMatMul(x, w13, offsets, rows)
	inter := b.SwiGLU(mm13)
	mm2 := b.GroupedMatMul(inter, w2, offsets)
	normed := b.PostExpertNorm(mm2, postW, eps)
	return b.WeightedTopKSum(normed, slots, weights)
}

// A TermProgram holds the e-graph and root term of a transformer decoder layer
// built by [BuildLayerProgram].
type TermProgram struct {
	Builder *Builder
	G       *egraph.EGraph
	Root    egraph.Id
	Config  ArchConfig
	BatchM  int
	SeqLen  int
	OutType DType
}

// Optimize saturates p.G and extracts the minimum-cost expression for p.Root.
func (p *TermProgram) Optimize() (*egraph.Expr, error) {
	if p == nil || p.G == nil {
		return nil, fmt.Errorf("optimize term program: nil program")
	}
	return Optimize(p.G, p.Root)
}

// Eval saturates p.G and evaluates the extracted layer expression on env.
func (p *TermProgram) Eval(env map[string]Value) (Value, error) {
	expr, err := p.Optimize()
	if err != nil {
		return Value{}, err
	}
	return Eval(expr, env)
}

// BuildLayerProgram constructs the complete e-graph [TermProgram] for one
// transformer decoder layer under cfg at batch/token count batchM and sequence
// length seqLen.
func BuildLayerProgram(cfg ArchConfig, batchM, seqLen int) (*TermProgram, error) {
	var err error
	cfg, err = cfg.norm()
	if err != nil {
		return nil, err
	}
	if batchM <= 0 {
		return nil, fmt.Errorf("build layer program: batchM %d must be positive", batchM)
	}
	if seqLen <= 0 {
		return nil, fmt.Errorf("build layer program: seqLen %d must be positive", seqLen)
	}

	H := cfg.HiddenSize
	hq := cfg.NumAttentionHeads
	hkv := cfg.NumKeyValueHeads
	qDim := hq * HeadDim
	eps := cfg.RMSNormEps
	useNVFP4 := cfg.WeightDType == NVFP4 && H >= 64 && H%16 == 0

	b := NewBuilder()
	addWeight2D := func(name string, n, k int) egraph.Id {
		if useNVFP4 && k >= 64 && k%16 == 0 {
			return b.NVFP4(b.Input(name+"_q", U8, n, k/2), b.Input(name+"_sf", U8, n, k/16), b.Input(name+"_gs", F32, 1))
		}
		return b.Input(name, BF16, n, k)
	}
	addWeight3D := func(name string, local, n, k int) egraph.Id {
		if useNVFP4 && k >= 64 && k%16 == 0 {
			return b.NVFP4(b.Input(name+"_q", U8, local, n, k/2), b.Input(name+"_sf", U8, local, n, k/16), b.Input(name+"_gs", F32, local))
		}
		return b.Input(name, BF16, local, n, k)
	}

	x := b.Input("x", BF16, batchM, H)
	attnNormW := b.Input("attn_norm_w", BF16, H)
	norm1 := b.Cast(BF16, b.RMSNorm(x, attnNormW, H, eps))

	wQ := addWeight2D("wq", qDim, H)
	qProj := b.MatMul(norm1, wQ)
	if cfg.QKNorm {
		qProj = b.RMSNorm(qProj, b.Input("qk_norm_w", BF16, qDim), qDim, eps)
	}
	qDType := BF16
	if cfg.KVDType == E4M3 && cfg.WeightDType == E4M3 {
		qDType = E4M3
	}
	q := b.Cast(qDType, qProj)

	const pageSize = 16
	maxPages := max((seqLen+pageSize-1)/pageSize, 1)
	cacheRows := batchM * maxPages * pageSize
	kCache := b.Input("k_cache", U8, cacheRows, hkv, HeadDim)
	vCache := b.Input("v_cache", U8, cacheRows, hkv, HeadDim)
	pageTable := b.Input("page_table", I32, batchM, maxPages)
	seqLens := b.Input("seqlens", I32, batchM)
	kScale := b.Input("k_scale", F32, 1)
	vScale := b.Input("v_scale", F32, 1)

	attnCfg := PagedAttention{Hq: hq, Hkv: hkv, QLen: 1, Page: pageSize, Q: qDType, Chunk: 64}
	var attn egraph.Id
	if cfg.AttentionGate {
		gate := b.Cast(BF16, b.MatMul(norm1, addWeight2D("w_attn_gate", qDim, H)))
		attn = b.AttentionWithGate(attnCfg, q, kCache, vCache, pageTable, seqLens, kScale, vScale, gate, eps)
	} else {
		attn = b.PagedAttention(attnCfg, q, kCache, vCache, pageTable, seqLens, kScale, vScale)
	}
	attnBF16 := b.Cast(BF16, attn)

	oProj := b.MatMul(attnBF16, addWeight2D("wo", H, qDim))
	h1 := b.Map(MustLambda("(add $0 $1)"), oProj, x)
	norm2 := b.Cast(BF16, b.RMSNorm(h1, b.Input("ffn_norm_w", BF16, H), H, eps))

	var mlpOut egraph.Id
	if cfg.IsMoE() {
		experts := min(max(32, ((cfg.NumExperts+31)/32)*32), 1024)
		local := min(max(cfg.NumLocalExperts, 1), min(32, experts))
		kRoutes := min(max(cfg.NumExpertsPerTok, 1), min(16, experts))
		moeI := cfg.MoEIntermediateSize
		logits := b.Input("router_logits", F32, batchM, experts)
		bias := b.Input("router_bias", F32, experts)
		localStart := b.Input("local_start", I32, 1)
		w13 := addWeight3D("w13", local, 2*moeI, H)
		w2 := addWeight3D("w2", local, H, moeI)
		postExpertW := b.Input("post_expert_w", BF16, H)
		routeCfg := RowTopK{
			D: experts, K: kRoutes, X: F32, Bias: F32,
			Score: MustLambda("(sigmoid $0)"), Key: MustLambda("(add $0 $1)"),
			Normalize: true, Eps: 1e-15, Scale: 1,
		}
		alignCfg := ExpertAlign{K: kRoutes, Experts: experts, Local: local, Warps: 4}
		mlpOut = b.MoEChain(routeCfg, alignCfg, norm2, logits, bias, localStart, w13, w2, postExpertW, 1e-8)
	} else {
		I := cfg.IntermediateSize
		gateUp := b.MatMul(norm2, addWeight2D("w_gate_up", 2*I, H))
		mlpOut = b.MatMul(b.SwiGLU(gateUp), addWeight2D("w_down", H, I))
	}

	root := b.ResidualRMSNorm(mlpOut, h1, b.Input("post_norm_w", BF16, H), H, eps)
	return &TermProgram{
		Builder: b, G: b.G, Root: root, Config: cfg,
		BatchM: batchM, SeqLen: seqLen, OutType: BF16,
	}, nil
}

// CostEstimate aliases [model.CostEstimate].
type CostEstimate = model.CostEstimate

// A CompiledLayer holds the optimized [Plan], per-step [CostEstimate] breakdown,
// execution mode, and emitted CUDA / Megakernel / TVM-FFI sources for one
// transformer decoder layer.
type CompiledLayer struct {
	Config           ArchConfig     `json:"config"`
	BatchM           int            `json:"batch_m"`
	SeqLen           int            `json:"seq_len"`
	Device           string         `json:"device"`
	Mode             string         `json:"mode"`
	ExecutionMode    string         `json:"execution_mode"`
	Plan             *Plan          `json:"plan"`
	Estimates        []CostEstimate `json:"estimates"`
	TotalNs          float64        `json:"total_ns"`
	CUDASource       string         `json:"cuda_source,omitempty"`
	MegakernelSource string         `json:"megakernel_source,omitempty"`
	TVMFFISource     string         `json:"tvmffi_source,omitempty"`
	Megakernels      []Megakernel   `json:"-"`
}

// CompileWithCost saturates p.G, lowers the optimal expression to a [Plan] named
// "layer", selects hardware schedules for device, and returns the plan and its
// per-step [CostEstimate] slice.
func (p *TermProgram) CompileWithCost(device string) (*Plan, []CostEstimate, error) {
	return p.compileNamedWithCost("layer", device)
}

func (p *TermProgram) compileNamedWithCost(name, device string) (*Plan, []CostEstimate, error) {
	if p == nil || p.G == nil {
		return nil, nil, fmt.Errorf("compile term program: nil program")
	}
	expr, err := p.Optimize()
	if err != nil {
		return nil, nil, err
	}
	outType := p.OutType
	if outType == 0 {
		outType = BF16
	}
	costCfg, err := (CostConfig{Device: device, MaxKeys: p.SeqLen}).norm()
	if err != nil {
		return nil, nil, err
	}
	plan, err := compileArchExpr(expr, name, outType, p.BatchM)
	if err != nil {
		return nil, nil, err
	}
	if err := selectArchSchedules(plan, p.Config, p.BatchM, costCfg); err != nil {
		return nil, nil, err
	}
	return plan, estimatePlanSteps(plan, costCfg), nil
}

func compileArchExpr(expr *egraph.Expr, name string, out DType, batchM int) (*Plan, error) {
	plan, err := Compile(expr, name, out)
	if err == nil {
		return plan, nil
	}
	if batchM > 16 || !strings.Contains(err.Error(), "skinny gemm") {
		return nil, err
	}
	const inflatedM = 32
	inflated := rescaleExprBatch(expr, batchM, inflatedM, make(map[*egraph.Expr]*egraph.Expr))
	plan, err = Compile(inflated, name, out)
	if err != nil {
		return nil, err
	}
	rescalePlanBatch(plan, inflatedM, batchM)
	return plan, nil
}

func isBatchInputTensor(name string) bool {
	switch name {
	case "x", "page_table", "seqlens", "router_logits", "k_cache", "v_cache":
		return true
	default:
		return false
	}
}

func rescaleExprBatch(e *egraph.Expr, fromM, toM int, memo map[*egraph.Expr]*egraph.Expr) *egraph.Expr {
	if e == nil {
		return nil
	}
	if c, ok := memo[e]; ok {
		return c
	}
	cp := *e
	if t, ok := cp.Data.(Tensor); ok {
		t.Shape = slices.Clone(t.Shape)
		if isBatchInputTensor(t.Name) {
			if (t.Name == "k_cache" || t.Name == "v_cache") && len(t.Shape) >= 2 && t.Shape[0]%fromM == 0 {
				t.Shape[0] = (t.Shape[0] / fromM) * toM
			} else if len(t.Shape) >= 1 && t.Shape[0] == fromM {
				t.Shape[0] = toM
			}
		}
		cp.Data = t
	} else if d, ok := cp.Data.(Dims); ok {
		d.Shape = slices.Clone(d.Shape)
		if len(d.Shape) >= 2 && d.Shape[0] == fromM {
			d.Shape[0] = toM
		}
		cp.Data = d
	}
	if len(e.Children) > 0 {
		cp.Children = make([]*egraph.Expr, len(e.Children))
		for i, ch := range e.Children {
			cp.Children[i] = rescaleExprBatch(ch, fromM, toM, memo)
		}
	}
	memo[e] = &cp
	return &cp
}

func rescalePlanBatch(p *Plan, fromM, toM int) {
	for i := range p.Inputs {
		t := &p.Inputs[i]
		if !isBatchInputTensor(t.Name) {
			continue
		}
		if (t.Name == "k_cache" || t.Name == "v_cache") && len(t.Shape) >= 2 && t.Shape[0]%fromM == 0 {
			t.Shape[0] = (t.Shape[0] / fromM) * toM
		} else if len(t.Shape) >= 1 && t.Shape[0] == fromM {
			t.Shape[0] = toM
		}
	}
	for i := range p.Temps {
		if len(p.Temps[i].Shape) >= 2 && p.Temps[i].Shape[0]%fromM == 0 {
			p.Temps[i].Shape[0] = (p.Temps[i].Shape[0] / fromM) * toM
		}
	}
	if len(p.Output.Shape) >= 2 && p.Output.Shape[0]%fromM == 0 {
		p.Output.Shape[0] = (p.Output.Shape[0] / fromM) * toM
	}
	for i := range p.Kernels {
		k := &p.Kernels[i]
		switch {
		case k.Tile != nil:
			if k.M%fromM == 0 {
				k.M = (k.M / fromM) * toM
			}
			if k.MaxRows%fromM == 0 {
				k.MaxRows = (k.MaxRows / fromM) * toM
			}
			if !k.Tile.Grouped && !k.Tile.GLU && !k.Tile.SumSq && k.M <= 16 {
				sg := SkinnyGEMM{
					Name: k.Tile.Name, N: k.Tile.N, K: k.Tile.K, MaxM: k.M,
					Weight: k.Tile.Weight, Out: k.Tile.Out, Epilogue: k.Tile.Epilogue, Extras: k.Tile.Extras,
				}
				if normed, err := sg.norm(); err == nil {
					k.GEMM = &normed
					k.Tile = nil
				}
			}
		case k.Row != nil, k.TopK != nil, k.Align != nil, k.ExpertSum != nil:
			if k.Rows%fromM == 0 {
				k.Rows = (k.Rows / fromM) * toM
			}
		case k.Attention != nil:
			if k.Batch%fromM == 0 {
				k.Batch = (k.Batch / fromM) * toM
			}
			wsBytes := max(k.Attention.WorkspaceBytes(max(k.Batch, 1), k.MaxKeys), 64)
			if len(k.Args) > 0 {
				wsName := k.Args[len(k.Args)-1]
				for j := range p.Temps {
					if p.Temps[j].Name == wsName {
						p.Temps[j].Shape = []int{wsBytes}
					}
				}
			}
		}
	}
}

func selectArchSchedules(p *Plan, archCfg ArchConfig, batchM int, costCfg CostConfig) error {
	if err := p.SelectSchedules(costCfg); err != nil {
		return err
	}
	hopperArch := costCfg.isHopper()
	blackwellArch := costCfg.isBlackwell()
	for i := range p.Kernels {
		k := &p.Kernels[i]
		if k.Tile == nil {
			continue
		}
		t := *k.Tile
		switch {
		case blackwellArch && t.Weight == WeightNVFP4:
			cand := t
			cand.WGMMA, cand.Swizzle128B, cand.Repacked, cand.WarpSpecialized = false, false, false, false
			cand.DequantWords, cand.BlackwellFP4 = 0, true
			if k.M >= 64 || batchM >= 64 {
				cand.GroupM = 8
			}
			if normed, err := cand.norm(); err == nil {
				*k.Tile = normed
			}
		case hopperArch && t.Weight == WeightNVFP4:
			cand := t
			cand.BlackwellFP4, cand.BM, cand.BN = false, 128, 128
			cand.WGMMA, cand.Swizzle128B, cand.Repacked = true, true, true
			cand.DequantWords, cand.GroupM = 2, 8
			if normed, err := cand.norm(); err == nil {
				*k.Tile = normed
			} else {
				cand.Repacked, cand.DequantWords = false, 0
				if cand.N%128 != 0 || (cand.GLU && (cand.N/2)%64 != 0) {
					cand.BN = 64
				}
				if normed2, err2 := cand.norm(); err2 == nil {
					*k.Tile = normed2
				}
			}
		case hopperArch && (k.M >= 64 || batchM >= 64) && t.K >= 64 && t.K%64 == 0 && t.Weight != WeightBF16HiLo:
			cand := t
			cand.BM = 128
			if cand.N%128 == 0 && (!cand.GLU || (cand.N/2)%64 == 0) {
				cand.BN = 128
			} else {
				cand.BN = 64
			}
			cand.WGMMA, cand.Swizzle128B, cand.GroupM = true, true, 8
			if batchM >= 64 && !cand.Grouped && cand.Weight == WeightBF16 {
				wsCand := cand
				wsCand.WarpSpecialized = true
				if normedWS, err := wsCand.norm(); err == nil {
					*k.Tile = normedWS
					continue
				}
			}
			if normed, err := cand.norm(); err == nil {
				*k.Tile = normed
			}
		}
		_ = archCfg
	}
	return nil
}

func estimatePlanSteps(p *Plan, costCfg CostConfig) []CostEstimate {
	normCfg, err := costCfg.norm()
	if err != nil {
		normCfg = costCfg
	}
	out := make([]CostEstimate, len(p.Kernels))
	for i, k := range p.Kernels {
		bytes, flops := k.Workload(normCfg)
		ns := k.estimateStepNs(normCfg)
		name, op, sched := describePlanKernel(k)
		out[i] = CostEstimate{Name: name, Op: op, Schedule: sched, Bytes: bytes, FLOPs: flops, EstimatedNs: ns}
	}
	return out
}

func describePlanKernel(k PlanKernel) (name, op, sched string) {
	var parts []string
	switch {
	case k.GEMM != nil:
		g := k.GEMM
		name, op = g.Name, "skinny_gemm"
		parts = append(parts, fmt.Sprintf("w=%s", g.Weight))
		if g.Prefetch > 0 {
			parts = append(parts, fmt.Sprintf("prefetch=%d", g.Prefetch))
		}
		if g.BlackwellFP4 {
			parts = append(parts, "blackwell_fp4")
		}
	case k.MMA != nil:
		g := k.MMA
		name, op = g.Name, "skinny_mma"
		parts = append(parts, fmt.Sprintf("tiles=%dx%d", g.Warps, g.Tiles), fmt.Sprintf("split=%d", g.SplitK))
		if g.GLU {
			parts = append(parts, "glu")
		}
	case k.Tile != nil:
		t := k.Tile
		name, op = t.Name, "tile_gemm"
		parts = append(parts, fmt.Sprintf("w=%s", t.Weight), fmt.Sprintf("tile=%dx%d", t.BM, t.BN))
		if t.WGMMA {
			parts = append(parts, "wgmma")
		}
		if t.Swizzle128B {
			parts = append(parts, "swizzle_128b")
		}
		if t.Repacked {
			parts = append(parts, "repacked")
		}
		if t.DequantWords > 0 {
			parts = append(parts, fmt.Sprintf("dequant_words=%d", t.DequantWords))
		}
		if t.WarpSpecialized {
			parts = append(parts, "warp_specialized")
		}
		if t.StreamK > 0 {
			parts = append(parts, fmt.Sprintf("stream_k=%d", t.StreamK))
		}
		if t.BlackwellFP4 {
			parts = append(parts, "blackwell_fp4")
		}
		if t.GroupM > 1 {
			parts = append(parts, fmt.Sprintf("group_m=%d", t.GroupM))
		}
	case k.Row != nil:
		name, op = k.Row.Name, "row_kernel"
		parts = append(parts, fmt.Sprintf("d=%d", k.Row.D))
		if k.Row.Cluster {
			parts = append(parts, "cluster")
		}
	case k.Attention != nil:
		a := k.Attention
		name, op = a.Name, "paged_attention"
		if a.Gate {
			parts = append(parts, "gated")
		}
		if a.WGMMA {
			parts = append(parts, "wgmma")
		}
		if a.Swizzle {
			parts = append(parts, "swizzle")
		}
		if a.TableAhead > 1 {
			parts = append(parts, fmt.Sprintf("table_ahead=%d", a.TableAhead))
		}
		if a.ClusterCombine {
			parts = append(parts, "cluster_combine")
		} else if a.DeviceSplits {
			parts = append(parts, "device_splits")
		}
	case k.TopK != nil:
		name, op = k.TopK.Name, "row_topk"
		parts = append(parts, fmt.Sprintf("d=%d,k=%d", k.TopK.D, k.TopK.K))
	case k.Align != nil:
		name, op = k.Align.Name, "expert_align"
		parts = append(parts, fmt.Sprintf("local=%d,experts=%d", k.Align.Local, k.Align.Experts))
	case k.ExpertSum != nil:
		name, op = k.ExpertSum.Name, "expert_sum"
		parts = append(parts, fmt.Sprintf("width=%d,k=%d", k.ExpertSum.Width, k.ExpertSum.K))
	}
	return name, op, strings.Join(parts, ",")
}

// CompileLayer builds and lowers a transformer decoder layer for cfg at
// (batchM, seqLen) targeting device, selecting hardware-specific schedules,
// execution mode, persistent megakernel sub-plans (when batchM <= 8), CUDA
// source, and TVM-FFI host wrapper headers.
func CompileLayer(cfg ArchConfig, batchM, seqLen int, device string) (*CompiledLayer, error) {
	return compileLayerNamed(cfg, batchM, seqLen, device, "layer")
}

func compileLayerNamed(cfg ArchConfig, batchM, seqLen int, device, planName string) (*CompiledLayer, error) {
	var err error
	cfg, err = cfg.norm()
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(device) == "" {
		device = "NVIDIA H200"
	}
	prog, err := BuildLayerProgram(cfg, batchM, seqLen)
	if err != nil {
		return nil, err
	}
	plan, estimates, err := prog.compileNamedWithCost(planName, device)
	if err != nil {
		return nil, err
	}
	var totalNs float64
	for _, est := range estimates {
		totalNs += est.EstimatedNs
	}
	mode := model.SelectLayerExecutionMode(batchM)
	var (
		megas   []Megakernel
		megaSrc string
	)
	if mode == ModePersistentMegakernel {
		megas, megaSrc = buildLayerMegakernels(plan)
	}
	cudaSrc, err := planCUDASource(plan, megaSrc)
	if err != nil {
		return nil, err
	}
	ffiSrc, err := planTVMFFISource(plan)
	if err != nil {
		return nil, err
	}
	return &CompiledLayer{
		Config: cfg, BatchM: batchM, SeqLen: seqLen, Device: device,
		Mode: mode, ExecutionMode: mode, Plan: plan, Estimates: estimates,
		TotalNs: totalNs, CUDASource: cudaSrc, MegakernelSource: megaSrc,
		TVMFFISource: ffiSrc, Megakernels: megas,
	}, nil
}

func buildLayerMegakernels(p *Plan) ([]Megakernel, string) {
	if p == nil || len(p.Kernels) < 2 {
		return nil, ""
	}
	tensorByName := make(map[string]Tensor)
	for _, t := range p.Inputs {
		tensorByName[t.Name] = t
	}
	for _, t := range p.Temps {
		tensorByName[t.Name] = t
	}
	if p.Output.Name != "" {
		tensorByName[p.Output.Name] = p.Output
	}
	var segments [][]PlanKernel
	var cur []PlanKernel
	flush := func() {
		if len(cur) >= 2 {
			segments = append(segments, cur)
		}
		cur = nil
	}
	for _, pk := range p.Kernels {
		if pk.Attention != nil {
			flush()
			continue
		}
		cp := pk
		if cp.Row != nil {
			rk := *cp.Row
			rk.Cluster = false
			cp.Row = &rk
		}
		if cp.Tile != nil {
			tk := *cp.Tile
			tk.WarpSpecialized = false
			cp.Tile = &tk
		}
		cur = append(cur, cp)
	}
	flush()
	var (
		megas []Megakernel
		srcs  []string
	)
	for idx, seg := range segments {
		sub := buildSubPlan(fmt.Sprintf("%s_sub%d", p.Name, idx+1), seg, tensorByName)
		m, err := sub.Megakernel(fmt.Sprintf("%s_mega%d", p.Name, idx+1))
		if err != nil {
			continue
		}
		src, err := m.Source()
		if err != nil {
			continue
		}
		megas = append(megas, m)
		srcs = append(srcs, src)
	}
	return megas, strings.Join(srcs, "\n")
}

func buildSubPlan(name string, seg []PlanKernel, tensorByName map[string]Tensor) *Plan {
	produced := make(map[string]bool)
	var inputs, temps []Tensor
	seenIn := make(map[string]bool)
	seenTmp := make(map[string]bool)
	for i, pk := range seg {
		outs := kernelOutputNames(pk)
		outSet := make(map[string]bool, len(outs))
		for _, o := range outs {
			outSet[o] = true
		}
		for _, a := range pk.Args {
			if !outSet[a] && !produced[a] && !seenIn[a] {
				if t, ok := tensorByName[a]; ok {
					seenIn[a] = true
					inputs = append(inputs, t)
				}
			}
		}
		for _, o := range outs {
			produced[o] = true
			if i < len(seg)-1 && !seenTmp[o] {
				if t, ok := tensorByName[o]; ok {
					seenTmp[o] = true
					temps = append(temps, t)
				}
			}
		}
	}
	var outTensor Tensor
	if lastOuts := kernelOutputNames(seg[len(seg)-1]); len(lastOuts) > 0 {
		outTensor = tensorByName[lastOuts[0]]
		for _, extraOut := range lastOuts[1:] {
			if t, ok := tensorByName[extraOut]; ok && !seenTmp[extraOut] {
				seenTmp[extraOut] = true
				temps = append(temps, t)
			}
		}
	}
	return &Plan{Name: name, Inputs: inputs, Temps: temps, Output: outTensor, Kernels: seg}
}

func kernelOutputNames(pk PlanKernel) []string {
	switch {
	case pk.GEMM != nil:
		outIdx := len(pk.Args) - 1
		if pk.GEMM.SplitK > 1 {
			outIdx--
		}
		if outIdx >= 0 && outIdx < len(pk.Args) {
			return []string{pk.Args[outIdx]}
		}
	case pk.Tile != nil:
		end := len(pk.Args)
		if pk.Tile.StreamK > 0 {
			end--
		}
		nOut := 1
		if pk.Tile.SumSq {
			nOut = 2
		}
		if end >= nOut {
			return pk.Args[end-nOut : end]
		}
	case pk.Row != nil:
		var outs []string
		for _, o := range pk.Row.Outputs {
			outs = append(outs, o.Name)
		}
		return outs
	case pk.TopK != nil && len(pk.Args) >= 2:
		return pk.Args[len(pk.Args)-2:]
	case pk.Align != nil && len(pk.Args) >= 3:
		return pk.Args[len(pk.Args)-3:]
	case pk.ExpertSum != nil && len(pk.Args) >= 1:
		return []string{pk.Args[len(pk.Args)-1]}
	}
	return nil
}

func planCUDASource(p *Plan, megaSrc string) (string, error) {
	src, err := p.Source()
	if err != nil {
		return "", err
	}
	if megaSrc == "" {
		return src, nil
	}
	return src + "\n" + strings.TrimPrefix(megaSrc, Prelude+"\n"), nil
}

func planTVMFFISource(p *Plan) (string, error) {
	var rows []RowKernel
	for _, pk := range p.Kernels {
		if pk.Row != nil {
			rows = append(rows, *pk.Row)
		}
	}
	if len(rows) == 0 {
		return "", fmt.Errorf("plan %s: no row kernels for TVM-FFI export", p.Name)
	}
	return SourceTVMFFI(rows...)
}

// A MultiBucketBundle holds per-bucket [CompiledLayer] plans across a set of M
// batch/token buckets and a unified C++ shape dispatcher entry point
// dispatch_layer_m(int M, ...).
type MultiBucketBundle struct {
	Config       ArchConfig             `json:"config"`
	Device       string                 `json:"device"`
	SeqLen       int                    `json:"seq_len"`
	Buckets      []int                  `json:"buckets"`
	Layers       map[int]*CompiledLayer `json:"layers"`
	Ordered      []*CompiledLayer       `json:"ordered,omitempty"`
	DispatchFunc string                 `json:"dispatch_func"`
	CUDASource   string                 `json:"cuda_source,omitempty"`
	TVMFFISource string                 `json:"tvmffi_source,omitempty"`
}

// CompileMultiBucketLayer compiles cfg across multiple M buckets at seqLen on
// device and emits the per-bucket [CompiledLayer] plans alongside a C++
// dispatch_layer_m(int M, ...) shape dispatcher entry point.
func CompileMultiBucketLayer(cfg ArchConfig, buckets []int, seqLen int, device string) (*MultiBucketBundle, error) {
	var err error
	cfg, err = cfg.norm()
	if err != nil {
		return nil, err
	}
	sorted, err := model.ValidateBuckets(buckets)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(device) == "" {
		device = "NVIDIA H200"
	}
	layers := make(map[int]*CompiledLayer, len(sorted))
	ordered := make([]*CompiledLayer, 0, len(sorted))
	var allRows []RowKernel
	var planSpecs []model.BucketPlanSpec

	for _, m := range sorted {
		cl, err := compileLayerNamed(cfg, m, seqLen, device, fmt.Sprintf("layer_m%d", m))
		if err != nil {
			return nil, fmt.Errorf("compile bucket M=%d: %w", m, err)
		}
		layers[m] = cl
		ordered = append(ordered, cl)
		for _, pk := range cl.Plan.Kernels {
			if pk.Row != nil {
				allRows = append(allRows, *pk.Row)
			}
		}
		pSrc, err := cl.Plan.Source()
		if err != nil {
			return nil, err
		}
		planSpecs = append(planSpecs, model.BucketPlanSpec{
			M:          m,
			PlanName:   cl.Plan.Name,
			Mode:       cl.Mode,
			TotalNs:    cl.TotalNs,
			PlanSource: pSrc,
		})
	}

	refPlan := layers[sorted[len(sorted)-1]].Plan
	var params, callArgs []string
	for _, t := range refPlan.Inputs {
		params = append(params, "const "+t.DType.cType()+"* "+t.Name)
		callArgs = append(callArgs, t.Name)
	}
	for _, t := range append(slices.Clone(refPlan.Temps), refPlan.Output) {
		params = append(params, t.DType.cType()+"* "+t.Name)
		callArgs = append(callArgs, t.Name)
	}
	cudaSrc := model.BuildMultiBucketDispatcherSource(Prelude, planSpecs, params, callArgs)
	var ffiSrc string
	if len(allRows) > 0 {
		ffiSrc, err = SourceTVMFFI(allRows...)
		if err != nil {
			return nil, err
		}
	}
	return &MultiBucketBundle{
		Config: cfg, Device: device, SeqLen: seqLen, Buckets: sorted,
		Layers: layers, Ordered: ordered, DispatchFunc: "dispatch_layer_m",
		CUDASource: cudaSrc, TVMFFISource: ffiSrc,
	}, nil
}
