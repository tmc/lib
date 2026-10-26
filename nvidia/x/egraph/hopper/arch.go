package hopper

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// NVFP4 is the DType alias for 4-bit E2M1 block-scaled weights.
const NVFP4 = E2M1

// Execution mode names selected by [CompileLayer].
const (
	ModePersistentMegakernel = "persistent_megakernel"
	ModeStreamKWarpSpec      = "streamk_warpspec"
	ModeMultiLaunch          = "multi_launch"
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

type rawArchConfig struct {
	ArchConfig
	NRoutedExperts     int    `json:"n_routed_experts,omitempty"`
	NumExpertsPerToken int    `json:"num_experts_per_token,omitempty"`
	KVCacheDType       string `json:"kv_cache_dtype,omitempty"`
	WeightDTypeName    string `json:"weight_dtype,omitempty"`
	QuantizationConfig *struct {
		QuantMethod   string `json:"quant_method"`
		QuantAlgo     string `json:"quant_algo"`
		Format        string `json:"format"`
		KVCacheScheme string `json:"kv_cache_scheme"`
	} `json:"quantization_config,omitempty"`
}

// ParseArchConfig unmarshals a HuggingFace-style config.json payload into an
// [ArchConfig], inferring WeightDType and KVDType from quantization_config when
// present and filling defaults.
func ParseArchConfig(data []byte) (ArchConfig, error) {
	var raw rawArchConfig
	if err := json.Unmarshal(data, &raw); err != nil {
		return ArchConfig{}, fmt.Errorf("parse arch config: %w", err)
	}
	cfg := raw.ArchConfig
	if cfg.NumExperts == 0 && raw.NRoutedExperts > 0 {
		cfg.NumExperts = raw.NRoutedExperts
	}
	if cfg.NumExpertsPerTok == 0 && raw.NumExpertsPerToken > 0 {
		cfg.NumExpertsPerTok = raw.NumExpertsPerToken
	}
	cfg.WeightDType = BF16
	cfg.KVDType = BF16

	if raw.QuantizationConfig != nil {
		qm := strings.ToLower(strings.TrimSpace(raw.QuantizationConfig.QuantMethod))
		qa := strings.ToLower(strings.TrimSpace(raw.QuantizationConfig.QuantAlgo + " " + raw.QuantizationConfig.Format))
		switch {
		case qm == "nvfp4" || qm == "fp4" || qm == "e2m1" || strings.Contains(qm, "nvfp4") || strings.Contains(qa, "nvfp4"):
			cfg.WeightDType = NVFP4
			cfg.KVDType = E4M3
		case qm == "fp8" || qm == "e4m3" || strings.Contains(qm, "fp8") || strings.Contains(qa, "fp8") || strings.Contains(qa, "e4m3"):
			cfg.WeightDType = E4M3
			cfg.KVDType = E4M3
		case qm == "modelopt":
			cfg.WeightDType = NVFP4
			cfg.KVDType = E4M3
		}
		if kvs := strings.ToLower(strings.TrimSpace(raw.QuantizationConfig.KVCacheScheme)); kvs != "" {
			if strings.Contains(kvs, "fp8") || strings.Contains(kvs, "e4m3") {
				cfg.KVDType = E4M3
			} else if strings.Contains(kvs, "bf16") {
				cfg.KVDType = BF16
			}
		}
	}
	if raw.WeightDTypeName != "" {
		switch strings.ToLower(strings.TrimSpace(raw.WeightDTypeName)) {
		case "nvfp4", "fp4", "e2m1":
			cfg.WeightDType = NVFP4
		case "fp8", "e4m3", "fp8_e4m3":
			cfg.WeightDType = E4M3
		case "bf16", "bfloat16":
			cfg.WeightDType = BF16
		}
	}
	if raw.KVCacheDType != "" {
		switch strings.ToLower(strings.TrimSpace(raw.KVCacheDType)) {
		case "fp8", "e4m3", "fp8_e4m3":
			cfg.KVDType = E4M3
		case "bf16", "bfloat16":
			cfg.KVDType = BF16
		}
	}
	return cfg.norm()
}

// IsMoE reports whether c configures a routed Mixture-of-Experts layer.
func (c ArchConfig) IsMoE() bool {
	return c.NumExperts > 0 || c.NumLocalExperts > 0
}

// WeightFormat returns the [WeightFormat] corresponding to c.WeightDType.
func (c ArchConfig) WeightFormat() WeightFormat {
	switch c.WeightDType {
	case NVFP4:
		return WeightNVFP4
	case E4M3:
		return WeightE4M3
	default:
		return WeightBF16
	}
}

func (c ArchConfig) norm() (ArchConfig, error) {
	if c.HiddenSize <= 0 {
		return c, fmt.Errorf("arch config: hidden_size %d must be positive", c.HiddenSize)
	}
	if c.NumAttentionHeads <= 0 {
		return c, fmt.Errorf("arch config: num_attention_heads %d must be positive", c.NumAttentionHeads)
	}
	if c.NumKeyValueHeads <= 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.NumAttentionHeads%c.NumKeyValueHeads != 0 {
		return c, fmt.Errorf("arch config: num_attention_heads %d not divisible by num_key_value_heads %d",
			c.NumAttentionHeads, c.NumKeyValueHeads)
	}
	if c.HeadDim <= 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if c.NumHiddenLayers <= 0 {
		c.NumHiddenLayers = 1
	}
	if c.RMSNormEps <= 0 {
		c.RMSNormEps = 1e-5
	}
	if c.NumLocalExperts == 0 && c.NumExperts > 0 {
		c.NumLocalExperts = c.NumExperts
	}
	if c.NumExperts == 0 && c.NumLocalExperts > 0 {
		c.NumExperts = c.NumLocalExperts
	}
	if c.IsMoE() {
		if c.NumExpertsPerTok <= 0 {
			c.NumExpertsPerTok = min(2, c.NumExperts)
		}
		if c.MoEIntermediateSize <= 0 {
			c.MoEIntermediateSize = c.IntermediateSize
		}
		if c.MoEIntermediateSize <= 0 {
			return c, fmt.Errorf("arch config: moe_intermediate_size must be positive")
		}
	} else if c.IntermediateSize <= 0 {
		return c, fmt.Errorf("arch config: intermediate_size %d must be positive", c.IntermediateSize)
	}
	if c.WeightDType == 0 {
		c.WeightDType = BF16
	}
	if c.KVDType == 0 {
		if c.WeightDType == E4M3 || c.WeightDType == NVFP4 {
			c.KVDType = E4M3
		} else {
			c.KVDType = BF16
		}
	}
	return c, nil
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
			wq := b.Input(name+"_q", U8, n, k/2)
			wsf := b.Input(name+"_sf", U8, n, k/16)
			wgs := b.Input(name+"_gs", F32, 1)
			return b.NVFP4(wq, wsf, wgs)
		}
		return b.Input(name, BF16, n, k)
	}
	addWeight3D := func(name string, local, n, k int) egraph.Id {
		if useNVFP4 && k >= 64 && k%16 == 0 {
			wq := b.Input(name+"_q", U8, local, n, k/2)
			wsf := b.Input(name+"_sf", U8, local, n, k/16)
			wgs := b.Input(name+"_gs", F32, local)
			return b.NVFP4(wq, wsf, wgs)
		}
		return b.Input(name, BF16, local, n, k)
	}

	// 1. Pre-attention RMSNorm + Q projection.
	x := b.Input("x", BF16, batchM, H)
	attnNormW := b.Input("attn_norm_w", BF16, H)
	norm1 := b.Cast(BF16, b.RMSNorm(x, attnNormW, H, eps))

	wQ := addWeight2D("wq", qDim, H)
	qProj := b.MatMul(norm1, wQ)
	if cfg.QKNorm {
		qNormW := b.Input("qk_norm_w", BF16, qDim)
		qProj = b.RMSNorm(qProj, qNormW, qDim, eps)
	}
	qDType := BF16
	if cfg.KVDType == E4M3 && cfg.WeightDType == E4M3 {
		qDType = E4M3
	}
	q := b.Cast(qDType, qProj)

	// 2. PagedAttention (with AttentionWithGate when cfg.AttentionGate is true).
	const pageSize = 16
	maxPages := max((seqLen+pageSize-1)/pageSize, 1)
	cacheRows := batchM * maxPages * pageSize
	kCache := b.Input("k_cache", U8, cacheRows, hkv, HeadDim)
	vCache := b.Input("v_cache", U8, cacheRows, hkv, HeadDim)
	pageTable := b.Input("page_table", I32, batchM, maxPages)
	seqLens := b.Input("seqlens", I32, batchM)
	kScale := b.Input("k_scale", F32, 1)
	vScale := b.Input("v_scale", F32, 1)

	attnCfg := PagedAttention{
		Hq:    hq,
		Hkv:   hkv,
		QLen:  1,
		Page:  pageSize,
		Q:     qDType,
		Chunk: 64,
	}
	var attn egraph.Id
	if cfg.AttentionGate {
		wGate := addWeight2D("w_attn_gate", qDim, H)
		gate := b.Cast(BF16, b.MatMul(norm1, wGate))
		attn = b.AttentionWithGate(attnCfg, q, kCache, vCache, pageTable, seqLens, kScale, vScale, gate, eps)
	} else {
		attn = b.PagedAttention(attnCfg, q, kCache, vCache, pageTable, seqLens, kScale, vScale)
	}
	attnBF16 := b.Cast(BF16, attn)

	// 3. Output projection (OProj) + residual add with input x.
	wO := addWeight2D("wo", H, qDim)
	oProj := b.MatMul(attnBF16, wO)
	h1 := b.Map(MustLambda("(add $0 $1)"), oProj, x)

	// 4. Post-attention RMSNorm + MLP (MoEChain or Dense SwiGLU).
	ffnNormW := b.Input("ffn_norm_w", BF16, H)
	norm2 := b.Cast(BF16, b.RMSNorm(h1, ffnNormW, H, eps))

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
			D:         experts,
			K:         kRoutes,
			X:         F32,
			Bias:      F32,
			Score:     MustLambda("(sigmoid $0)"),
			Key:       MustLambda("(add $0 $1)"),
			Normalize: true,
			Eps:       1e-15,
			Scale:     1,
		}
		alignCfg := ExpertAlign{
			K:       kRoutes,
			Experts: experts,
			Local:   local,
			Warps:   4,
		}
		mlpOut = b.MoEChain(routeCfg, alignCfg, norm2, logits, bias, localStart, w13, w2, postExpertW, 1e-8)
	} else {
		I := cfg.IntermediateSize
		wGateUp := addWeight2D("w_gate_up", 2*I, H)
		gateUp := b.MatMul(norm2, wGateUp)
		act := b.SwiGLU(gateUp)
		wDown := addWeight2D("w_down", H, I)
		mlpOut = b.MatMul(act, wDown)
	}

	postNormW := b.Input("post_norm_w", BF16, H)
	root := b.ResidualRMSNorm(mlpOut, h1, postNormW, H, eps)

	return &TermProgram{
		Builder: b,
		G:       b.G,
		Root:    root,
		Config:  cfg,
		BatchM:  batchM,
		SeqLen:  seqLen,
		OutType: BF16,
	}, nil
}

// A CostEstimate records the roofline traffic, FLOPs, estimated latency, and
// selected schedule of one kernel step in a compiled [Plan].
type CostEstimate struct {
	Name        string  `json:"name"`
	Op          string  `json:"op"`
	Schedule    string  `json:"schedule,omitempty"`
	Bytes       uint64  `json:"bytes"`
	FLOPs       uint64  `json:"flops"`
	EstimatedNs float64 `json:"estimated_ns"`
}

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
	estimates := estimatePlanSteps(plan, costCfg)
	return plan, estimates, nil
}

// compileArchExpr lowers expr to a Plan, falling back to TileGEMM when batchM <= 16
// has inner dimension K smaller than SkinnyGEMM's minimum 32*chunk stride.
func compileArchExpr(expr *egraph.Expr, name string, out DType, batchM int) (*Plan, error) {
	plan, err := Compile(expr, name, out)
	if err == nil {
		return plan, nil
	}
	if batchM > 16 || !strings.Contains(err.Error(), "skinny gemm") {
		return nil, err
	}
	const inflatedM = 32
	memo := make(map[*egraph.Expr]*egraph.Expr)
	inflated := rescaleExprBatch(expr, batchM, inflatedM, memo)
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
		t := &p.Temps[i]
		if len(t.Shape) >= 2 && t.Shape[0]%fromM == 0 {
			t.Shape[0] = (t.Shape[0] / fromM) * toM
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
			// Convert eligible non-grouped TileGEMMs back to SkinnyGEMM when K fits.
			if !k.Tile.Grouped && !k.Tile.GLU && !k.Tile.SumSq && k.M <= 16 {
				sg := SkinnyGEMM{
					Name:     k.Tile.Name,
					N:        k.Tile.N,
					K:        k.Tile.K,
					MaxM:     k.M,
					Weight:   k.Tile.Weight,
					Out:      k.Tile.Out,
					Epilogue: k.Tile.Epilogue,
					Extras:   k.Tile.Extras,
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
			cand.WGMMA = false
			cand.Swizzle128B = false
			cand.Repacked = false
			cand.DequantWords = 0
			cand.WarpSpecialized = false
			cand.BlackwellFP4 = true
			if k.M >= 64 || batchM >= 64 {
				cand.GroupM = 8
			}
			if normed, err := cand.norm(); err == nil {
				*k.Tile = normed
			}

		case hopperArch && t.Weight == WeightNVFP4:
			cand := t
			cand.BlackwellFP4 = false
			cand.BM = 128
			cand.BN = 128
			cand.WGMMA = true
			cand.Swizzle128B = true
			cand.Repacked = true
			cand.DequantWords = 2
			cand.GroupM = 8
			if normed, err := cand.norm(); err == nil {
				*k.Tile = normed
			} else {
				cand.Repacked = false
				cand.DequantWords = 0
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
			cand.WGMMA = true
			cand.Swizzle128B = true
			cand.GroupM = 8
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
		out[i] = CostEstimate{
			Name:        name,
			Op:          op,
			Schedule:    sched,
			Bytes:       bytes,
			FLOPs:       flops,
			EstimatedNs: ns,
		}
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
		r := k.Row
		name, op = r.Name, "row_kernel"
		parts = append(parts, fmt.Sprintf("d=%d", r.D))
		if r.Cluster {
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

	mode := ModeMultiLaunch
	switch {
	case batchM <= 8:
		mode = ModePersistentMegakernel
	case batchM >= 64:
		mode = ModeStreamKWarpSpec
	}

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
		Config:           cfg,
		BatchM:           batchM,
		SeqLen:           seqLen,
		Device:           device,
		Mode:             mode,
		ExecutionMode:    mode,
		Plan:             plan,
		Estimates:        estimates,
		TotalNs:          totalNs,
		CUDASource:       cudaSrc,
		MegakernelSource: megaSrc,
		TVMFFISource:     ffiSrc,
		Megakernels:      megas,
	}, nil
}

// buildLayerMegakernels partitions p.Kernels into maximal contiguous segments
// of megakernel-compatible stages (excluding PagedAttention) of length >= 2 and
// lowers each segment via Plan.Megakernel.
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
			if tk.WarpSpecialized {
				tk.WarpSpecialized = false
			}
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
		megaName := fmt.Sprintf("%s_mega%d", p.Name, idx+1)
		m, err := sub.Megakernel(megaName)
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
			if outSet[a] {
				continue
			}
			if !produced[a] && !seenIn[a] {
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
		if len(lastOuts) > 1 {
			for _, extraOut := range lastOuts[1:] {
				if t, ok := tensorByName[extraOut]; ok && !seenTmp[extraOut] {
					seenTmp[extraOut] = true
					temps = append(temps, t)
				}
			}
		}
	}
	return &Plan{
		Name:    name,
		Inputs:  inputs,
		Temps:   temps,
		Output:  outTensor,
		Kernels: seg,
	}
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
	case pk.TopK != nil:
		if len(pk.Args) >= 2 {
			return pk.Args[len(pk.Args)-2:]
		}
	case pk.Align != nil:
		if len(pk.Args) >= 3 {
			return pk.Args[len(pk.Args)-3:]
		}
	case pk.ExpertSum != nil:
		if len(pk.Args) >= 1 {
			return []string{pk.Args[len(pk.Args)-1]}
		}
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
	trimmedMega := strings.TrimPrefix(megaSrc, Prelude+"\n")
	return src + "\n" + trimmedMega, nil
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
	if len(buckets) == 0 {
		return nil, fmt.Errorf("compile multi-bucket layer: empty buckets")
	}
	seen := make(map[int]bool, len(buckets))
	var sorted []int
	for _, m := range buckets {
		if m <= 0 {
			return nil, fmt.Errorf("compile multi-bucket layer: bucket M %d must be positive", m)
		}
		if !seen[m] {
			seen[m] = true
			sorted = append(sorted, m)
		}
	}
	slices.Sort(sorted)
	if strings.TrimSpace(device) == "" {
		device = "NVIDIA H200"
	}

	layers := make(map[int]*CompiledLayer, len(sorted))
	ordered := make([]*CompiledLayer, 0, len(sorted))
	var allRows []RowKernel

	for _, m := range sorted {
		planName := fmt.Sprintf("layer_m%d", m)
		cl, err := compileLayerNamed(cfg, m, seqLen, device, planName)
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
	}

	cudaSrc, err := buildMultiBucketCUDASource(sorted, layers)
	if err != nil {
		return nil, err
	}
	var ffiSrc string
	if len(allRows) > 0 {
		ffiSrc, err = SourceTVMFFI(allRows...)
		if err != nil {
			return nil, err
		}
	}

	return &MultiBucketBundle{
		Config:       cfg,
		Device:       device,
		SeqLen:       seqLen,
		Buckets:      sorted,
		Layers:       layers,
		Ordered:      ordered,
		DispatchFunc: "dispatch_layer_m",
		CUDASource:   cudaSrc,
		TVMFFISource: ffiSrc,
	}, nil
}

func buildMultiBucketCUDASource(buckets []int, layers map[int]*CompiledLayer) (string, error) {
	var b strings.Builder
	b.WriteString(Prelude)

	for _, m := range buckets {
		cl := layers[m]
		pSrc, err := cl.Plan.Source()
		if err != nil {
			return "", err
		}
		pSrc = strings.TrimPrefix(pSrc, Prelude)
		fmt.Fprintf(&b, "\n// === Bucket M <= %d (mode: %s, est: %.1f ns) ===\n", m, cl.Mode, cl.TotalNs)
		b.WriteString(pSrc)
	}

	// Build the shape dispatcher entry point dispatch_layer_m(int M, ...).
	refPlan := layers[buckets[len(buckets)-1]].Plan
	var params []string
	var callArgs []string
	for _, t := range refPlan.Inputs {
		params = append(params, "const "+t.DType.cType()+"* "+t.Name)
		callArgs = append(callArgs, t.Name)
	}
	for _, t := range append(slices.Clone(refPlan.Temps), refPlan.Output) {
		params = append(params, t.DType.cType()+"* "+t.Name)
		callArgs = append(callArgs, t.Name)
	}
	paramSig := strings.Join(append([]string{"int M"}, params...), ", ")
	argList := strings.Join(callArgs, ", ")

	fmt.Fprintf(&b, "\ninline void dispatch_layer_m(%s, cudaStream_t stream) {\n", paramSig)
	for i, m := range buckets {
		runFn := fmt.Sprintf("%s_run", layers[m].Plan.Name)
		if i < len(buckets)-1 {
			fmt.Fprintf(&b, "    if (M <= %d) {\n        %s(%s, stream);\n        return;\n    }\n", m, runFn, argList)
		} else {
			fmt.Fprintf(&b, "    %s(%s, stream);\n", runFn, argList)
		}
	}
	b.WriteString("}\n")
	fmt.Fprintf(&b, "\nextern \"C\" void dispatch_layer_m_c(%s, cudaStream_t stream) {\n    dispatch_layer_m(M, %s, stream);\n}\n",
		paramSig, argList)
	return b.String(), nil
}
