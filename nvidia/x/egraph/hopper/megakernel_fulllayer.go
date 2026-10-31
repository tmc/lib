package hopper

import (
	"fmt"
	"strings"
)

// FullDecoderLayerSpec specifies the dimensions, paging parameters, and target
// architecture for a single monolithic Transformer decoder layer plan.
type FullDecoderLayerSpec struct {
	LayerIndex int
	BatchM     int
	HiddenDim  int
	IntermDim  int
	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	PageSize   int
	MaxPages   int
	RoPETheta  float32
	RMSEps     float32
	Arch       Arch
}

type resolvedLayerDims struct {
	layerIdx int
	m        int
	d        int
	interm   int
	hq       int
	hkv      int
	hd       int
	qDim     int
	kvDim    int
	qkvDim   int
	pageSize int
	maxPages int
	eps      float32
	theta    float32
	arch     Arch
}

func roundUpMultiple(v, mult int) int {
	if v <= mult {
		return mult
	}
	return (v + mult - 1) / mult * mult
}

func roundUpHiddenDim(d int) int {
	if d <= 1024 {
		return roundUpMultiple(d, 256)
	}
	return roundUpMultiple(d, 512)
}

func chooseSkinnyRowsPerBlock(n, maxM int) int {
	for _, r := range []int{32, 16, 8, 4, 2, 1} {
		if n%r == 0 && r*max(maxM, 1) <= 256 {
			return r
		}
	}
	return 1
}

func (s FullDecoderLayerSpec) resolve() (resolvedLayerDims, error) {
	if s.LayerIndex < 0 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: negative LayerIndex %d", s.LayerIndex)
	}
	if s.BatchM < 0 || s.BatchM > 16 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: BatchM %d outside 1..16", s.BatchM)
	}
	if s.HiddenDim <= 0 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: HiddenDim %d must be > 0", s.HiddenDim)
	}
	if s.IntermDim <= 0 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: IntermDim %d must be > 0", s.IntermDim)
	}
	if s.NumQHeads <= 0 || s.NumKVHeads <= 0 || s.NumQHeads%s.NumKVHeads != 0 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: invalid heads NumQHeads=%d NumKVHeads=%d", s.NumQHeads, s.NumKVHeads)
	}
	if s.HeadDim <= 0 || s.HeadDim%2 != 0 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: invalid HeadDim=%d", s.HeadDim)
	}
	if s.PageSize < 0 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: negative PageSize %d", s.PageSize)
	}
	if s.MaxPages < 0 {
		return resolvedLayerDims{}, fmt.Errorf("full decoder layer spec: negative MaxPages %d", s.MaxPages)
	}

	m := s.BatchM
	if m == 0 {
		m = 1
	}
	pageSize := s.PageSize
	if pageSize == 0 {
		pageSize = 16
	}
	maxPages := s.MaxPages
	if maxPages == 0 {
		maxPages = 4
	}
	eps := s.RMSEps
	if eps <= 0 {
		eps = 1e-5
	}
	theta := s.RoPETheta
	if theta <= 0 {
		theta = 10000.0
	}
	arch := s.Arch
	if arch == "" {
		arch = TargetSM121A
	}

	// PagedAttention operates on 64-element heads (const HeadDim = 64), and
	// SkinnyGEMM requires K to be a multiple of 256. Align the effective hardware
	// dimensions to tile multiples so both tiny test configs and full model
	// configs lower cleanly into hardware stages.
	hd := HeadDim
	hkv := s.NumKVHeads
	gRatio := max(s.NumQHeads/s.NumKVHeads, 1)
	if gRatio > 64 {
		gRatio = 64
	}
	hq := hkv * gRatio
	for hq*hd < 256 || (hq*hd)%256 != 0 {
		if (hq+hkv)/hkv <= 64 {
			hq += hkv
		} else {
			hkv++
			hq = hkv * gRatio
		}
	}

	d := roundUpHiddenDim(s.HiddenDim)
	interm := roundUpMultiple(s.IntermDim, 256)
	qDim := hq * hd
	kvDim := hkv * hd
	qkvDim := qDim + 2*kvDim

	return resolvedLayerDims{
		layerIdx: s.LayerIndex,
		m:        m,
		d:        d,
		interm:   interm,
		hq:       hq,
		hkv:      hkv,
		hd:       hd,
		qDim:     qDim,
		kvDim:    kvDim,
		qkvDim:   qkvDim,
		pageSize: pageSize,
		maxPages: maxPages,
		eps:      eps,
		theta:    theta,
		arch:     arch,
	}, nil
}

// BuildFullDecoderLayerPlan constructs a single monolithic *Plan for one full
// Transformer decoder layer containing all 9 kernels in sequence:
//  1. Row (attn_norm RMSNorm [M, D])
//  2. GEMM (qkv_proj [M, (Hq+2*Hkv)*Hd])
//  3. RoPE (QKNormRoPEKVAppend in-place Q/K RMSNorm + RoPE + paged KV cache append)
//  4. Attention (PagedAttention flash-decoding over paged KV cache)
//  5. GEMM (o_proj [M, D] with residual add into x_attn)
//  6. Row (ffn_norm RMSNorm [M, D])
//  7. GEMM (gate_proj [M, I])
//  8. GEMM (up_proj [M, I] + SwiGLU epilogue)
//  9. GEMM (down_proj [M, D] with residual add into x_next)
func BuildFullDecoderLayerPlan(spec FullDecoderLayerSpec) (*Plan, error) {
	dims, err := spec.resolve()
	if err != nil {
		return nil, err
	}

	pfx := fmt.Sprintf("l%d_", dims.layerIdx)
	rmsBody := MustLambda(fmt.Sprintf("(mul (mul $0 (rsqrt (add (div $1 %d) %g))) $2)",
		dims.d, dims.eps))

	// 1. Pre-attention RMSNorm [M, D]: x (F32) * rms(x) * attn_norm_w (F32) -> x_norm (BF16).
	attnNorm := RowKernel{
		Name: pfx + "attn_norm",
		D:    dims.d,
		Inputs: []RowTensor{
			{Name: "in0", Buffer: "x", DType: F32, Kind: RowData},
			{Name: "in1", Buffer: "attn_norm_w", DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "v_sq", Body: MustLambda("(mul $0 $0)"), Args: []string{"in0"}},
			{Name: "r_ss", Reduce: "sum", Body: Identity, Args: []string{"v_sq"}},
			{Name: "v_out", Body: rmsBody, Args: []string{"in0", "r_ss", "in1"}},
		},
		Outputs: []RowTensor{
			{Name: "x_norm", DType: BF16, Kind: RowData, Value: "v_out"},
		},
	}

	// 2. QKV projection [M, (Hq + 2*Hkv)*Hd]: x_norm (BF16) @ w_qkv^T -> qkv (BF16).
	qkvProj := SkinnyGEMM{
		Name:         pfx + "qkv_proj",
		N:            dims.qkvDim,
		K:            dims.d,
		MaxM:         dims.m,
		RowsPerBlock: chooseSkinnyRowsPerBlock(dims.qkvDim, dims.m),
		Weight:       WeightBF16,
		X:            BF16,
		Out:          BF16,
	}

	// 3. Fused Q/K RMSNorm + RoPE + paged FP8 KV cache append.
	ropeKV := QKNormRoPEKVAppend{
		Name:      pfx + "rope_kv",
		NumHeads:  dims.hq,
		KVHeads:   dims.hkv,
		HeadDim:   dims.hd,
		RotaryDim: dims.hd,
		PageSize:  dims.pageSize,
		MaxPages:  dims.maxPages,
		Eps:       dims.eps,
		QKNorm:    true,
		KVFP8:     true,
		DType:     BF16,
	}

	// 4. PagedAttention flash-decoding over paged KV cache.
	pagedAttn := PagedAttention{
		Name: pfx + "paged_attn",
		Hq:   dims.hq,
		Hkv:  dims.hkv,
		QLen: 1,
		Page: dims.pageSize,
		Q:    BF16,
		Out:  BF16,
	}

	// 5. Output projection [M, D] with residual add (+ x) into x_attn (F32).
	oProj := SkinnyGEMM{
		Name:         pfx + "o_proj",
		N:            dims.d,
		K:            dims.qDim,
		MaxM:         dims.m,
		RowsPerBlock: chooseSkinnyRowsPerBlock(dims.d, dims.m),
		Weight:       WeightBF16,
		X:            BF16,
		Out:          F32,
		Epilogue:     MustLambda("(add $0 $1)"),
		Extras: []Operand{
			{Name: "res_in", DType: F32, Broadcast: Full},
		},
	}

	// 6. Post-attention FFN RMSNorm [M, D]: x_attn (F32) -> ffn_norm_out (BF16).
	ffnNorm := RowKernel{
		Name: pfx + "ffn_norm",
		D:    dims.d,
		Inputs: []RowTensor{
			{Name: "in0", Buffer: "x_attn", DType: F32, Kind: RowData},
			{Name: "in1", Buffer: "ffn_norm_w", DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "v_sq", Body: MustLambda("(mul $0 $0)"), Args: []string{"in0"}},
			{Name: "r_ss", Reduce: "sum", Body: Identity, Args: []string{"v_sq"}},
			{Name: "v_out", Body: rmsBody, Args: []string{"in0", "r_ss", "in1"}},
		},
		Outputs: []RowTensor{
			{Name: "ffn_norm_out", DType: BF16, Kind: RowData, Value: "v_out"},
		},
	}

	// 7. Gate projection [M, I]: ffn_norm_out (BF16) @ w_gate^T -> gate_out (F32).
	gateProj := SkinnyGEMM{
		Name:         pfx + "gate_proj",
		N:            dims.interm,
		K:            dims.d,
		MaxM:         dims.m,
		RowsPerBlock: chooseSkinnyRowsPerBlock(dims.interm, dims.m),
		Weight:       WeightBF16,
		X:            BF16,
		Out:          F32,
	}

	// 8. Up projection [M, I] + SwiGLU epilogue: silu(gate_out) * (ffn_norm_out @ w_up^T) -> swiglu_out (BF16).
	upProj := SkinnyGEMM{
		Name:         pfx + "up_proj",
		N:            dims.interm,
		K:            dims.d,
		MaxM:         dims.m,
		RowsPerBlock: chooseSkinnyRowsPerBlock(dims.interm, dims.m),
		Weight:       WeightBF16,
		X:            BF16,
		Out:          BF16,
		Epilogue:     MustLambda("(mul (silu $1) $0)"),
		Extras: []Operand{
			{Name: "gate_in", DType: F32, Broadcast: Full},
		},
	}

	// 9. Down projection [M, D] with residual add (+ x_attn) into x_next (F32).
	downProj := SkinnyGEMM{
		Name:         pfx + "down_proj",
		N:            dims.d,
		K:            dims.interm,
		MaxM:         dims.m,
		RowsPerBlock: chooseSkinnyRowsPerBlock(dims.d, dims.m),
		Weight:       WeightBF16,
		X:            BF16,
		Out:          F32,
		Epilogue:     MustLambda("(add $0 $1)"),
		Extras: []Operand{
			{Name: "res_in", DType: F32, Broadcast: Full},
		},
	}

	maxKeys := dims.pageSize
	totalPages := dims.m * dims.maxPages
	kvCacheElems := totalPages * dims.pageSize * dims.hkv * dims.hd
	cosSinRows := max(dims.pageSize*dims.maxPages, 16)
	wsBytes := max(pagedAttn.WorkspaceBytes(dims.m, maxKeys), 64)

	plan := &Plan{
		Name: fmt.Sprintf("full_layer_l%d", dims.layerIdx),
		Inputs: []Tensor{
			{Name: "x", DType: F32, Shape: []int{dims.m, dims.d}},
			{Name: "attn_norm_w", DType: F32, Shape: []int{dims.d}},
			{Name: "w_qkv", DType: BF16, Shape: []int{dims.qkvDim, dims.d}},
			{Name: "q_norm_w", DType: BF16, Shape: []int{dims.hd}},
			{Name: "k_norm_w", DType: BF16, Shape: []int{dims.hd}},
			{Name: "cos_sin", DType: F32, Shape: []int{cosSinRows, dims.hd}},
			{Name: "positions", DType: I32, Shape: []int{dims.m}},
			{Name: "seq_ids", DType: I32, Shape: []int{dims.m}},
			{Name: "seqlens", DType: I32, Shape: []int{dims.m}},
			{Name: "kv_positions", DType: I32, Shape: []int{dims.m}},
			{Name: "page_table", DType: I32, Shape: []int{dims.m, dims.maxPages}},
			{Name: "k_scale", DType: F32, Shape: []int{1}},
			{Name: "v_scale", DType: F32, Shape: []int{1}},
			{Name: "w_o", DType: BF16, Shape: []int{dims.d, dims.qDim}},
			{Name: "ffn_norm_w", DType: F32, Shape: []int{dims.d}},
			{Name: "w_gate", DType: BF16, Shape: []int{dims.interm, dims.d}},
			{Name: "w_up", DType: BF16, Shape: []int{dims.interm, dims.d}},
			{Name: "w_down", DType: BF16, Shape: []int{dims.d, dims.interm}},
		},
		Temps: []Tensor{
			{Name: "x_norm", DType: BF16, Shape: []int{dims.m, dims.d}},
			{Name: "qkv", DType: BF16, Shape: []int{dims.m, dims.qkvDim}},
			{Name: "q_out", DType: BF16, Shape: []int{dims.m, dims.qDim}},
			{Name: "k_cache", DType: E4M3, Shape: []int{kvCacheElems}},
			{Name: "v_cache", DType: E4M3, Shape: []int{kvCacheElems}},
			{Name: "attn_out", DType: BF16, Shape: []int{dims.m, dims.qDim}},
			{Name: "attn_ws", DType: E4M3, Shape: []int{wsBytes}},
			{Name: "x_attn", DType: F32, Shape: []int{dims.m, dims.d}},
			{Name: "ffn_norm_out", DType: BF16, Shape: []int{dims.m, dims.d}},
			{Name: "gate_out", DType: F32, Shape: []int{dims.m, dims.interm}},
			{Name: "swiglu_out", DType: BF16, Shape: []int{dims.m, dims.interm}},
		},
		Output: Tensor{Name: "x_next", DType: F32, Shape: []int{dims.m, dims.d}},
		Kernels: []PlanKernel{
			{
				Row:  &attnNorm,
				Rows: dims.m,
				Args: []string{"x", "attn_norm_w", "x_norm"},
			},
			{
				GEMM: &qkvProj,
				M:    dims.m,
				Args: []string{"x_norm", "w_qkv", "qkv"},
			},
			{
				RoPE: &ropeKV,
				M:    dims.m,
				Rows: dims.m,
				Args: []string{
					"qkv", "q_norm_w", "k_norm_w", "cos_sin",
					"positions", "seq_ids", "seqlens", "kv_positions", "page_table",
					"k_scale", "v_scale", "q_out", "k_cache", "v_cache",
				},
			},
			{
				Attention: &pagedAttn,
				Batch:     dims.m,
				MaxKeys:   maxKeys,
				MaxPages:  dims.maxPages,
				Args: []string{
					"q_out", "k_cache", "v_cache", "page_table",
					"seqlens", "k_scale", "v_scale", "attn_out", "attn_ws",
				},
			},
			{
				GEMM: &oProj,
				M:    dims.m,
				Args: []string{"attn_out", "w_o", "x", "x_attn"},
			},
			{
				Row:  &ffnNorm,
				Rows: dims.m,
				Args: []string{"x_attn", "ffn_norm_w", "ffn_norm_out"},
			},
			{
				GEMM: &gateProj,
				M:    dims.m,
				Args: []string{"ffn_norm_out", "w_gate", "gate_out"},
			},
			{
				GEMM: &upProj,
				M:    dims.m,
				Args: []string{"ffn_norm_out", "w_up", "gate_out", "swiglu_out"},
			},
			{
				GEMM: &downProj,
				M:    dims.m,
				Args: []string{"swiglu_out", "w_down", "x_attn", "x_next"},
			},
		},
	}
	return plan, nil
}

func prefixLayerTensorsExceptBridge(p *Plan, prefix, bridge string) {
	names := collectPlanTensorNames(p, bridge)
	rename := make(map[string]string, len(names))
	for n := range names {
		rename[n] = ensureLayerPrefix(n, prefix)
	}
	applyPlanTensorRename(p, rename)
}

// BuildFullLayerMegakernel builds each layer's *Plan via
// [BuildFullDecoderLayerPlan], fuses all L full layers via
// [FuseCrossLayerPlans], and lowers the entire L-layer stack (including
// QKNormRoPEKVAppend and PagedAttention inside every layer) into a single
// persistent *Megakernel with producer-consumer ready[stage] barriers and
// device-resident epoch resolution (eg_ready_resolve_epoch).
func BuildFullLayerMegakernel(name string, specs []FullDecoderLayerSpec) (*Plan, *Megakernel, error) {
	if len(specs) == 0 {
		return nil, nil, fmt.Errorf("full-layer megakernel: empty specs")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		name = "full_layer_mega"
	}
	if !cIdent(name) {
		return nil, nil, fmt.Errorf("full-layer megakernel: bad name %q", name)
	}

	plans := make([]*Plan, len(specs))
	usedIdx := make(map[int]bool, len(specs))
	layerIndices := make([]int, len(specs))
	for i, s := range specs {
		if s.LayerIndex == 0 && i > 0 && usedIdx[0] {
			s.LayerIndex = i
		}
		if usedIdx[s.LayerIndex] {
			return nil, nil, fmt.Errorf("full-layer megakernel: duplicate LayerIndex %d", s.LayerIndex)
		}
		usedIdx[s.LayerIndex] = true
		layerIndices[i] = s.LayerIndex
		p, err := BuildFullDecoderLayerPlan(s)
		if err != nil {
			return nil, nil, fmt.Errorf("full-layer megakernel layer %d: %w", i, err)
		}
		plans[i] = p
	}

	fused := plans[0]
	if len(plans) == 1 {
		fused.Name = name
		mega, err := fused.Megakernel(name)
		if err != nil {
			return nil, nil, err
		}
		return fused, &mega, nil
	}

	for i := 1; i < len(plans); i++ {
		prevIdx := layerIndices[i-1]
		currIdx := layerIndices[i]
		nextPlan := plans[i]
		if i >= 2 {
			// When fusing layer >= 2 onto an already-prefixed multi-layer plan,
			// prefix its non-bridge tensors upfront so every layer keeps its own
			// l<idx>_ namespace.
			nextPlan = clonePlan(nextPlan)
			prefixLayerTensorsExceptBridge(nextPlan, fmt.Sprintf("l%d_", currIdx), "x")
		}
		stepName := name
		if i < len(plans)-1 {
			stepName = fmt.Sprintf("%s_upto_l%d", name, currIdx)
		}
		glue := CrossLayerGlue{
			Name:               stepName,
			LayerLPrefix:       fmt.Sprintf("l%d_", prevIdx),
			LayerNextPrefix:    fmt.Sprintf("l%d_", currIdx),
			BridgeInput:        "x",
			RetainBridgeOutput: true,
		}
		var err error
		fused, err = FuseCrossLayerPlans(fused, nextPlan, glue)
		if err != nil {
			return nil, nil, fmt.Errorf("full-layer megakernel fuse l%d->l%d: %w", prevIdx, currIdx, err)
		}
	}

	mega, err := fused.Megakernel(name)
	if err != nil {
		return nil, nil, err
	}
	return fused, &mega, nil
}
