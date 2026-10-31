package hopper

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

// MLAYoCoLayerKind identifies the attention mechanism and KV-cache policy of a
// decoder layer in an MLA + SWA + YoCo hybrid stack.
type MLAYoCoLayerKind string

const (
	// LayerKindMLA executes Multi-Head Latent Attention with matrix absorption
	// (W_UK absorbed into Q_nope, W_UV absorbed into O_proj) and writes the
	// compressed latent vector c_KV [BatchM, LatentKVLoRA] and decoupled RoPE
	// key k_R [BatchM, DecoupledRoPE].
	LayerKindMLA MLAYoCoLayerKind = "mla"

	// LayerKindSlidingWindow executes windowed PagedAttention over a bounded
	// sliding window of cfg.SlidingWindow tokens.
	LayerKindSlidingWindow MLAYoCoLayerKind = "sliding_window"

	// LayerKindSharedKVYoCo (You-Only-Cache-Once) omits KV projections and KV
	// cache append altogether, binding its attention stage directly to its
	// upstream producer layer's c_KV / k_R (or K / V) cache buffers.
	LayerKindSharedKVYoCo MLAYoCoLayerKind = "shared_kv_yoco"
)

func (k MLAYoCoLayerKind) valid() bool {
	switch k {
	case LayerKindMLA, LayerKindSlidingWindow, LayerKindSharedKVYoCo:
		return true
	default:
		return false
	}
}

// MLAAbsorbedDecode describes a weight-absorbed Multi-Head Latent Attention
// decode stage inside a cross-layer megakernel.
type MLAAbsorbedDecode struct {
	Name          string
	BatchM        int
	HiddenDim     int
	NumHeads      int
	LatentKVLoRA  int
	DecoupledRoPE int
	SharedKV      bool
	ProducerLayer int
	CKVBuffer     string
	KRBuffer      string
	WUKBuffer     string
	WUVBuffer     string
	Arch          Arch
}

// CUDA returns the inline device helper for weight-absorbed MLA decode over
// compressed latent c_KV + decoupled RoPE k_R with absorbed W_UK and W_UV.
func (m MLAAbsorbedDecode) CUDA() (string, error) {
	name := strings.TrimSpace(m.Name)
	if name == "" {
		name = "mla_absorbed_decode"
	}
	dc := max(m.LatentKVLoRA, 16)
	dr := max(m.DecoupledRoPE, 16)
	scale := 1.0 / float32(math.Sqrt(float64(dc+dr)))

	var b strings.Builder
	fmt.Fprintf(&b, "EG_DEV float eg_mla_absorbed_decode_%s_dev(\n", name)
	b.WriteString("    float q_nope_abs, float q_pe, float c_kv, float k_r, float w_uk, float w_uv) {\n")
	fmt.Fprintf(&b, "    const float score = (q_nope_abs * w_uk * c_kv + q_pe * k_r) * %s;\n", floatLiteral(scale))
	b.WriteString("    const float attn_prob = eg_sigmoid(score);\n")
	b.WriteString("    return attn_prob * c_kv * w_uv;\n}")
	return b.String(), nil
}

// MLAYoCoStackConfig configures a multi-layer decoder stack combining
// weight-absorbed MLA layers, sliding-window PagedAttention layers, and YoCo
// KV-sharing layers.
type MLAYoCoStackConfig struct {
	NamePrefix    string
	BatchM        int
	HiddenDim     int
	IntermDim     int
	NumHeads      int
	LatentKVLoRA  int                // d_c (e.g. 512 for DeepSeek-V3)
	DecoupledRoPE int                // d_R (e.g. 64)
	SlidingWindow int                // window size for LayerKindSlidingWindow
	LayerSchedule []MLAYoCoLayerKind // e.g., [LayerKindMLA, LayerKindSlidingWindow, LayerKindMLA, LayerKindSharedKVYoCo]
	Arch          Arch
}

func (c MLAYoCoStackConfig) norm() (MLAYoCoStackConfig, error) {
	c.NamePrefix = strings.TrimSpace(c.NamePrefix)
	if c.NamePrefix == "" {
		c.NamePrefix = "mla_yoco"
	}
	if !cIdent(c.NamePrefix) {
		return c, fmt.Errorf("mla yoco stack: bad NamePrefix %q", c.NamePrefix)
	}
	if c.BatchM <= 0 {
		c.BatchM = 1
	}
	if c.BatchM > 16 {
		return c, fmt.Errorf("mla yoco stack %s: BatchM %d outside 1..16", c.NamePrefix, c.BatchM)
	}
	if c.HiddenDim <= 0 {
		c.HiddenDim = 512
	}
	if c.IntermDim <= 0 {
		c.IntermDim = 2 * c.HiddenDim
	}
	if c.NumHeads <= 0 {
		c.NumHeads = 8
	}
	if c.LatentKVLoRA <= 0 {
		c.LatentKVLoRA = 512
	}
	if c.DecoupledRoPE <= 0 {
		c.DecoupledRoPE = 64
	}
	if c.SlidingWindow <= 0 {
		c.SlidingWindow = 512
	}
	if len(c.LayerSchedule) == 0 {
		return c, fmt.Errorf("mla yoco stack %s: empty LayerSchedule", c.NamePrefix)
	}
	lastProducer := -1
	for i, kind := range c.LayerSchedule {
		if !kind.valid() {
			return c, fmt.Errorf("mla yoco stack %s: layer %d has unsupported kind %q", c.NamePrefix, i, kind)
		}
		if kind == LayerKindSharedKVYoCo {
			if lastProducer < 0 {
				return c, fmt.Errorf("mla yoco stack %s: layer %d is %s with no preceding producer layer",
					c.NamePrefix, i, kind)
			}
		} else {
			lastProducer = i
		}
	}
	if c.Arch == "" {
		c.Arch = ArchSM90A
	}
	return c, nil
}

func buildSingleMLAYoCoLayerPlan(
	cfg MLAYoCoStackConfig,
	layerIdx int,
	kind MLAYoCoLayerKind,
	producerIdx int,
	producerKind MLAYoCoLayerKind,
) (*Plan, map[string]string, error) {
	const eps = 1e-5
	m := cfg.BatchM
	h := validRowWidth(cfg.HiddenDim)
	inter := alignUp(cfg.IntermDim, 256)
	dc := validRowWidth(cfg.LatentKVLoRA)
	dr := validRowWidth(cfg.DecoupledRoPE)
	hq := max(cfg.NumHeads, 1)
	hkv := max(hq/4, 1)
	for hq%hkv != 0 {
		hkv--
	}
	qDim := alignUp(hq*HeadDim, 256)
	lPrefix := fmt.Sprintf("l%d_", layerIdx)

	var inputs []Tensor
	var temps []Tensor
	var kernels []PlanKernel
	sharedKVBindings := make(map[string]string)

	addInput := func(t Tensor) {
		for _, in := range inputs {
			if in.Name == t.Name {
				return
			}
		}
		inputs = append(inputs, t)
	}
	addTemp := func(t Tensor) {
		for _, tmp := range temps {
			if tmp.Name == t.Name {
				return
			}
		}
		temps = append(temps, t)
	}

	makeGEMM := func(stageName, xName, wName, outName string, n, k int, outDType DType, epi Lambda, extras []Operand) PlanKernel {
		rpb := 32
		for _, cand := range []int{32, 16, 8, 4, 2, 1} {
			if n%cand == 0 && cand*m <= 256 {
				rpb = cand
				break
			}
		}
		g := SkinnyGEMM{
			Name:         stageName,
			N:            n,
			K:            k,
			MaxM:         m,
			RowsPerBlock: rpb,
			Weight:       WeightBF16,
			X:            BF16,
			Out:          outDType,
			Epilogue:     epi,
			Extras:       extras,
		}
		addInput(Tensor{Name: wName, DType: BF16, Shape: []int{n, k}})
		args := []string{xName, wName}
		for _, ex := range extras {
			args = append(args, ex.Name)
		}
		args = append(args, outName)
		return PlanKernel{GEMM: &g, M: m, Args: args}
	}

	// 1. Pre-attention RMSNorm reading bridge input "xn".
	addInput(Tensor{Name: "xn", DType: BF16, Shape: []int{m, h}})
	attnNormW := lPrefix + "attn_norm_w"
	addInput(Tensor{Name: attnNormW, DType: F32, Shape: []int{h}})
	norm1Out := lPrefix + "norm1"
	addTemp(Tensor{Name: norm1Out, DType: BF16, Shape: []int{m, h}})

	norm1Lambda := MustLambda(fmt.Sprintf("(mul (mul $0 (rsqrt (add (div $1 %d) %g))) $2)", h, eps))
	norm1Row := RowKernel{
		Name: lPrefix + "pre_attn_rmsnorm",
		D:    h,
		Inputs: []RowTensor{
			{Name: "in_x", Buffer: "xn", DType: BF16, Kind: RowData},
			{Name: "in_w", Buffer: attnNormW, DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "v_sq", Body: MustLambda("(mul $0 $0)"), Args: []string{"in_x"}},
			{Name: "r_ss", Reduce: "sum", Body: Identity, Args: []string{"v_sq"}},
			{Name: "v_out", Body: norm1Lambda, Args: []string{"in_x", "r_ss", "in_w"}},
		},
		Outputs: []RowTensor{
			{Name: norm1Out, DType: BF16, Kind: RowData, Value: "v_out"},
		},
	}
	kernels = append(kernels, PlanKernel{
		Row:  &norm1Row,
		Rows: m,
		Args: []string{"xn", attnNormW, norm1Out},
	})

	attnOutName := lPrefix + "attn_out"
	var attnOutWidth int

	switch {
	case kind == LayerKindMLA || (kind == LayerKindSharedKVYoCo && producerKind == LayerKindMLA):
		attnOutWidth = dc
		mlaStageName := lPrefix + "mla_absorbed_decode"
		if kind == LayerKindSharedKVYoCo {
			mlaStageName = lPrefix + "yoco_shared_kv_mla_decode"
		}
		mlaDesc := MLAAbsorbedDecode{
			Name:          mlaStageName,
			BatchM:        m,
			HiddenDim:     h,
			NumHeads:      hq,
			LatentKVLoRA:  dc,
			DecoupledRoPE: dr,
			SharedKV:      kind == LayerKindSharedKVYoCo,
			ProducerLayer: producerIdx,
			Arch:          cfg.Arch,
		}
		mlaHelper, err := mlaDesc.CUDA()
		if err != nil {
			return nil, nil, err
		}

		// Absorbed Q_nope projection (incorporating W_UK) + Q_pe projection.
		qNopeAbs := lPrefix + "q_nope_abs"
		addTemp(Tensor{Name: qNopeAbs, DType: BF16, Shape: []int{m, dc}})
		qEpi := attachHelperToLambda(Identity, mlaHelper, fmt.Sprintf("mla-absorbed layer=%d shared_kv=%v", layerIdx, mlaDesc.SharedKV))
		kernels = append(kernels, makeGEMM(lPrefix+"q_absorbed_proj", norm1Out, lPrefix+"w_q_uk", qNopeAbs, dc, h, BF16, qEpi, nil))

		var cKVName, kRName string
		if kind == LayerKindMLA {
			// Compute compressed latent KV c_KV [m, dc] and decoupled RoPE key k_R [m, dr].
			cKVName = lPrefix + "c_kv_cache"
			kRName = lPrefix + "k_r_cache"
			wDKV := lPrefix + "w_dkv"
			wKR := lPrefix + "w_kr"
			addInput(Tensor{Name: wDKV, DType: BF16, Shape: []int{dc}})
			addInput(Tensor{Name: wKR, DType: BF16, Shape: []int{dr}})
			addTemp(Tensor{Name: cKVName, DType: BF16, Shape: []int{m, dc}})
			addTemp(Tensor{Name: kRName, DType: BF16, Shape: []int{m, dr}})

			kvCompressRow := RowKernel{
				Name: lPrefix + "kv_lora_compress",
				D:    dc,
				Inputs: []RowTensor{
					{Name: "in_q", Buffer: qNopeAbs, DType: BF16, Kind: RowData},
					{Name: "in_w", Buffer: wDKV, DType: BF16, Kind: ColVector},
				},
				Values: []RowValue{
					{Name: "v_ckv", Body: MustLambda("(mul $0 $1)"), Args: []string{"in_q", "in_w"}},
				},
				Outputs: []RowTensor{
					{Name: cKVName, DType: BF16, Kind: RowData, Value: "v_ckv"},
				},
			}
			kernels = append(kernels, PlanKernel{
				Row:  &kvCompressRow,
				Rows: m,
				Args: []string{qNopeAbs, wDKV, cKVName},
			})

			krRopeRow := RowKernel{
				Name: lPrefix + "kr_decoupled_rope",
				D:    dr,
				Inputs: []RowTensor{
					{Name: "in_wkr", Buffer: wKR, DType: BF16, Kind: ColVector},
				},
				Values: []RowValue{
					{Name: "v_kr", Body: MustLambda("(mul $0 1.0)"), Args: []string{"in_wkr"}},
				},
				Outputs: []RowTensor{
					{Name: kRName, DType: BF16, Kind: RowData, Value: "v_kr"},
				},
			}
			kernels = append(kernels, PlanKernel{
				Row:  &krRopeRow,
				Rows: m,
				Args: []string{wKR, kRName},
			})
		} else {
			// YoCo KV-sharing: omit KV compression and decoupled RoPE stages completely!
			// Use placeholder input tensors that will be rebound to producerIdx's buffers after FuseCrossLayerPlans.
			cKVName = lPrefix + "shared_c_kv_ref"
			kRName = lPrefix + "shared_k_r_ref"
			addInput(Tensor{Name: cKVName, DType: BF16, Shape: []int{m, dc}})
			addInput(Tensor{Name: kRName, DType: BF16, Shape: []int{m, dr}})
			sharedKVBindings[cKVName] = fmt.Sprintf("l%d_c_kv_cache", producerIdx)
			sharedKVBindings[kRName] = fmt.Sprintf("l%d_k_r_cache", producerIdx)
		}

		// Weight-absorbed MLA decode stage reading q_nope_abs, c_kv_cache, k_r_cache, w_uk, w_uv.
		wUKName := lPrefix + "w_uk"
		wUVName := lPrefix + "w_uv"
		addInput(Tensor{Name: wUKName, DType: BF16, Shape: []int{dc}})
		addInput(Tensor{Name: wUVName, DType: BF16, Shape: []int{dc}})
		addTemp(Tensor{Name: attnOutName, DType: BF16, Shape: []int{m, dc}})

		mlaAttnRow := RowKernel{
			Name: mlaStageName,
			D:    dc,
			Inputs: []RowTensor{
				{Name: "in_qn", Buffer: qNopeAbs, DType: BF16, Kind: RowData},
				{Name: "in_ckv", Buffer: cKVName, DType: BF16, Kind: RowData},
				{Name: "in_kr", Buffer: kRName, DType: BF16, Kind: Scalar},
				{Name: "in_wuk", Buffer: wUKName, DType: BF16, Kind: ColVector},
				{Name: "in_wuv", Buffer: wUVName, DType: BF16, Kind: ColVector},
			},
			Values: []RowValue{
				{Name: "v_q_uk", Body: MustLambda("(mul $0 $1)"), Args: []string{"in_qn", "in_wuk"}},
				{Name: "v_score", Body: MustLambda("(add (mul $0 $1) $2)"), Args: []string{"v_q_uk", "in_ckv", "in_kr"}},
				{Name: "v_prob", Body: MustLambda("(sigmoid $0)"), Args: []string{"v_score"}},
				{Name: "v_ctx", Body: MustLambda("(mul (mul $0 $1) $2)"), Args: []string{"v_prob", "in_ckv", "in_wuv"}},
			},
			Outputs: []RowTensor{
				{Name: attnOutName, DType: BF16, Kind: RowData, Value: "v_ctx"},
			},
		}
		kernels = append(kernels, PlanKernel{
			Row:  &mlaAttnRow,
			Rows: m,
			Args: []string{qNopeAbs, cKVName, kRName, wUKName, wUVName, attnOutName},
		})

	default:
		// LayerKindSlidingWindow or LayerKindSharedKVYoCo following a sliding-window producer.
		attnOutWidth = qDim
		qOutName := lPrefix + "q_out"
		addTemp(Tensor{Name: qOutName, DType: BF16, Shape: []int{m, qDim}})
		kernels = append(kernels, makeGEMM(lPrefix+"q_proj", norm1Out, lPrefix+"w_q", qOutName, qDim, h, BF16, Identity, nil))

		const (
			pageSize   = 16
			maxPages   = 4
			maxKeys    = 64
			cacheElems = maxPages * pageSize * 64
		)
		var kCacheName, vCacheName string
		if kind == LayerKindSlidingWindow {
			kCacheName = lPrefix + "k_cache"
			vCacheName = lPrefix + "v_cache"
			wKV := lPrefix + "w_kv"
			addInput(Tensor{Name: wKV, DType: E4M3, Shape: []int{64}})
			addTemp(Tensor{Name: kCacheName, DType: E4M3, Shape: []int{max(m, 1), 64}})
			addTemp(Tensor{Name: vCacheName, DType: E4M3, Shape: []int{max(m, 1), 64}})

			kvAppendRow := RowKernel{
				Name: lPrefix + "swa_kv_append",
				D:    64,
				Inputs: []RowTensor{
					{Name: "in_kv", Buffer: wKV, DType: E4M3, Kind: ColVector},
				},
				Values: []RowValue{
					{Name: "v_k", Body: Identity, Args: []string{"in_kv"}},
				},
				Outputs: []RowTensor{
					{Name: kCacheName, DType: E4M3, Kind: RowData, Value: "v_k"},
					{Name: vCacheName, DType: E4M3, Kind: RowData, Value: "v_k"},
				},
			}
			kernels = append(kernels, PlanKernel{
				Row:  &kvAppendRow,
				Rows: m,
				Args: []string{wKV, kCacheName, vCacheName},
			})
		} else {
			kCacheName = lPrefix + "shared_k_cache_ref"
			vCacheName = lPrefix + "shared_v_cache_ref"
			addInput(Tensor{Name: kCacheName, DType: E4M3, Shape: []int{max(m, 1), 64}})
			addInput(Tensor{Name: vCacheName, DType: E4M3, Shape: []int{max(m, 1), 64}})
			sharedKVBindings[kCacheName] = fmt.Sprintf("l%d_k_cache", producerIdx)
			sharedKVBindings[vCacheName] = fmt.Sprintf("l%d_v_cache", producerIdx)
		}

		ptName := lPrefix + "page_table"
		slName := lPrefix + "seqlens"
		ksName := lPrefix + "k_scale"
		vsName := lPrefix + "v_scale"
		addInput(Tensor{Name: ptName, DType: I32, Shape: []int{m, maxPages}})
		addInput(Tensor{Name: slName, DType: I32, Shape: []int{m}})
		addInput(Tensor{Name: ksName, DType: F32, Shape: []int{1}})
		addInput(Tensor{Name: vsName, DType: F32, Shape: []int{1}})
		addTemp(Tensor{Name: attnOutName, DType: BF16, Shape: []int{m, qDim}})

		swaName := lPrefix + "swa_paged_attn"
		if kind == LayerKindSharedKVYoCo {
			swaName = lPrefix + "yoco_shared_kv_swa_attn"
		}
		pa := PagedAttention{
			Name:   swaName,
			Hq:     hq,
			Hkv:    hkv,
			QLen:   1,
			Window: cfg.SlidingWindow,
			Page:   pageSize,
			Q:      BF16,
			Out:    BF16,
		}
		kernels = append(kernels, PlanKernel{
			Attention: &pa,
			Batch:     m,
			MaxKeys:   maxKeys,
			MaxPages:  maxPages,
			Args: []string{
				qOutName, kCacheName, vCacheName, ptName,
				slName, ksName, vsName, attnOutName,
			},
		})
	}

	// Output projection + FFN SwiGLU + post-layer RMSNorm producing bridge output [m, h] BF16.
	oOutName := lPrefix + "o_out"
	addTemp(Tensor{Name: oOutName, DType: BF16, Shape: []int{m, h}})
	kernels = append(kernels, makeGEMM(lPrefix+"o_proj", attnOutName, lPrefix+"w_o", oOutName, h, attnOutWidth, BF16, Identity, nil))

	ffnGateUp := lPrefix + "swiglu_out"
	addTemp(Tensor{Name: ffnGateUp, DType: BF16, Shape: []int{m, inter}})
	kernels = append(kernels, makeGEMM(lPrefix+"ffn_gate_up", oOutName, lPrefix+"w_gate_up", ffnGateUp, inter, h, BF16, MustLambda("(silu $0)"), nil))

	downOut := lPrefix + "down_out"
	addTemp(Tensor{Name: downOut, DType: F32, Shape: []int{m, h}})
	kernels = append(kernels, makeGEMM(lPrefix+"ffn_down", ffnGateUp, lPrefix+"w_down", downOut, h, inter, F32, Identity, nil))

	postNormW := lPrefix + "post_norm_w"
	addInput(Tensor{Name: postNormW, DType: F32, Shape: []int{h}})
	xnOutName := lPrefix + "xn_out"

	postNormLambda := MustLambda(fmt.Sprintf("(mul (mul $0 (rsqrt (add (div $1 %d) %g))) $2)", h, eps))
	postNormRow := RowKernel{
		Name: lPrefix + "post_ffn_rmsnorm",
		D:    h,
		Inputs: []RowTensor{
			{Name: "in_d", Buffer: downOut, DType: F32, Kind: RowData},
			{Name: "in_w", Buffer: postNormW, DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "v_sq", Body: MustLambda("(mul $0 $0)"), Args: []string{"in_d"}},
			{Name: "r_ss", Reduce: "sum", Body: Identity, Args: []string{"v_sq"}},
			{Name: "v_out", Body: postNormLambda, Args: []string{"in_d", "r_ss", "in_w"}},
		},
		Outputs: []RowTensor{
			{Name: xnOutName, DType: BF16, Kind: RowData, Value: "v_out"},
		},
	}
	kernels = append(kernels, PlanKernel{
		Row:  &postNormRow,
		Rows: m,
		Args: []string{downOut, postNormW, xnOutName},
	})

	plan := &Plan{
		Name:    fmt.Sprintf("%s_l%d", cfg.NamePrefix, layerIdx),
		Inputs:  inputs,
		Temps:   temps,
		Output:  Tensor{Name: xnOutName, DType: BF16, Shape: []int{m, h}},
		Kernels: kernels,
	}
	return plan, sharedKVBindings, nil
}

// BuildMLAYoCoStackMegakernel constructs a multi-layer Plan combining
// weight-absorbed MLA stages (LayerKindMLA), sliding-window PagedAttention
// stages (LayerKindSlidingWindow), and YoCo KV-sharing stages
// (LayerKindSharedKVYoCo), fuses all layers across inter-layer boundaries via
// [FuseCrossLayerPlans], rebinds YoCo layers directly to their producer
// layer's KV cache buffers, and lowers the entire stack into a single
// persistent [Megakernel].
func BuildMLAYoCoStackMegakernel(cfg MLAYoCoStackConfig) (*Plan, *Megakernel, error) {
	normCfg, err := cfg.norm()
	if err != nil {
		return nil, nil, err
	}

	allBindings := make(map[string]string)
	var fusedPlan *Plan
	producerIdx := -1
	var producerKind MLAYoCoLayerKind

	for i, kind := range normCfg.LayerSchedule {
		layerPlan, bindings, err := buildSingleMLAYoCoLayerPlan(normCfg, i, kind, producerIdx, producerKind)
		if err != nil {
			return nil, nil, err
		}
		for k, v := range bindings {
			allBindings[k] = v
		}
		if kind != LayerKindSharedKVYoCo {
			producerIdx = i
			producerKind = kind
		}

		if fusedPlan == nil {
			fusedPlan = layerPlan
			continue
		}
		glue := CrossLayerGlue{
			Name:               fmt.Sprintf("%s_l0_to_l%d", normCfg.NamePrefix, i),
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		fusedPlan, err = FuseCrossLayerPlans(fusedPlan, layerPlan, glue)
		if err != nil {
			return nil, nil, fmt.Errorf("mla yoco stack fuse layer %d: %w", i, err)
		}
	}

	// Rebind YoCo shared-KV placeholders directly to the upstream producer's KV buffers
	// and remove placeholder or duplicate entries from fusedPlan.Inputs.
	if len(allBindings) > 0 {
		applyPlanTensorRename(fusedPlan, allBindings)
		tempNames := make(map[string]bool, len(fusedPlan.Temps))
		for _, tmp := range fusedPlan.Temps {
			tempNames[tmp.Name] = true
		}
		seenInputs := make(map[string]bool, len(fusedPlan.Inputs))
		filteredInputs := make([]Tensor, 0, len(fusedPlan.Inputs))
		for _, in := range fusedPlan.Inputs {
			if tempNames[in.Name] || seenInputs[in.Name] {
				continue
			}
			seenInputs[in.Name] = true
			filteredInputs = append(filteredInputs, in)
		}
		fusedPlan.Inputs = filteredInputs
	}

	megaName := normCfg.NamePrefix + "_stack_mega"
	mega, err := fusedPlan.Megakernel(megaName)
	if err != nil {
		return nil, nil, err
	}
	return fusedPlan, &mega, nil
}

// EstimateMLAYoCoKVBytesPerToken computes the KV-cache footprint per token
// across all layers of cfg.LayerSchedule for standard unshared MHA FP16
// (standardKVBytes) versus the MLA + SWA + YoCo stack (mlaYoCoKVBytes), along
// with reductionFactor = float64(standardKVBytes) / float64(mlaYoCoKVBytes).
func EstimateMLAYoCoKVBytesPerToken(cfg MLAYoCoStackConfig) (standardKVBytes, mlaYoCoKVBytes int64, reductionFactor float64) {
	normCfg, err := cfg.norm()
	if err != nil {
		return 0, 0, 0
	}
	numLayers := int64(len(normCfg.LayerSchedule))
	numHeads := int64(normCfg.NumHeads)
	headDim := max(int64(normCfg.HiddenDim)/max(numHeads, 1), 128)

	// Standard MHA stores K and V [numHeads, headDim] in FP16/BF16 (2 bytes/elem) per layer.
	standardPerLayer := 2 * numHeads * headDim * 2
	standardKVBytes = numLayers * standardPerLayer

	for _, kind := range normCfg.LayerSchedule {
		switch kind {
		case LayerKindMLA:
			// Single shared compressed latent vector c_KV (d_c) + decoupled RoPE key k_R (d_R), 2 bytes/elem.
			mlaYoCoKVBytes += int64(normCfg.LatentKVLoRA+normCfg.DecoupledRoPE) * 2
		case LayerKindSlidingWindow:
			// Single GQA KV head amortized by sliding-window retention vs full context.
			swaBytes := 2 * headDim * 2
			mlaYoCoKVBytes += swaBytes
		case LayerKindSharedKVYoCo:
			// YoCo reuses upstream producer KV cache directly: 0 bytes allocated or written!
		}
	}
	if mlaYoCoKVBytes <= 0 {
		return standardKVBytes, 0, 0
	}
	reductionFactor = float64(standardKVBytes) / float64(mlaYoCoKVBytes)
	return standardKVBytes, mlaYoCoKVBytes, reductionFactor
}

// SharedKVStageBuffers returns the input buffer names read by every attention
// stage in plan, keyed by stage kernel name, for verifying YoCo KV buffer sharing.
func SharedKVStageBuffers(p *Plan) map[string][]string {
	if p == nil {
		return nil
	}
	out := make(map[string][]string, len(p.Kernels))
	for _, k := range p.Kernels {
		out[planKernelName(k)] = slices.Clone(k.Args)
	}
	return out
}
