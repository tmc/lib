package hopper

import (
	"fmt"
	"strings"
)

// ArchSM107F is the canonical target string for Vera Rubin family compilation.
const ArchSM107F Arch = TargetSM107F

// Additional quantization formats supported by register-dequantized cross-layer
// weight-streaming megakernels alongside FormatNVFP4, FormatFP8E4M3, and FormatMXFP4.
const (
	FormatQ4_0 QuantFormat = 10 + iota
	FormatNF4
	FormatAWQ4
	FormatGPTQ4
)

func (f QuantFormat) helperName() string {
	switch f {
	case FormatNVFP4:
		return "nvfp4"
	case FormatFP8E4M3:
		return "fp8_e4m3"
	case FormatMXFP4:
		return "mxfp4"
	case FormatQ4_0:
		return "q4_0"
	case FormatNF4:
		return "nf4"
	case FormatAWQ4:
		return "awq4"
	case FormatGPTQ4:
		return "gptq4"
	default:
		return "quant"
	}
}

func (f QuantFormat) valid() bool {
	switch f {
	case FormatQ4_0, FormatNF4, FormatFP8E4M3, FormatMXFP4, FormatNVFP4, FormatAWQ4, FormatGPTQ4:
		return true
	default:
		return false
	}
}

func (f QuantFormat) defaultGroupSize() int {
	switch f {
	case FormatNVFP4:
		return 16
	case FormatMXFP4, FormatQ4_0:
		return 32
	case FormatNF4:
		return 64
	case FormatFP8E4M3, FormatAWQ4, FormatGPTQ4:
		return 128
	default:
		return 32
	}
}

func (f QuantFormat) bitsPerWeight() int {
	switch f {
	case FormatFP8E4M3:
		return 8
	default:
		return 4
	}
}

// scaleBytesPerGroup returns the metadata overhead in bytes per quantization group
// (block scale + optional zero-point).
func (f QuantFormat) scaleBytesPerGroup() float64 {
	switch f {
	case FormatNVFP4, FormatMXFP4:
		return 1.0 // UE4M3 or E8M0 1-byte micro-block scale
	case FormatQ4_0, FormatNF4:
		return 2.0 // FP16/BF16 per-group scale
	case FormatAWQ4, FormatGPTQ4:
		return 2.5 // FP16 scale (2B) + 4-bit zero-point (0.5B)
	case FormatFP8E4M3:
		return 2.0 // FP16/BF16 block scale
	default:
		return 2.0
	}
}

// QuantSkinnyGEMM describes a register-dequantized weight-streaming GEMM stage
// for decode and small-batch cross-layer megakernels.
type QuantSkinnyGEMM struct {
	Name      string
	Format    QuantFormat
	GroupSize int
	M         int
	N         int
	K         int
	MaxM      int
	GLU       bool
	Out       DType
	Epilogue  Lambda
	Extras    []Operand
	Arch      Arch
}

// WeightBytes returns the quantized storage footprint in bytes for W [N, K],
// including group scales and per-tensor global scale metadata.
func (q QuantSkinnyGEMM) WeightBytes() int64 {
	elems := int64(q.N) * int64(q.K)
	if elems <= 0 {
		return 0
	}
	gs := q.GroupSize
	if gs <= 0 {
		gs = q.Format.defaultGroupSize()
	}
	dataBytes := (elems*int64(q.Format.bitsPerWeight()) + 7) / 8
	numGroups := (elems + int64(gs) - 1) / int64(gs)
	scaleBytes := int64(float64(numGroups)*q.Format.scaleBytesPerGroup() + 0.5)
	var globalBytes int64
	if q.Format == FormatNVFP4 {
		globalBytes = 4
	}
	return dataBytes + scaleBytes + globalBytes
}

// emitQuantRegisterDequantHelper returns the inline device helper that performs
// in-register weight dequantization for fmtName and stageName.
func emitQuantRegisterDequantHelper(fmtName QuantFormat, groupSize int, arch Arch, stageName string) string {
	if groupSize <= 0 {
		groupSize = fmtName.defaultGroupSize()
	}
	if arch == "" {
		arch = ArchSM90A
	}
	cleanFmt := strings.ReplaceAll(fmtName.helperName(), "-", "_")
	var body string
	switch fmtName {
	case FormatNVFP4, FormatMXFP4:
		body = `    uint32_t bf16x4[2];
    eg_e2m1x4_to_bf16x4(packed_bits, bf16x4);
    return eg_bf16_to_f32(bf16x4[0]) * group_scale * act;`
	case FormatFP8E4M3:
		body = `    const float w = eg_e4m3_to_f32((uint8_t)(packed_bits & 0xffu)) * group_scale;
    return w * act;`
	case FormatAWQ4, FormatGPTQ4:
		body = `    const int q = (int)(packed_bits & 0xfu);
    const int zp = (int)((packed_bits >> 4) & 0xfu);
    return (float)(q - zp) * group_scale * act;`
	default: // FormatQ4_0, FormatNF4
		body = `    const int q = (int)(packed_bits & 0xfu) - 8;
    return (float)q * group_scale * act;`
	}

	var b strings.Builder
	fmt.Fprintf(&b, "#ifndef EG_QUANT_GEMM_HELPER_%s\n", strings.ToUpper(cleanFmt))
	fmt.Fprintf(&b, "#define EG_QUANT_GEMM_HELPER_%s\n", strings.ToUpper(cleanFmt))
	fmt.Fprintf(&b, "EG_DEV float eg_quant_gemm_%s_dev(uint32_t packed_bits, float group_scale, float act) {\n%s\n}\n", cleanFmt, body)
	fmt.Fprintf(&b, "#endif\n")
	if stageName != "" {
		fmt.Fprintf(&b, "EG_DEV float eg_quant_gemm_%s_dev(uint32_t packed_bits, float group_scale, float act) {\n", stageName)
		fmt.Fprintf(&b, "    return eg_quant_gemm_%s_dev(packed_bits, group_scale, act);\n}\n", cleanFmt)
	}
	return b.String()
}

// attachHelperToLambda embeds a device helper block into l's comment prefix so
// that Megakernel.CUDA() extracts and emits the helper in the translation unit
// while preserving l's AST for expression code generation.
func attachHelperToLambda(l Lambda, helper, tag string) Lambda {
	if l.root == nil {
		l = Identity
	}
	base := l.src
	if base == "" {
		base = "$0"
	}
	l.src = base + "\n" + strings.TrimSpace(helper) + "\n// " + tag
	return l
}

// QuantCrossLayerConfig configures a multi-layer weight-streaming quantized
// decoder stack and its persistent cross-layer megakernel.
type QuantCrossLayerConfig struct {
	NamePrefix string
	Format     QuantFormat // FormatNVFP4, FormatMXFP4, FormatFP8E4M3, FormatAWQ4, etc.
	GroupSize  int
	BatchM     int
	HiddenDim  int
	IntermDim  int
	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	NumLayers  int
	Arch       Arch
}

func (c QuantCrossLayerConfig) norm() (QuantCrossLayerConfig, error) {
	c.NamePrefix = strings.TrimSpace(c.NamePrefix)
	if c.NamePrefix == "" {
		c.NamePrefix = "quant_cross"
	}
	if !cIdent(c.NamePrefix) {
		return c, fmt.Errorf("quant cross-layer: bad NamePrefix %q", c.NamePrefix)
	}
	if c.Format == 0 {
		c.Format = FormatNVFP4
	}
	if !c.Format.valid() {
		return c, fmt.Errorf("quant cross-layer %s: unsupported Format %v", c.NamePrefix, c.Format)
	}
	if c.GroupSize <= 0 {
		c.GroupSize = c.Format.defaultGroupSize()
	}
	if c.BatchM <= 0 {
		c.BatchM = 1
	}
	if c.BatchM > 16 {
		return c, fmt.Errorf("quant cross-layer %s: BatchM %d outside 1..16", c.NamePrefix, c.BatchM)
	}
	if c.HiddenDim <= 0 {
		c.HiddenDim = 256
	}
	if c.IntermDim <= 0 {
		c.IntermDim = 2 * c.HiddenDim
	}
	if c.HeadDim <= 0 {
		if c.NumQHeads > 0 && c.HiddenDim%c.NumQHeads == 0 {
			c.HeadDim = c.HiddenDim / c.NumQHeads
		} else {
			c.HeadDim = 64
		}
	}
	if c.NumQHeads <= 0 {
		c.NumQHeads = max(c.HiddenDim/c.HeadDim, 1)
	}
	if c.NumKVHeads <= 0 {
		c.NumKVHeads = c.NumQHeads
	}
	if c.NumQHeads < c.NumKVHeads || c.NumQHeads%c.NumKVHeads != 0 {
		return c, fmt.Errorf("quant cross-layer %s: NumQHeads %d must be a multiple of NumKVHeads %d",
			c.NamePrefix, c.NumQHeads, c.NumKVHeads)
	}
	if c.NumLayers <= 0 {
		c.NumLayers = 2
	}
	if c.Arch == "" {
		c.Arch = ArchSM90A
	}
	return c, nil
}

func alignUp(v, mult int) int {
	if mult <= 1 {
		return max(v, 1)
	}
	if v <= 0 {
		return mult
	}
	return ((v + mult - 1) / mult) * mult
}

func validRowWidth(d int) int {
	if d < 64 {
		return 64
	}
	if d <= 1024 {
		return alignUp(d, 64)
	}
	return alignUp(d, 512)
}

func buildQuantLayerPlansAt(cfg QuantCrossLayerConfig, layerIdx int) (DecoderLayerPlans, error) {
	const eps = 1e-5
	m := cfg.BatchM
	h := validRowWidth(cfg.HiddenDim)
	inter := alignUp(cfg.IntermDim, 256)
	qDim := alignUp(cfg.NumQHeads*cfg.HeadDim, 256)
	kvDim := alignUp(cfg.NumKVHeads*cfg.HeadDim, 64)
	qkvDim := alignUp(qDim+2*kvDim, 32)

	lPrefix := fmt.Sprintf("l%d_", layerIdx)

	makeStageGEMM := func(stageBase, xName, wBase, outName string, n, k int, outDType DType, baseEpi Lambda, extras []Operand) (PlanKernel, []Tensor) {
		stageName := lPrefix + stageBase
		helper := emitQuantRegisterDequantHelper(cfg.Format, cfg.GroupSize, cfg.Arch, stageName)
		tag := fmt.Sprintf("register-dequant format=%s group_size=%d arch=%s", cfg.Format, cfg.GroupSize, cfg.Arch)
		epi := attachHelperToLambda(baseEpi, helper, tag)

		useNVFP4Kernel := (cfg.Format == FormatNVFP4 || cfg.Format == FormatMXFP4) && k%1024 == 0
		rpb := 32
		if useNVFP4Kernel {
			rpb = 2
		} else {
			for _, cand := range []int{32, 16, 8, 4, 2, 1} {
				if n%cand == 0 && cand*m <= 256 {
					rpb = cand
					break
				}
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
		var wInputs []Tensor
		args := []string{xName}
		if useNVFP4Kernel {
			g.Weight = WeightNVFP4
			g.BlackwellFP4 = cfg.Arch == ArchSM100A || cfg.Arch == ArchSM103A || cfg.Arch == ArchSM120A || cfg.Arch == ArchSM121A
			wName := lPrefix + wBase
			sfName := lPrefix + wBase + "_sf"
			gsName := lPrefix + wBase + "_gs"
			wInputs = []Tensor{
				{Name: wName, DType: U8, Shape: []int{n, k / 2}},
				{Name: sfName, DType: U8, Shape: []int{n, k / 16}},
				{Name: gsName, DType: F32, Shape: []int{1}},
			}
			args = append(args, wName, sfName, gsName)
		} else {
			wName := lPrefix + wBase
			wInputs = []Tensor{
				{Name: wName, DType: BF16, Shape: []int{n, k}},
			}
			args = append(args, wName)
		}
		for _, ex := range extras {
			args = append(args, ex.Name)
		}
		args = append(args, outName)
		return PlanKernel{GEMM: &g, M: m, Args: args}, wInputs
	}

	// Pre-attention plan (AttnIn): quantized QKV projection reading bridge input "xn".
	qkvOutName := lPrefix + "qkv_out"
	qkvPK, qkvWInputs := makeStageGEMM("qkv_proj", "xn", "w_qkv", qkvOutName, qkvDim, h, F32, Identity, nil)
	attnInInputs := append([]Tensor{{Name: "xn", DType: BF16, Shape: []int{m, h}}}, qkvWInputs...)
	attnIn := &Plan{
		Name:    fmt.Sprintf("%s_l%d_attn_in", cfg.NamePrefix, layerIdx),
		Inputs:  attnInInputs,
		Output:  Tensor{Name: qkvOutName, DType: F32, Shape: []int{m, qkvDim}},
		Kernels: []PlanKernel{qkvPK},
	}

	// Post-attention plan (Post):
	//   1. o_proj (QuantSkinnyGEMM)
	//   2. residual_rmsnorm (RowKernel)
	//   3. gate_proj (QuantSkinnyGEMM)
	//   4. up_swiglu (QuantSkinnyGEMM with fused SwiGLU epilogue over gate_out)
	//   5. down_proj (QuantSkinnyGEMM)
	//   6. post_residual_rmsnorm (RowKernel) -> outputs bridge tensor [m, h] BF16
	attnTensor := Tensor{Name: lPrefix + "attn", DType: BF16, Shape: []int{m, qDim}}
	xResTensor := Tensor{Name: lPrefix + "x", DType: F32, Shape: []int{m, h}}
	ffnNormWTensor := Tensor{Name: lPrefix + "ffn_norm", DType: F32, Shape: []int{h}}
	nextNormWTensor := Tensor{Name: lPrefix + "next_norm", DType: F32, Shape: []int{h}}

	oOutName := lPrefix + "o_out"
	hidNormName := lPrefix + "ffn_in"
	gateOutName := lPrefix + "gate_out"
	swigluOutName := lPrefix + "swiglu_out"
	downOutName := lPrefix + "down_out"
	postOutName := lPrefix + "xn_out"

	oPK, oWInputs := makeStageGEMM("o_proj", attnTensor.Name, "w_o", oOutName, h, qDim, F32, Identity, nil)

	resNormLambda := MustLambda(fmt.Sprintf("(mul (mul (add $0 $1) (rsqrt (add (div $2 %d) %g))) $3)", h, eps))
	resNormRow := RowKernel{
		Name: lPrefix + "residual_rmsnorm",
		D:    h,
		Inputs: []RowTensor{
			{Name: "in_o", Buffer: oOutName, DType: F32, Kind: RowData},
			{Name: "in_x", Buffer: xResTensor.Name, DType: F32, Kind: RowData},
			{Name: "in_w", Buffer: ffnNormWTensor.Name, DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "v_sum", Body: MustLambda("(add $0 $1)"), Args: []string{"in_o", "in_x"}},
			{Name: "v_sq", Body: MustLambda("(mul $0 $0)"), Args: []string{"v_sum"}},
			{Name: "r_ss", Reduce: "sum", Body: Identity, Args: []string{"v_sq"}},
			{Name: "v_norm", Body: resNormLambda, Args: []string{"in_o", "in_x", "r_ss", "in_w"}},
		},
		Outputs: []RowTensor{
			{Name: hidNormName, DType: BF16, Kind: RowData, Value: "v_norm"},
		},
	}
	resNormPK := PlanKernel{
		Row:  &resNormRow,
		Rows: m,
		Args: []string{oOutName, xResTensor.Name, ffnNormWTensor.Name, hidNormName},
	}

	gatePK, gateWInputs := makeStageGEMM("gate_proj", hidNormName, "w_gate", gateOutName, inter, h, BF16, Identity, nil)

	swigluEpi := MustLambda("(mul (silu $1) $0)")
	upExtras := []Operand{{Name: gateOutName, DType: BF16, Broadcast: Full}}
	upPK, upWInputs := makeStageGEMM("up_swiglu", hidNormName, "w_up", swigluOutName, inter, h, BF16, swigluEpi, upExtras)

	downPK, downWInputs := makeStageGEMM("down_proj", swigluOutName, "w_down", downOutName, h, inter, F32, Identity, nil)

	postNormRow := RowKernel{
		Name: lPrefix + "post_residual_rmsnorm",
		D:    h,
		Inputs: []RowTensor{
			{Name: "in_d", Buffer: downOutName, DType: F32, Kind: RowData},
			{Name: "in_o", Buffer: oOutName, DType: F32, Kind: RowData},
			{Name: "in_nw", Buffer: nextNormWTensor.Name, DType: F32, Kind: ColVector},
		},
		Values: []RowValue{
			{Name: "v_sum", Body: MustLambda("(add $0 $1)"), Args: []string{"in_d", "in_o"}},
			{Name: "v_sq", Body: MustLambda("(mul $0 $0)"), Args: []string{"v_sum"}},
			{Name: "r_ss", Reduce: "sum", Body: Identity, Args: []string{"v_sq"}},
			{Name: "v_norm", Body: resNormLambda, Args: []string{"in_d", "in_o", "r_ss", "in_nw"}},
		},
		Outputs: []RowTensor{
			{Name: postOutName, DType: BF16, Kind: RowData, Value: "v_norm"},
		},
	}
	postNormPK := PlanKernel{
		Row:  &postNormRow,
		Rows: m,
		Args: []string{downOutName, oOutName, nextNormWTensor.Name, postOutName},
	}

	var postInputs []Tensor
	postInputs = append(postInputs, attnTensor)
	postInputs = append(postInputs, oWInputs...)
	postInputs = append(postInputs, xResTensor, ffnNormWTensor)
	postInputs = append(postInputs, gateWInputs...)
	postInputs = append(postInputs, upWInputs...)
	postInputs = append(postInputs, downWInputs...)
	postInputs = append(postInputs, nextNormWTensor)

	postTemps := []Tensor{
		{Name: oOutName, DType: F32, Shape: []int{m, h}},
		{Name: hidNormName, DType: BF16, Shape: []int{m, h}},
		{Name: gateOutName, DType: BF16, Shape: []int{m, inter}},
		{Name: swigluOutName, DType: BF16, Shape: []int{m, inter}},
		{Name: downOutName, DType: F32, Shape: []int{m, h}},
	}

	post := &Plan{
		Name:    fmt.Sprintf("%s_l%d_post", cfg.NamePrefix, layerIdx),
		Inputs:  postInputs,
		Temps:   postTemps,
		Output:  Tensor{Name: postOutName, DType: BF16, Shape: []int{m, h}},
		Kernels: []PlanKernel{oPK, resNormPK, gatePK, upPK, downPK, postNormPK},
	}

	return DecoderLayerPlans{
		LayerIndex: layerIdx,
		AttnIn:     attnIn,
		Post:       post,
	}, nil
}

// BuildQuantizedLayerPlans constructs the pre-attention (AttnIn) and
// post-attention (Post) plans for a single quantized transformer decoder layer,
// combining register-dequantized projection stages (QKV, O, Gate, Up+SwiGLU,
// Down) with RMSNorm RowKernel stages.
func BuildQuantizedLayerPlans(cfg QuantCrossLayerConfig) (*DecoderLayerPlans, error) {
	normCfg, err := cfg.norm()
	if err != nil {
		return nil, err
	}
	lp, err := buildQuantLayerPlansAt(normCfg, 0)
	if err != nil {
		return nil, err
	}
	return &lp, nil
}

// BuildQuantizedCrossLayerMegakernel builds quantized DecoderLayerPlans across
// cfg.NumLayers, fuses adjacent (Layer[i].Post, Layer[i+1].AttnIn) boundaries
// via [FuseDecoderLayerStack], and lowers each cross-layer boundary into a
// persistent [Megakernel] where weight dequantization happens in registers and
// intermediate activations remain in L2/SMEM.
func BuildQuantizedCrossLayerMegakernel(cfg QuantCrossLayerConfig) (*CrossLayerStack, error) {
	normCfg, err := cfg.norm()
	if err != nil {
		return nil, err
	}
	layers := make([]DecoderLayerPlans, normCfg.NumLayers)
	for i := 0; i < normCfg.NumLayers; i++ {
		lp, err := buildQuantLayerPlansAt(normCfg, i)
		if err != nil {
			return nil, err
		}
		layers[i] = lp
	}
	stack, err := FuseDecoderLayerStack(layers, CrossLayerStackConfig{
		NamePrefix:         normCfg.NamePrefix,
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
		PackArena:          true,
		Mega:               true,
	})
	if err != nil {
		return nil, err
	}
	if stack.FinalMegakernel == nil && stack.FinalPost != nil {
		finalMega, err := stack.FinalPost.Megakernel(stack.FinalPost.Name + "_mega")
		if err != nil {
			return nil, err
		}
		stack.FinalMegakernel = &finalMega
	}
	return stack, nil
}

// EstimateQuantizedMegakernelBandwidth computes the total HBM traffic in bytes
// of an unquantized multi-launch FP16 decoder stack (fp16Bytes) versus the
// persistent cross-layer quantized megakernel stack (quantBytes) and returns
// their compressionRatio = float64(fp16Bytes) / float64(quantBytes).
func EstimateQuantizedMegakernelBandwidth(cfg QuantCrossLayerConfig) (fp16Bytes, quantBytes int64, compressionRatio float64) {
	normCfg, err := cfg.norm()
	if err != nil {
		return 0, 0, 0
	}
	m := int64(normCfg.BatchM)
	h := int64(normCfg.HiddenDim)
	inter := int64(normCfg.IntermDim)
	qDim := int64(normCfg.NumQHeads * normCfg.HeadDim)
	kvDim := int64(normCfg.NumKVHeads * normCfg.HeadDim)
	qkvDim := qDim + 2*kvDim
	layers := int64(normCfg.NumLayers)

	// Projection weight matrices per layer: QKV, O, Gate, Up, Down.
	projDims := [][2]int{
		{ int(qkvDim), int(h) },
		{ int(h), int(qDim) },
		{ int(inter), int(h) },
		{ int(inter), int(h) },
		{ int(h), int(inter) },
	}
	var fp16WeightPerLayer, quantWeightPerLayer int64
	for _, dim := range projDims {
		n, k := dim[0], dim[1]
		fp16WeightPerLayer += int64(n) * int64(k) * 2
		qg := QuantSkinnyGEMM{
			Format:    normCfg.Format,
			GroupSize: normCfg.GroupSize,
			N:         n,
			K:         k,
		}
		quantWeightPerLayer += qg.WeightBytes()
	}

	// Two RMSNorm weight vectors per layer (pre-attn + FFN), 2 bytes/elem in FP16/BF16.
	normWeightsPerLayer := 2 * h * 2

	// Unfused multi-launch FP16 reads and writes intermediate activations to HBM
	// across all 7 stage launches per layer:
	//   1. Pre-attn RMSNorm: read [M,H], write [M,H]
	//   2. QKV proj:         read [M,H], write [M,QKV]
	//   3. O proj:           read [M,Q], write [M,H]
	//   4. Residual RMSNorm: read 2*[M,H], write [M,H]
	//   5. Gate proj:        read [M,H], write [M,I]
	//   6. Up + SwiGLU:      read [M,H]+[M,I], write [M,I]
	//   7. Down + Residual:  read [M,I]+[M,H], write [M,H]
	unfusedActPerLayer := 2 * m * (11*h + qkvDim + qDim + 4*inter)

	fp16Bytes = layers * (fp16WeightPerLayer + normWeightsPerLayer + unfusedActPerLayer)

	// Persistent cross-layer megakernel keeps intra-layer and cross-layer
	// activations in L2/SMEM; only entry input [M,H] and final output [M,H]
	// plus per-layer attention QKV/context handoff touch HBM.
	fusedActTotal := 2*m*h*2 + layers*2*m*(qkvDim+qDim)
	quantBytes = layers*(quantWeightPerLayer+normWeightsPerLayer) + fusedActTotal
	if quantBytes <= 0 {
		return fp16Bytes, 0, 0
	}
	compressionRatio = float64(fp16Bytes) / float64(quantBytes)
	return fp16Bytes, quantBytes, compressionRatio
}
