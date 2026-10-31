// Package sparse implements fine-grained 2:4 structured sparsity combined with
// sub-byte NVFP4/MXFP4 and FP8 E4M3 quantization for persistent megakernels.
package sparse

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// DualSparseQuantConfig configures a multi-layer 2:4 structured sparse +
// low-bit quantized (NVFP4 / MXFP4 / FP8 E4M3) persistent megakernel.
type DualSparseQuantConfig struct {
	NamePrefix       string
	BatchM           int
	HiddenDim        int
	IntermDim        int
	NumLayers        int
	Format           hopper.QuantFormat // FormatNVFP4, FormatFP8E4M3, FormatMXFP4
	GroupSize        int
	ActivationSparse bool // Dynamic ReLU^2 / top-k activation pruning in FFN
	Arch             hopper.Arch
}

func (cfg DualSparseQuantConfig) norm() (DualSparseQuantConfig, error) {
	if strings.TrimSpace(cfg.NamePrefix) == "" {
		cfg.NamePrefix = "dual_sparse_quant"
	}
	if !cIdent(cfg.NamePrefix) {
		return cfg, fmt.Errorf("dual sparse quant: invalid NamePrefix %q", cfg.NamePrefix)
	}
	if cfg.BatchM <= 0 {
		return cfg, fmt.Errorf("dual sparse quant %s: BatchM %d must be >= 1", cfg.NamePrefix, cfg.BatchM)
	}
	if cfg.HiddenDim < 16 || cfg.HiddenDim%4 != 0 {
		return cfg, fmt.Errorf("dual sparse quant %s: HiddenDim %d must be a multiple of 4 >= 16", cfg.NamePrefix, cfg.HiddenDim)
	}
	if cfg.IntermDim < 16 || cfg.IntermDim%4 != 0 {
		return cfg, fmt.Errorf("dual sparse quant %s: IntermDim %d must be a multiple of 4 >= 16", cfg.NamePrefix, cfg.IntermDim)
	}
	if cfg.NumLayers <= 0 {
		return cfg, fmt.Errorf("dual sparse quant %s: NumLayers %d must be >= 1", cfg.NamePrefix, cfg.NumLayers)
	}
	if cfg.Format == 0 {
		cfg.Format = hopper.FormatNVFP4
	}
	switch cfg.Format {
	case hopper.FormatNVFP4, hopper.FormatFP8E4M3, hopper.FormatMXFP4:
	default:
		return cfg, fmt.Errorf("dual sparse quant %s: unsupported QuantFormat %v", cfg.NamePrefix, cfg.Format)
	}
	if cfg.GroupSize <= 0 {
		cfg.GroupSize = 16
	}
	if strings.TrimSpace(cfg.Arch) == "" {
		cfg.Arch = hopper.ArchSM90A
	}
	return cfg, nil
}

func cIdent(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		ok := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || (i > 0 && c >= '0' && c <= '9')
		if !ok {
			return false
		}
	}
	return true
}

// selectTop2Of4 returns the indices (0 <= idx0 < idx1 <= 3) of the two elements
// in block[0..3] with the largest absolute magnitudes, breaking ties
// deterministically by lower index (0 < 1 < 2 < 3).
func selectTop2Of4(block [4]float32) (idx0, idx1 int) {
	order := [4]int{0, 1, 2, 3}
	var mag [4]float64
	for i := range 4 {
		mag[i] = math.Abs(float64(block[i]))
	}
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			if mag[order[j]] > mag[order[i]] || (mag[order[j]] == mag[order[i]] && order[j] < order[i]) {
				order[i], order[j] = order[j], order[i]
			}
		}
	}
	idx0, idx1 = order[0], order[1]
	if idx0 > idx1 {
		idx0, idx1 = idx1, idx0
	}
	return idx0, idx1
}

// Compress2To4QuantizedWeights prunes each contiguous 4-element block of
// weights [rows, cols] to its top-2 magnitude elements (2:4 structured
// sparsity), encodes the 2-bit selector metadata E into metaE (4 groups of
// 2x2-bit selectors per uint16 word), and quantizes the surviving 50%
// non-zeros into FormatNVFP4 (4-bit E2M1), FormatMXFP4 (4-bit E2M1 with UE8M0
// power-of-two scale), or FormatFP8E4M3 (8-bit E4M3).
//
// Compared to an uncompressed FP16 [rows, cols] weight matrix (2 * rows * cols
// bytes):
//   - FormatNVFP4 / FormatMXFP4 packs 2 surviving 4-bit non-zeros per 4-element
//     dense block into 1 byte (rows * cols / 4 bytes total), achieving an 8x
//     weight footprint reduction over FP16.
//   - FormatFP8E4M3 stores 2 surviving 8-bit non-zeros per 4-element dense
//     block in 2 bytes (rows * cols / 2 bytes total), achieving a 4x weight
//     footprint reduction over FP16.
func Compress2To4QuantizedWeights(
	weights []float32,
	rows, cols int,
	qfmt hopper.QuantFormat,
) (packedNonZeros []byte, metaE []uint16, scales []float32, err error) {
	if rows <= 0 || cols <= 0 || cols%4 != 0 {
		return nil, nil, nil, fmt.Errorf("compress 2:4 quantized weights: invalid shape [%d, %d] (cols must be a positive multiple of 4)", rows, cols)
	}
	if len(weights) != rows*cols {
		return nil, nil, nil, fmt.Errorf("compress 2:4 quantized weights: len(weights)=%d, want %d", len(weights), rows*cols)
	}
	switch qfmt {
	case hopper.FormatNVFP4, hopper.FormatFP8E4M3, hopper.FormatMXFP4:
	default:
		return nil, nil, nil, fmt.Errorf("compress 2:4 quantized weights: unsupported QuantFormat %v", qfmt)
	}

	totalGroups := (rows * cols) / 4
	totalNonZeros := (rows * cols) / 2
	nonZeros := make([]float32, totalNonZeros)
	metaWords := (totalGroups + 3) / 4
	metaE = make([]uint16, metaWords)

	for g := range totalGroups {
		base := 4 * g
		blk := [4]float32{weights[base], weights[base+1], weights[base+2], weights[base+3]}
		idx0, idx1 := selectTop2Of4(blk)
		nonZeros[2*g] = blk[idx0]
		nonZeros[2*g+1] = blk[idx1]

		// 2-bit selector per non-zero element -> 4-bit nibble per 4-element block.
		sel := uint16(idx0&0x3) | (uint16(idx1&0x3) << 2)
		metaE[g/4] |= sel << (4 * (g % 4))
	}

	// Quantize surviving non-zeros in contiguous scale groups of up to 16 non-zeros
	// (corresponding to 32 dense elements along K).
	const scaleGroup = 16
	numScaleGroups := (totalNonZeros + scaleGroup - 1) / scaleGroup
	scales = make([]float32, numScaleGroups)

	switch qfmt {
	case hopper.FormatNVFP4, hopper.FormatMXFP4:
		packedNonZeros = make([]byte, totalNonZeros/2) // 2 nibbles per byte = totalGroups bytes
		for sg := range numScaleGroups {
			start := sg * scaleGroup
			end := min(start+scaleGroup, totalNonZeros)
			var amax float32
			for i := start; i < end; i++ {
				if a := float32(math.Abs(float64(nonZeros[i]))); a > amax {
					amax = a
				}
			}
			var s float32
			if amax == 0 {
				s = 1.0
			} else if qfmt == hopper.FormatMXFP4 {
				s = hopper.FromUE8M0(hopper.ToUE8M0(amax / 6.0))
			} else {
				s = amax / 6.0
			}
			if !(s > 0) {
				s = 1.0
			}
			scales[sg] = s
			invS := 1.0 / s
			for i := start; i < end; i += 2 {
				q0 := hopper.EncodeE2M1(nonZeros[i] * invS)
				q1 := hopper.EncodeE2M1(nonZeros[i+1] * invS)
				packedNonZeros[i/2] = (q0 & 0x0f) | ((q1 & 0x0f) << 4)
			}
		}

	case hopper.FormatFP8E4M3:
		packedNonZeros = make([]byte, totalNonZeros) // 1 byte per surviving non-zero
		for sg := range numScaleGroups {
			start := sg * scaleGroup
			end := min(start+scaleGroup, totalNonZeros)
			var amax float32
			for i := start; i < end; i++ {
				if a := float32(math.Abs(float64(nonZeros[i]))); a > amax {
					amax = a
				}
			}
			s := amax / 448.0
			if !(s > 0) {
				s = 1.0
			}
			scales[sg] = s
			invS := 1.0 / s
			for i := start; i < end; i++ {
				packedNonZeros[i] = hopper.E4M3Bits(nonZeros[i] * invS)
			}
		}
	}

	return packedNonZeros, metaE, scales, nil
}

// Decompress2To4QuantizedWeights reconstructs the dense [rows, cols] float32
// weight matrix from the dual-compressed 2:4 sparse non-zeros, 2-bit selector
// metadata metaE, and per-group scales produced by [Compress2To4QuantizedWeights].
func Decompress2To4QuantizedWeights(
	packedNonZeros []byte,
	metaE []uint16,
	scales []float32,
	rows, cols int,
	qfmt hopper.QuantFormat,
) ([]float32, error) {
	if rows <= 0 || cols <= 0 || cols%4 != 0 {
		return nil, fmt.Errorf("decompress 2:4 quantized weights: invalid shape [%d, %d]", rows, cols)
	}
	totalGroups := (rows * cols) / 4
	totalNonZeros := (rows * cols) / 2
	wantMeta := (totalGroups + 3) / 4
	if len(metaE) != wantMeta {
		return nil, fmt.Errorf("decompress 2:4 quantized weights: len(metaE)=%d, want %d", len(metaE), wantMeta)
	}
	const scaleGroup = 16
	wantScales := (totalNonZeros + scaleGroup - 1) / scaleGroup
	if len(scales) != wantScales {
		return nil, fmt.Errorf("decompress 2:4 quantized weights: len(scales)=%d, want %d", len(scales), wantScales)
	}

	nonZeros := make([]float32, totalNonZeros)
	switch qfmt {
	case hopper.FormatNVFP4, hopper.FormatMXFP4:
		if len(packedNonZeros) != totalNonZeros/2 {
			return nil, fmt.Errorf("decompress 2:4 quantized weights: len(packedNonZeros)=%d, want %d", len(packedNonZeros), totalNonZeros/2)
		}
		for g := range totalGroups {
			b := packedNonZeros[g]
			q0 := b & 0x0f
			q1 := (b >> 4) & 0x0f
			s0 := scales[(2*g)/scaleGroup]
			s1 := scales[(2*g+1)/scaleGroup]
			nonZeros[2*g] = hopper.FromE2M1(q0) * s0
			nonZeros[2*g+1] = hopper.FromE2M1(q1) * s1
		}
	case hopper.FormatFP8E4M3:
		if len(packedNonZeros) != totalNonZeros {
			return nil, fmt.Errorf("decompress 2:4 quantized weights: len(packedNonZeros)=%d, want %d", len(packedNonZeros), totalNonZeros)
		}
		for i := range totalNonZeros {
			nonZeros[i] = hopper.FromE4M3Bits(packedNonZeros[i]) * scales[i/scaleGroup]
		}
	default:
		return nil, fmt.Errorf("decompress 2:4 quantized weights: unsupported QuantFormat %v", qfmt)
	}

	dense := make([]float32, rows*cols)
	for g := range totalGroups {
		nibble := (metaE[g/4] >> (4 * (g % 4))) & 0x0f
		idx0 := int(nibble & 0x03)
		idx1 := int((nibble >> 2) & 0x03)
		if idx0 >= idx1 {
			return nil, fmt.Errorf("decompress 2:4 quantized weights: group %d has invalid 2:4 selector indices (%d, %d)", g, idx0, idx1)
		}
		base := 4 * g
		dense[base+idx0] = nonZeros[2*g]
		dense[base+idx1] = nonZeros[2*g+1]
	}
	return dense, nil
}

// DualCompressionFootprint returns the dense FP16 byte size, the compressed
// non-zero weight payload byte size, and the resulting compression ratio
// (8.0x for FormatNVFP4 / FormatMXFP4, 4.0x for FormatFP8E4M3).
func DualCompressionFootprint(rows, cols int, qfmt hopper.QuantFormat) (denseFP16Bytes, sparseQuantBytes int, ratio float64) {
	denseFP16Bytes = rows * cols * 2
	switch qfmt {
	case hopper.FormatNVFP4, hopper.FormatMXFP4:
		sparseQuantBytes = (rows * cols) / 4
	case hopper.FormatFP8E4M3:
		sparseQuantBytes = (rows * cols) / 2
	default:
		return denseFP16Bytes, 0, 0
	}
	if sparseQuantBytes > 0 {
		ratio = float64(denseFP16Bytes) / float64(sparseQuantBytes)
	}
	return denseFP16Bytes, sparseQuantBytes, ratio
}

// BuildDualSparseQuantMegakernel lowers a multi-layer decoder stack using
// 2:4 sparse-quantized projections into a persistent [*hopper.Plan] and
// [*hopper.Megakernel], and emits the persistent megakernel CUDA/PTX preview
// containing mma.sp.sync.aligned sparse Tensor Core instructions, 2-bit E
// metadata selector decoding, and in-register NVFP4/FP8 dequantization.
func BuildDualSparseQuantMegakernel(
	cfg DualSparseQuantConfig,
) (*hopper.Plan, *hopper.Megakernel, string, error) {
	normCfg, err := cfg.norm()
	if err != nil {
		return nil, nil, "", err
	}

	m := min(max(normCfg.BatchM, 1), 16)
	h := max(((normCfg.HiddenDim+255)/256)*256, 256)
	inter := max(((normCfg.IntermDim+255)/256)*256, 256)
	const eps = 1e-5

	buildLayerPlan := func(layerIdx int) (*hopper.Plan, error) {
		prefix := fmt.Sprintf("l%d", layerIdx)
		b := hopper.NewBuilder()
		xn := b.Input("xn", hopper.BF16, m, h)
		xRes := b.Input(prefix+"_res_in", hopper.F32, m, h)

		// Sparse-quantized QKV + O projections
		wQKV := b.Input(prefix+"_sp_qkv_w", hopper.BF16, h, h)
		qkv := b.MatMul(xn, wQKV)
		attnScale := b.Input(prefix+"_attn_scale", hopper.F32, m, h)
		attnCtx := b.Cast(hopper.BF16, b.Map(hopper.MustLambda("(mul $0 $1)"), qkv, attnScale))
		wO := b.Input(prefix+"_sp_o_w", hopper.BF16, h, h)
		oProj := b.MatMul(attnCtx, wO)
		attnHid := b.Map(hopper.MustLambda("(add $0 $1)"), oProj, xRes)

		// Post-attention RMSNorm + FFN (with optional ReLU^2 / top-k dynamic activation sparsity)
		ffnNormW := b.Input(prefix+"_ffn_norm_w", hopper.F32, h)
		ffnIn := b.Cast(hopper.BF16, b.RMSNorm(attnHid, ffnNormW, h, eps))

		var act egraph.Id
		if normCfg.ActivationSparse {
			wGate := b.Input(prefix+"_sp_ffn_gate_w", hopper.BF16, inter, h)
			wUp := b.Input(prefix+"_sp_ffn_up_w", hopper.BF16, inter, h)
			gate := b.MatMul(ffnIn, wGate)
			up := b.MatMul(ffnIn, wUp)
			// Dynamic ReLU^2 activation pruning: max(gate, 0)^2 * up
			actMask := b.Input(prefix+"_act_sparse_mask", hopper.F32, m, inter)
			reluSq := b.Map(hopper.MustLambda("(mul (mul $0 $1) $2)"), gate, actMask, up)
			act = b.Cast(hopper.BF16, reluSq)
		} else {
			wGU := b.Input(prefix+"_sp_ffn_gu_w", hopper.BF16, 2*inter, h)
			act = b.SwiGLU(b.MatMul(ffnIn, wGU))
		}

		wDown := b.Input(prefix+"_sp_ffn_down_w", hopper.BF16, h, inter)
		down := b.MatMul(act, wDown)
		postHid := b.Map(hopper.MustLambda("(add $0 $1)"), down, attnHid)
		nextNormW := b.Input(prefix+"_next_norm_w", hopper.F32, h)
		outNode := b.Cast(hopper.BF16, b.RMSNorm(postHid, nextNormW, h, eps))

		planName := fmt.Sprintf("%s_%s", normCfg.NamePrefix, prefix)
		expr, err := hopper.Optimize(b.G, outNode)
		if err != nil {
			return nil, fmt.Errorf("optimize %s: %w", planName, err)
		}
		pl, err := hopper.Compile(expr, planName, hopper.BF16)
		if err != nil {
			return nil, fmt.Errorf("compile %s: %w", planName, err)
		}
		normalizeSparseGEMMRows(pl, m)
		return pl, nil
	}

	var fused *hopper.Plan
	for l := range normCfg.NumLayers {
		lp, err := buildLayerPlan(l)
		if err != nil {
			return nil, nil, "", fmt.Errorf("dual sparse quant layer %d: %w", l, err)
		}
		if fused == nil {
			fused = lp
			continue
		}
		glue := hopper.CrossLayerGlue{
			Name:               fmt.Sprintf("%s_upto_l%d", normCfg.NamePrefix, l),
			LayerLPrefix:       fmt.Sprintf("s%d_", l-1),
			LayerNextPrefix:    fmt.Sprintf("l%d_", l),
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		fused, err = hopper.FuseCrossLayerPlans(fused, lp, glue)
		if err != nil {
			return nil, nil, "", fmt.Errorf("dual sparse quant fuse layer %d: %w", l, err)
		}
		normalizeSparseGEMMRows(fused, m)
	}

	fused.Name = normCfg.NamePrefix
	megaName := normCfg.NamePrefix + "_mega"
	mega, err := fused.Megakernel(megaName)
	if err != nil {
		return nil, nil, "", fmt.Errorf("dual sparse quant megakernel: %w", err)
	}

	baseCUDA, err := mega.CUDA()
	if err != nil {
		return nil, nil, "", fmt.Errorf("dual sparse quant cuda: %w", err)
	}
	preview := emitSparseQuantMegakernelPreview(normCfg, megaName, baseCUDA)
	return fused, &mega, preview, nil
}

func emitSparseQuantMegakernelPreview(cfg DualSparseQuantConfig, megaName, baseCUDA string) string {
	var dequantComment, dequantHelper, mmaSpMnemonic string
	switch cfg.Format {
	case hopper.FormatNVFP4:
		dequantComment = "NVFP4 (4-bit E2M1 + UE4M3/FP32 group scale, 8x compression vs FP16)"
		dequantHelper = `// Register NVFP4 -> BF16x2 dequantization before sparse Tensor Core issue
EG_DEV void eg_sparse_dequant_nvfp4x4(uint16_t packed_e2m1x4, float scale, uint32_t out_bf16x2[2]) {
    uint32_t raw_bf16x4[2];
    eg_e2m1x4_to_bf16x4((uint32_t)packed_e2m1x4, raw_bf16x4);
    const float s = scale;
    const float v0 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[0] & 0xffffu)) * s;
    const float v1 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[0] >> 16)) * s;
    const float v2 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[1] & 0xffffu)) * s;
    const float v3 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[1] >> 16)) * s;
    out_bf16x2[0] = (uint32_t)eg_f32_to_bf16(v0) | ((uint32_t)eg_f32_to_bf16(v1) << 16);
    out_bf16x2[1] = (uint32_t)eg_f32_to_bf16(v2) | ((uint32_t)eg_f32_to_bf16(v3) << 16);
}`
		mmaSpMnemonic = "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"
	case hopper.FormatMXFP4:
		dequantComment = "MXFP4 (4-bit E2M1 + UE8M0 power-of-two block scale, 8x compression vs FP16)"
		dequantHelper = `// Register MXFP4 (UE8M0 power-of-two scale) -> BF16x2 dequantization
EG_DEV void eg_sparse_dequant_mxfp4x4(uint16_t packed_e2m1x4, uint8_t ue8m0_scale, uint32_t out_bf16x2[2]) {
    uint32_t raw_bf16x4[2];
    eg_e2m1x4_to_bf16x4((uint32_t)packed_e2m1x4, raw_bf16x4);
    const float s = __uint_as_float((uint32_t)ue8m0_scale << 23);
    const float v0 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[0] & 0xffffu)) * s;
    const float v1 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[0] >> 16)) * s;
    const float v2 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[1] & 0xffffu)) * s;
    const float v3 = eg_bf16_to_f32((uint16_t)(raw_bf16x4[1] >> 16)) * s;
    out_bf16x2[0] = (uint32_t)eg_f32_to_bf16(v0) | ((uint32_t)eg_f32_to_bf16(v1) << 16);
    out_bf16x2[1] = (uint32_t)eg_f32_to_bf16(v2) | ((uint32_t)eg_f32_to_bf16(v3) << 16);
}`
		mmaSpMnemonic = "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32"
	case hopper.FormatFP8E4M3:
		dequantComment = "FP8 E4M3 (8-bit float8_e4m3fn + group scale, 4x compression vs FP16)"
		dequantHelper = `// Register FP8 E4M3 -> BF16x2 / E4M3x4 dequantization before sparse Tensor Core issue
EG_DEV void eg_sparse_dequant_fp8e4m3x4(uint32_t packed_e4m3x4, float scale, uint32_t out_bf16x2[2]) {
    const float v0 = eg_e4m3_to_f32((uint8_t)(packed_e4m3x4 & 0xffu)) * scale;
    const float v1 = eg_e4m3_to_f32((uint8_t)((packed_e4m3x4 >> 8) & 0xffu)) * scale;
    const float v2 = eg_e4m3_to_f32((uint8_t)((packed_e4m3x4 >> 16) & 0xffu)) * scale;
    const float v3 = eg_e4m3_to_f32((uint8_t)((packed_e4m3x4 >> 24) & 0xffu)) * scale;
    out_bf16x2[0] = (uint32_t)eg_f32_to_bf16(v0) | ((uint32_t)eg_f32_to_bf16(v1) << 16);
    out_bf16x2[1] = (uint32_t)eg_f32_to_bf16(v2) | ((uint32_t)eg_f32_to_bf16(v3) << 16);
}`
		mmaSpMnemonic = "mma.sp.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "// ============================================================================\n")
	fmt.Fprintf(&b, "// Dual-Compressed 2:4 Structured Sparsity + %s Persistent Megakernel\n", cfg.Format)
	fmt.Fprintf(&b, "// Kernel: %s | Arch: %s | Layers: %d | BatchM: %d | H: %d | I: %d | GroupSize: %d\n",
		megaName, cfg.Arch, cfg.NumLayers, cfg.BatchM, cfg.HiddenDim, cfg.IntermDim, cfg.GroupSize)
	fmt.Fprintf(&b, "// Format: %s\n", dequantComment)
	fmt.Fprintf(&b, "// Sparse Tensor Core Instruction: %s\n", mmaSpMnemonic)
	fmt.Fprintf(&b, "// ============================================================================\n\n")

	b.WriteString(dequantHelper)
	b.WriteString("\n\n")
	b.WriteString(`// 2-bit E metadata selector decoding for 2:4 structured sparsity:
// Each 4-bit nibble in meta_e16 encodes idx0 = E[1:0] and idx1 = E[3:2] (0 <= idx0 < idx1 <= 3)
// selecting the 2 active columns out of each 4-element dense K block for mma.sp.sync.aligned.
EG_DEV void eg_decode_2to4_meta_selector(uint16_t meta_e16, int group_idx, int* idx0, int* idx1) {
    const uint16_t nib = (meta_e16 >> (4 * (group_idx & 3))) & 0xfu;
    *idx0 = (int)(nib & 0x3u);
    *idx1 = (int)((nib >> 2) & 0x3u);
}

// Sparse Tensor Core mma.sp.sync.aligned tile dispatch with 2-bit E metadata selector
EG_DEV void eg_mma_sp_2to4_tile(
    float d[4],
    const uint32_t a_sparse[4],
    const uint32_t b_dense[4],
    const float c[4],
    uint32_t meta_e) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0x0;\n\t"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a_sparse[0]), "r"(a_sparse[1]), "r"(a_sparse[2]), "r"(a_sparse[3]),
          "r"(b_dense[0]), "r"(b_dense[1]), "r"(b_dense[2]), "r"(b_dense[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "r"(meta_e)
    );
#else
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}
`)

	if cfg.ActivationSparse {
		b.WriteString(`
// Dynamic ReLU^2 / top-k FFN activation pruning helper: zeroes sub-threshold activations
// and packs 2:4 sparse activation tiles on the fly before down_proj mma.sp.sync.aligned.
EG_DEV float eg_ffn_relu2_topk_prune(float gate, float up, float threshold) {
    const float r = gate > threshold ? gate : 0.0f;
    return (r * r) * up;
}
`)
	}

	b.WriteString("\n")
	b.WriteString(baseCUDA)
	return b.String()
}

func normalizeSparseGEMMRows(pl *hopper.Plan, m int) {
	if pl == nil {
		return
	}
	for i := range pl.Kernels {
		if g := pl.Kernels[i].GEMM; g != nil {
			if g.MaxM < m {
				g.MaxM = max(m, 1)
			}
			for _, r := range []int{32, 16, 8, 4, 2, 1} {
				if g.GLU && r%2 != 0 {
					continue
				}
				if g.N%r == 0 && r*max(m, g.MaxM, 1) <= 256 {
					g.RowsPerBlock = r
					break
				}
			}
		}
	}
	_ = slices.Clone(pl.Kernels)
}
