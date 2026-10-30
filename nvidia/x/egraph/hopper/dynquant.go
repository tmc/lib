package hopper

import (
	"fmt"
	"math"
	"strings"
)

// ScaleLayout selects the memory layout of UE4M3 block scales produced by
// [RMSNormDynamicNVFP4Quant] and [SwiGLUDynamicNVFP4Quant].
type ScaleLayout uint8

const (
	// ScaleLayoutRowMajor stores UE4M3 block scales linearly as [M, K/16] bytes
	// at offset row*(K/16) + kb.
	ScaleLayoutRowMajor ScaleLayout = iota

	// ScaleLayoutTCGen05Swizzled stores UE4M3 block scales in the 32x128b
	// swizzled layout consumed by tcgen05.cp.cta_group::1.32x128b.warpx4
	// (see [SwizzleTCGen05ScaleOffset] and [UnswizzleTCGen05Scales]).
	ScaleLayoutTCGen05Swizzled
)

func (l ScaleLayout) String() string {
	switch l {
	case ScaleLayoutRowMajor:
		return "row_major"
	case ScaleLayoutTCGen05Swizzled:
		return "tcgen05_swizzled"
	}
	return fmt.Sprintf("ScaleLayout(%d)", uint8(l))
}

func (l ScaleLayout) valid() bool {
	return l == ScaleLayoutRowMajor || l == ScaleLayoutTCGen05Swizzled
}

// SwizzleTCGen05ScaleOffset returns the byte offset of scale block (row, kb) in
// the 32x128b swizzled scale layout for tcgen05.cp.cta_group::1.32x128b.warpx4,
// where numKBlocks = K/16.
//
// Within each 32-row x 4-block (128-byte) atom, the 32 rows are decomposed into
// four 8-row sub-partitions (sub = (row % 32) / 8 in 0..3, lane8 = row % 8 in
// 0..7) and transposed into 16-byte lines (line = lane8, byte = 4*sub + (kb % 4)).
// Partial tail tiles at the row or K-block boundary fall back to linear addressing
// within the tail so every (row, kb) in [0, M) x [0, numKBlocks) maps bijectively
// onto [0, M*numKBlocks).
func SwizzleTCGen05ScaleOffset(row, kb, numKBlocks int) int {
	if row < 0 || kb < 0 || numKBlocks <= 0 {
		return 0
	}
	if numKBlocks%4 != 0 {
		return row*numKBlocks + kb
	}
	rTile := row / 32
	rRem := row % 32
	kTile := kb / 4
	kRem := kb % 4
	sub := rRem / 8
	lane8 := rRem % 8
	atomIdx := rTile*(numKBlocks/4) + kTile
	return atomIdx*128 + lane8*16 + sub*4 + kRem
}

// SwizzleTCGen05Scales converts a row-major [M, numKBlocks] scale buffer into
// the [ScaleLayoutTCGen05Swizzled] layout. When M is not a multiple of 32 or
// numKBlocks is not a multiple of 4, full 32x4 atoms are swizzled and tail rows
// are preserved linearly.
func SwizzleTCGen05Scales(scales []uint8, m, numKBlocks int) ([]uint8, error) {
	if m < 0 || numKBlocks <= 0 || len(scales) != m*numKBlocks {
		return nil, fmt.Errorf("swizzle tcgen05 scales: bad shape [%d, %d] for %d bytes", m, numKBlocks, len(scales))
	}
	out := make([]uint8, len(scales))
	fullM := (m / 32) * 32
	for r := range m {
		for kb := range numKBlocks {
			var dst int
			if r < fullM && numKBlocks%4 == 0 {
				dst = SwizzleTCGen05ScaleOffset(r, kb, numKBlocks)
			} else {
				dst = r*numKBlocks + kb
			}
			out[dst] = scales[r*numKBlocks+kb]
		}
	}
	return out, nil
}

// UnswizzleTCGen05Scales is the exact inverse of [SwizzleTCGen05Scales]: it
// recovers the row-major [M, numKBlocks] UE4M3 scale matrix from a swizzled
// scale buffer.
func UnswizzleTCGen05Scales(swizzled []uint8, m, numKBlocks int) ([]uint8, error) {
	if m < 0 || numKBlocks <= 0 || len(swizzled) != m*numKBlocks {
		return nil, fmt.Errorf("unswizzle tcgen05 scales: bad shape [%d, %d] for %d bytes", m, numKBlocks, len(swizzled))
	}
	out := make([]uint8, len(swizzled))
	fullM := (m / 32) * 32
	for r := range m {
		for kb := range numKBlocks {
			var src int
			if r < fullM && numKBlocks%4 == 0 {
				src = SwizzleTCGen05ScaleOffset(r, kb, numKBlocks)
			} else {
				src = r*numKBlocks + kb
			}
			out[r*numKBlocks+kb] = swizzled[src]
		}
	}
	return out, nil
}

// EncodeE2M1 quantizes f to a 4-bit float4_e2m1fn code: bits [2:0] select the
// nearest magnitude in [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] (ties to lower
// index, matching the device threshold cascade) and bit 3 holds the sign.
func EncodeE2M1(f float32) uint8 {
	var sign uint8
	if math.Signbit(float64(f)) && f != 0 {
		sign = 8
		f = -f
	} else if f < 0 {
		sign = 8
		f = -f
	}
	var mag uint8
	switch {
	case f > 5.0:
		mag = 7
	case f > 3.5:
		mag = 6
	case f > 2.5:
		mag = 5
	case f > 1.75:
		mag = 4
	case f > 1.25:
		mag = 3
	case f > 0.75:
		mag = 2
	case f > 0.25:
		mag = 1
	default:
		mag = 0
	}
	return sign | mag
}

// EncodeUE4M3 encodes a non-negative block scale s = amax / 6.0 into a positive
// unsigned-E4M3 (float8_e4m3fn with sign bit 0) byte, clamping zero to the
// smallest positive subnormal 0x01 (2^-9 = 1/512) so downstream division by the
// decoded scale never divides by zero.
func EncodeUE4M3(s float32) uint8 {
	if !(s > 0) {
		return 1
	}
	b := E4M3Bits(s) & 0x7f
	if b == 0 {
		return 1
	}
	if b == 0x7f {
		return 0x7e
	}
	return b
}

const dynQuantDeviceHelpers = `#ifndef EG_DYNQUANT_HELPERS
#define EG_DYNQUANT_HELPERS

// eg_f32_to_ue4m3 encodes a non-negative scale s into a positive E4M3 byte,
// clamping underflow/zero to 0x01 (2^-9) so s_rep = eg_e4m3_to_f32(scale_u8)
// is always strictly positive and finite.
EG_DEV uint8_t eg_f32_to_ue4m3(float s) {
    if (!(s > 0.0f)) return 1u;
    uint8_t b = (uint8_t)(eg_f32x2_to_e4m3x2(s, 0.0f) & 0x7fu);
    if (b == 0u) return 1u;
    if (b == 0x7fu) return 0x7eu;
    return b;
}

// eg_f32_to_e2m1_code quantizes x to a 4-bit E2M1 nibble in [0..15]:
// bits [2:0] index [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] and bit 3 is sign.
EG_DEV uint8_t eg_f32_to_e2m1_code(float x) {
    uint8_t sign = 0u;
    if (x < 0.0f) {
        sign = 8u;
        x = -x;
    }
    uint8_t mag = 0u;
    if (x > 5.0f) mag = 7u;
    else if (x > 3.5f) mag = 6u;
    else if (x > 2.5f) mag = 5u;
    else if (x > 1.75f) mag = 4u;
    else if (x > 1.25f) mag = 3u;
    else if (x > 0.75f) mag = 2u;
    else if (x > 0.25f) mag = 1u;
    return sign | mag;
}

// eg_tcgen05_swizzle_scale_offset computes the byte offset of scale block
// (row, kb) in the 32x128b swizzled layout for tcgen05.cp.cta_group::1.32x128b.warpx4.
EG_DEV size_t eg_tcgen05_swizzle_scale_offset(int row, int kb, int m_rows, int num_k_blocks) {
    const int full_m = (m_rows / 32) * 32;
    if (row >= full_m || (num_k_blocks & 3) != 0) {
        return (size_t)row * num_k_blocks + kb;
    }
    const int r_tile = row >> 5;
    const int r_rem = row & 31;
    const int k_tile = kb >> 2;
    const int k_rem = kb & 3;
    const int sub = r_rem >> 3;
    const int lane8 = r_rem & 7;
    const int atom_idx = r_tile * (num_k_blocks >> 2) + k_tile;
    return (size_t)atom_idx * 128 + lane8 * 16 + sub * 4 + k_rem;
}

#endif // EG_DYNQUANT_HELPERS
`

// RMSNormDynamicNVFP4Quant configures a fused RMSNorm (with optional residual
// addition) and dynamic NVFP4 activation quantizer over rows of Cols values.
//
// For each row m in [0, M):
//  1. When Residual is true, computes h[m, j] = x[m, j] + residual[m, j] and
//     writes h_out[m, j] (rounded to In); otherwise h[m, j] = x[m, j].
//  2. Computes rms = sqrt(mean_j(h[m, j]^2) + Eps) and normalized values
//     y[m, j] = (h[m, j] / rms) * weight[j].
//  3. For each 16-element block b in [0, Cols/16):
//     - Finds amax_b = max_{0 <= t < 16} |y[m, 16*b + t]|,
//     - Encodes scale_u8 = eg_f32_to_ue4m3(amax_b / 6.0f) and decodes the exact
//     representable FP32 scale s_rep = eg_e4m3_to_f32(scale_u8),
//     - Quantizes each of the 16 elements q_t = eg_f32_to_e2m1_code(y[m, 16*b+t] / s_rep)
//     and packs 8 bytes of e2m1x2 (q_{2i} | (q_{2i+1} << 4)),
//     - Stores scale_u8 into out_scales using ScaleLayoutRowMajor or
//     ScaleLayoutTCGen05Swizzled.
type RMSNormDynamicNVFP4Quant struct {
	Name        string
	Cols        int         // K: row width, a positive multiple of 16
	Eps         float32     // RMSNorm epsilon; 0 defaults to 1e-5
	In          DType       // F32 or BF16; 0 defaults to BF16
	Residual    bool        // fuse h = x + residual and write h_out
	ScaleLayout ScaleLayout // ScaleLayoutRowMajor or ScaleLayoutTCGen05Swizzled
}

func (k RMSNormDynamicNVFP4Quant) norm() (RMSNormDynamicNVFP4Quant, error) {
	if k.Name == "" {
		k.Name = "rmsnorm_dyn_nvfp4_quant"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("rmsnorm dynamic nvfp4 quant: bad name %q", k.Name)
	}
	if k.Cols < 16 || k.Cols > 65536 || k.Cols%16 != 0 {
		return k, fmt.Errorf("rmsnorm dynamic nvfp4 quant %s: Cols %d must be a multiple of 16 in 16..65536", k.Name, k.Cols)
	}
	if k.Eps <= 0 {
		k.Eps = 1e-5
	}
	if k.In == 0 {
		k.In = BF16
	}
	if k.In != F32 && k.In != BF16 {
		return k, fmt.Errorf("rmsnorm dynamic nvfp4 quant %s: In must be F32 or BF16, got %v", k.Name, k.In)
	}
	if !k.ScaleLayout.valid() {
		return k, fmt.Errorf("rmsnorm dynamic nvfp4 quant %s: invalid ScaleLayout %v", k.Name, k.ScaleLayout)
	}
	return k, nil
}

// Grid returns the launch grid for m rows (one CTA per row).
func (k RMSNormDynamicNVFP4Quant) Grid(m int) [3]int {
	if m < 0 {
		m = 0
	}
	return [3]int{m, 1, 1}
}

// Block returns the CTA thread block dimensions (256 threads = 8 warps).
func (k RMSNormDynamicNVFP4Quant) Block() [3]int {
	return [3]int{256, 1, 1}
}

// SharedBytes returns the dynamic shared memory bytes per CTA (Cols * 4 bytes
// to stage the row in FP32 across the two reduction/quantization passes).
func (k RMSNormDynamicNVFP4Quant) SharedBytes() int {
	k, err := k.norm()
	if err != nil {
		return 0
	}
	return k.Cols * 4
}

func (k RMSNormDynamicNVFP4Quant) params() []string {
	inType := k.In.cType()
	p := []string{
		"int rows",
		"const " + inType + "* __restrict__ x",
	}
	if k.Residual {
		p = append(p, "const "+inType+"* __restrict__ residual")
	}
	p = append(p, "const "+inType+"* __restrict__ weight")
	if k.Residual {
		p = append(p, inType+"* __restrict__ h_out")
	}
	p = append(p,
		"uint8_t* __restrict__ out_fp4",
		"uint8_t* __restrict__ out_scales",
	)
	return p
}

// CUDA returns the device kernel source (without [Prelude]).
func (k RMSNormDynamicNVFP4Quant) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	loadElem := func(expr string) string {
		if k.In == BF16 {
			return "eg_bf16_to_f32(" + expr + ")"
		}
		return expr
	}
	storeH := func(dst, val string) string {
		if k.In == BF16 {
			return dst + " = eg_f32_to_bf16(" + val + ")"
		}
		return dst + " = " + val
	}

	var b strings.Builder
	b.WriteString(dynQuantDeviceHelpers)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused RMSNorm + dynamic NVFP4 activation quantizer (Cols=%d, In=%v, Residual=%v, ScaleLayout=%s).\n",
		k.Name, k.Cols, k.In, k.Residual, k.ScaleLayout)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n",
		k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int K = %d, K_BYTES = K / 2, K_BLOCKS = K / 16;\n", k.Cols)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))
	b.WriteString(`    const int r = blockIdx.x, tid = threadIdx.x;
    if (r >= rows) return;
    __shared__ float eg_red[8];
    EG_DYNAMIC_SMEM(float, s_h);

    const size_t row_off = (size_t)r * K;
    float sum_sq = 0.0f;
    for (int c = tid; c < K; c += 256) {
`)
	fmt.Fprintf(&b, "        float h = %s;\n", loadElem("x[row_off + c]"))
	if k.Residual {
		fmt.Fprintf(&b, "        h += %s;\n", loadElem("residual[row_off + c]"))
		fmt.Fprintf(&b, "        %s;\n", storeH("h_out[row_off + c]", "h"))
		if k.In == BF16 {
			b.WriteString("        h = eg_bf16_to_f32(eg_f32_to_bf16(h));\n")
		}
	}
	b.WriteString(`        s_h[c] = h;
        sum_sq += h * h;
    }
    sum_sq = eg_warp_sum(sum_sq);
    sum_sq = eg_block_combine<0>(sum_sq, eg_red);
    const float inv_rms = rsqrtf(sum_sq * (1.0f / (float)K) + EPS);

    // Normalize in place in shared memory: y[c] = (h[c] / rms) * weight[c].
    for (int c = tid; c < K; c += 256) {
`)
	fmt.Fprintf(&b, "        const float w = %s;\n", loadElem("weight[c]"))
	b.WriteString(`        s_h[c] = (s_h[c] * inv_rms) * w;
    }
    __syncthreads();

    // Dynamic per-16-element NVFP4 quantization: each thread quantizes blocks b = tid, tid+256, ...
    for (int kb = tid; kb < K_BLOCKS; kb += 256) {
        const int base = kb * 16;
        float amax = 0.0f;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            amax = fmaxf(amax, fabsf(s_h[base + t]));
        }
        const uint8_t scale_u8 = eg_f32_to_ue4m3(amax * (1.0f / 6.0f));
        float s_rep = eg_e4m3_to_f32(scale_u8);
        if (!(s_rep > 0.0f)) s_rep = 1.0f / 512.0f;
        const float inv_s = 1.0f / s_rep;

        uint8_t* const dst_fp4 = out_fp4 + (size_t)r * K_BYTES + kb * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t q0 = eg_f32_to_e2m1_code(s_h[base + 2 * i] * inv_s);
            const uint8_t q1 = eg_f32_to_e2m1_code(s_h[base + 2 * i + 1] * inv_s);
            dst_fp4[i] = (uint8_t)(q0 | (q1 << 4));
        }
`)
	if k.ScaleLayout == ScaleLayoutTCGen05Swizzled {
		b.WriteString("        out_scales[eg_tcgen05_swizzle_scale_offset(r, kb, rows, K_BLOCKS)] = scale_u8;\n")
	} else {
		b.WriteString("        out_scales[(size_t)r * K_BLOCKS + kb] = scale_u8;\n")
	}
	b.WriteString("    }\n}\n")
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(int rows, <params...>, cudaStream_t stream)
func (k RMSNormDynamicNVFP4Quant) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := paramNames(params)
	smem := k.SharedBytes()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    constexpr int smem = %d;\n", smem)
	fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", k.Name)
	b.WriteString("    const dim3 grid(rows, 1, 1), block(256, 1, 1);\n")
	fmt.Fprintf(&b, "    if (rows > 0) EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", k.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude], the kernel, and its launcher.
func (k RMSNormDynamicNVFP4Quant) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// Reference executes the CPU reference for [RMSNormDynamicNVFP4Quant] over M
// rows of Cols values. It returns the updated residual sum hOut (when Residual
// is true, or the rounded input x otherwise), the packed E2M1 bytes [M, Cols/2],
// the UE4M3 block scales [M, Cols/16] (in k.ScaleLayout), and the dequantized
// FP32 activations [M, Cols].
func (k RMSNormDynamicNVFP4Quant) Reference(x, residual, weight []float32, m int) (hOut []float32, packedFP4 []uint8, scales []uint8, dequant []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if m < 1 {
		return nil, nil, nil, nil, fmt.Errorf("rmsnorm dynamic nvfp4 quant %s reference: M %d < 1", k.Name, m)
	}
	if len(x) != m*k.Cols {
		return nil, nil, nil, nil, fmt.Errorf("rmsnorm dynamic nvfp4 quant %s reference: len(x) = %d, want %d", k.Name, len(x), m*k.Cols)
	}
	if k.Residual && len(residual) != m*k.Cols {
		return nil, nil, nil, nil, fmt.Errorf("rmsnorm dynamic nvfp4 quant %s reference: len(residual) = %d, want %d", k.Name, len(residual), m*k.Cols)
	}
	if len(weight) != k.Cols {
		return nil, nil, nil, nil, fmt.Errorf("rmsnorm dynamic nvfp4 quant %s reference: len(weight) = %d, want %d", k.Name, len(weight), k.Cols)
	}

	K := k.Cols
	kBytes := K / 2
	kBlocks := K / 16
	hOut = make([]float32, m*K)
	packedFP4 = make([]uint8, m*kBytes)
	rowScales := make([]uint8, m*kBlocks)
	dequant = make([]float32, m*K)

	wRound := make([]float32, K)
	for c := range K {
		wRound[c] = k.In.Round(weight[c])
	}

	for r := range m {
		rowOff := r * K
		// Match the 256-thread / 8-warp reduction order in the kernel:
		// lane l in warp w sums c = (32*w + l) + 256*step, then warp shuffle
		// tree (m = 16, 8, 4, 2, 1), then warps 0..7 in order.
		var warpSums [8]float32
		for w := range 8 {
			var laneSums [32]float32
			for l := range 32 {
				tid := w*32 + l
				var sq float32
				for c := tid; c < K; c += 256 {
					xv := k.In.Round(x[rowOff+c])
					hv := xv
					if k.Residual {
						rv := k.In.Round(residual[rowOff+c])
						hv = k.In.Round(xv + rv)
					}
					hOut[rowOff+c] = hv
					sq += hv * hv
				}
				laneSums[l] = sq
			}
			for mask := 16; mask > 0; mask >>= 1 {
				var next [32]float32
				for l := range 32 {
					next[l] = laneSums[l] + laneSums[l^mask]
				}
				laneSums = next
			}
			warpSums[w] = laneSums[0]
		}
		sumSq := warpSums[0]
		for w := 1; w < 8; w++ {
			sumSq += warpSums[w]
		}
		invRMS := float32(1.0 / math.Sqrt(float64(sumSq*(1.0/float32(K))+k.Eps)))

		for kb := range kBlocks {
			base := rowOff + kb*16
			var y [16]float32
			var amax float32
			for t := range 16 {
				yv := (hOut[base+t] * invRMS) * wRound[kb*16+t]
				y[t] = yv
				if a := float32(math.Abs(float64(yv))); a > amax {
					amax = a
				}
			}
			sb := EncodeUE4M3(amax * (1.0 / 6.0))
			rowScales[r*kBlocks+kb] = sb
			sRep := FromE4M3Bits(sb)
			if !(sRep > 0) {
				sRep = 1.0 / 512.0
			}
			invS := 1.0 / sRep
			for i := range 8 {
				q0 := EncodeE2M1(y[2*i] * invS)
				q1 := EncodeE2M1(y[2*i+1] * invS)
				packedFP4[r*kBytes+kb*8+i] = q0 | (q1 << 4)
				dequant[base+2*i] = FromE2M1(q0) * sRep
				dequant[base+2*i+1] = FromE2M1(q1) * sRep
			}
		}
	}

	if k.ScaleLayout == ScaleLayoutTCGen05Swizzled {
		scales, err = SwizzleTCGen05Scales(rowScales, m, kBlocks)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	} else {
		scales = rowScales
	}
	return hOut, packedFP4, scales, dequant, nil
}

// SwiGLUDynamicNVFP4Quant configures a fused SwiGLU (y = silu(gate) * up) and
// dynamic NVFP4 activation quantizer over rows of Cols output values.
//
// For each row m in [0, M) and each 16-element block b in [0, Cols/16):
//  1. Computes y[m, 16*b + t] = silu(gate[m, 16*b + t]) * up[m, 16*b + t] for t in 0..15.
//  2. Finds amax_b = max_{0 <= t < 16} |y[m, 16*b + t]|.
//  3. Encodes scale_u8 = eg_f32_to_ue4m3(amax_b / 6.0f) and decodes
//     s_rep = eg_e4m3_to_f32(scale_u8).
//  4. Quantizes each element to E2M1 and writes 8 packed e2m1x2 bytes to
//     out_fp4 and scale_u8 to out_scales (in ScaleLayoutRowMajor or
//     ScaleLayoutTCGen05Swizzled).
type SwiGLUDynamicNVFP4Quant struct {
	Name        string
	Cols        int         // K: output width after gating, a positive multiple of 16
	In          DType       // F32 or BF16; 0 defaults to BF16
	ScaleLayout ScaleLayout // ScaleLayoutRowMajor or ScaleLayoutTCGen05Swizzled
}

func (k SwiGLUDynamicNVFP4Quant) norm() (SwiGLUDynamicNVFP4Quant, error) {
	if k.Name == "" {
		k.Name = "swiglu_dyn_nvfp4_quant"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("swiglu dynamic nvfp4 quant: bad name %q", k.Name)
	}
	if k.Cols < 16 || k.Cols > 65536 || k.Cols%16 != 0 {
		return k, fmt.Errorf("swiglu dynamic nvfp4 quant %s: Cols %d must be a multiple of 16 in 16..65536", k.Name, k.Cols)
	}
	if k.In == 0 {
		k.In = BF16
	}
	if k.In != F32 && k.In != BF16 {
		return k, fmt.Errorf("swiglu dynamic nvfp4 quant %s: In must be F32 or BF16, got %v", k.Name, k.In)
	}
	if !k.ScaleLayout.valid() {
		return k, fmt.Errorf("swiglu dynamic nvfp4 quant %s: invalid ScaleLayout %v", k.Name, k.ScaleLayout)
	}
	return k, nil
}

// Grid returns the launch grid for m rows (one CTA per row).
func (k SwiGLUDynamicNVFP4Quant) Grid(m int) [3]int {
	if m < 0 {
		m = 0
	}
	return [3]int{m, 1, 1}
}

// Block returns the CTA thread block dimensions (256 threads).
func (k SwiGLUDynamicNVFP4Quant) Block() [3]int {
	return [3]int{256, 1, 1}
}

// SharedBytes returns the dynamic shared memory bytes per CTA (0, since each
// 16-element block is gated and quantized in registers by a single thread).
func (k SwiGLUDynamicNVFP4Quant) SharedBytes() int {
	return 0
}

func (k SwiGLUDynamicNVFP4Quant) params() []string {
	inType := k.In.cType()
	return []string{
		"int rows",
		"const " + inType + "* __restrict__ gate",
		"const " + inType + "* __restrict__ up",
		"uint8_t* __restrict__ out_fp4",
		"uint8_t* __restrict__ out_scales",
	}
}

// CUDA returns the device kernel source (without [Prelude]).
func (k SwiGLUDynamicNVFP4Quant) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	loadElem := func(expr string) string {
		if k.In == BF16 {
			return "eg_bf16_to_f32(" + expr + ")"
		}
		return expr
	}

	var b strings.Builder
	b.WriteString(dynQuantDeviceHelpers)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: fused SwiGLU + dynamic NVFP4 activation quantizer (Cols=%d, In=%v, ScaleLayout=%s).\n",
		k.Name, k.Cols, k.In, k.ScaleLayout)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n",
		k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int K = %d, K_BYTES = K / 2, K_BLOCKS = K / 16;\n", k.Cols)
	b.WriteString(`    const int r = blockIdx.x, tid = threadIdx.x;
    if (r >= rows) return;
    const size_t row_off = (size_t)r * K;

    for (int kb = tid; kb < K_BLOCKS; kb += 256) {
        const size_t base = row_off + (size_t)kb * 16;
        float y[16];
        float amax = 0.0f;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
`)
	fmt.Fprintf(&b, "            const float g = %s;\n", loadElem("gate[base + t]"))
	fmt.Fprintf(&b, "            const float u = %s;\n", loadElem("up[base + t]"))
	b.WriteString(`            const float v = eg_silu(g) * u;
            y[t] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        const uint8_t scale_u8 = eg_f32_to_ue4m3(amax * (1.0f / 6.0f));
        float s_rep = eg_e4m3_to_f32(scale_u8);
        if (!(s_rep > 0.0f)) s_rep = 1.0f / 512.0f;
        const float inv_s = 1.0f / s_rep;

        uint8_t* const dst_fp4 = out_fp4 + (size_t)r * K_BYTES + kb * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t q0 = eg_f32_to_e2m1_code(y[2 * i] * inv_s);
            const uint8_t q1 = eg_f32_to_e2m1_code(y[2 * i + 1] * inv_s);
            dst_fp4[i] = (uint8_t)(q0 | (q1 << 4));
        }
`)
	if k.ScaleLayout == ScaleLayoutTCGen05Swizzled {
		b.WriteString("        out_scales[eg_tcgen05_swizzle_scale_offset(r, kb, rows, K_BLOCKS)] = scale_u8;\n")
	} else {
		b.WriteString("        out_scales[(size_t)r * K_BLOCKS + kb] = scale_u8;\n")
	}
	b.WriteString("    }\n}\n")
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(int rows, <params...>, cudaStream_t stream)
func (k SwiGLUDynamicNVFP4Quant) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := paramNames(params)
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	b.WriteString("    const dim3 grid(rows, 1, 1), block(256, 1, 1);\n")
	fmt.Fprintf(&b, "    if (rows > 0) EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", k.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude], the kernel, and its launcher.
func (k SwiGLUDynamicNVFP4Quant) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// Reference executes the CPU reference for [SwiGLUDynamicNVFP4Quant] over M
// rows of Cols values. It returns the packed E2M1 bytes [M, Cols/2], the UE4M3
// block scales [M, Cols/16] (in k.ScaleLayout), and the dequantized FP32
// activations [M, Cols].
func (k SwiGLUDynamicNVFP4Quant) Reference(gate, up []float32, m int) (packedFP4 []uint8, scales []uint8, dequant []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	if m < 1 {
		return nil, nil, nil, fmt.Errorf("swiglu dynamic nvfp4 quant %s reference: M %d < 1", k.Name, m)
	}
	if len(gate) != m*k.Cols || len(up) != m*k.Cols {
		return nil, nil, nil, fmt.Errorf("swiglu dynamic nvfp4 quant %s reference: len(gate)=%d len(up)=%d, want %d",
			k.Name, len(gate), len(up), m*k.Cols)
	}

	K := k.Cols
	kBytes := K / 2
	kBlocks := K / 16
	packedFP4 = make([]uint8, m*kBytes)
	rowScales := make([]uint8, m*kBlocks)
	dequant = make([]float32, m*K)

	for r := range m {
		rowOff := r * K
		for kb := range kBlocks {
			base := rowOff + kb*16
			var y [16]float32
			var amax float32
			for t := range 16 {
				g := k.In.Round(gate[base+t])
				u := k.In.Round(up[base+t])
				silu := g / (1.0 + float32(math.Exp(float64(-g))))
				yv := silu * u
				y[t] = yv
				if a := float32(math.Abs(float64(yv))); a > amax {
					amax = a
				}
			}
			sb := EncodeUE4M3(amax * (1.0 / 6.0))
			rowScales[r*kBlocks+kb] = sb
			sRep := FromE4M3Bits(sb)
			if !(sRep > 0) {
				sRep = 1.0 / 512.0
			}
			invS := 1.0 / sRep
			for i := range 8 {
				q0 := EncodeE2M1(y[2*i] * invS)
				q1 := EncodeE2M1(y[2*i+1] * invS)
				packedFP4[r*kBytes+kb*8+i] = q0 | (q1 << 4)
				dequant[base+2*i] = FromE2M1(q0) * sRep
				dequant[base+2*i+1] = FromE2M1(q1) * sRep
			}
		}
	}

	if k.ScaleLayout == ScaleLayoutTCGen05Swizzled {
		scales, err = SwizzleTCGen05Scales(rowScales, m, kBlocks)
		if err != nil {
			return nil, nil, nil, err
		}
	} else {
		scales = rowScales
	}
	return packedFP4, scales, dequant, nil
}
