package hopper

import (
	"fmt"
	"math"
	"strings"
)

// QKNormRoPEKVAppend fuses 5 post-QKV operations into a single kernel launch
// between a fused QKV projection [M, (NumHeads + 2*KVHeads)*HeadDim] and
// [PagedAttention] or [MTPPagedAttention]:
//  1. Unpacks packed qkv[m, :] into Q [NumHeads, HeadDim], K [KVHeads, HeadDim],
//     and V [KVHeads, HeadDim];
//  2. Applies optional per-head RMSNorm (QKNorm: true) to each Q head (using
//     q_norm_weight[HeadDim] and Eps) and each K head (using
//     k_norm_weight[HeadDim] and Eps);
//  3. Applies Rotary Position Embedding (RoPE) to the first RotaryDim (default
//     HeadDim) elements of each Q head and each K head using
//     cos_sin_table[pos * RotaryDim + ...], where pos = positions[m],
//     supporting both NeoX half-rotated pairs (i, i + RotaryDim/2) and
//     interleaved pairs (2*i, 2*i + 1) via InterleavedRoPE;
//  4. Writes rotated Q to q_out[m, h, d] (in BF16 or F32);
//  5. Writes rotated K and unpacked V directly into the paged KV cache
//     (k_cache, v_cache) at page_id = page_table[seq_id * MaxPages + token_pos / PageSize]
//     and page_offset = token_pos % PageSize (where seq_id = seq_ids[m] and
//     token_pos = kv_positions[m] if kv_positions != nullptr else seqlens[seq_id] - 1),
//     with optional FP8 (E4M3) quantization (KVFP8: true, dividing by k_scale /
//     v_scale and converting via eg_f32_to_e4m3).
type QKNormRoPEKVAppend struct {
	Name            string
	NumHeads        int     // H_q
	KVHeads         int     // H_kv
	HeadDim         int     // D (e.g., 64, 128)
	RotaryDim       int     // D_rope <= D (0 defaults to HeadDim)
	PageSize        int     // tokens per KV page
	MaxPages        int     // stride of page_table per sequence
	Eps             float32 // RMSNorm epsilon when QKNorm is true
	QKNorm          bool    // apply per-head RMSNorm to Q and K before RoPE
	InterleavedRoPE bool    // true: (2i, 2i+1) pairs; false: NeoX (i, i + RotaryDim/2) pairs
	KVFP8           bool    // quantize K and V to FP8 E4M3 (with inv_k_scale, inv_v_scale) when writing to paged cache
	DType           DType   // BF16 or F32 for qkv input and q_out
}

func (k QKNormRoPEKVAppend) norm() (QKNormRoPEKVAppend, error) {
	if k.Name == "" {
		k.Name = "qk_norm_rope_kv_append"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("qk norm rope kv append: bad name %q", k.Name)
	}
	if k.NumHeads < 1 || k.KVHeads < 1 || k.NumHeads%k.KVHeads != 0 {
		return k, fmt.Errorf("qk norm rope kv append %s: bad heads %d/%d", k.Name, k.NumHeads, k.KVHeads)
	}
	if k.HeadDim < 2 || k.HeadDim > 1024 || k.HeadDim%2 != 0 {
		return k, fmt.Errorf("qk norm rope kv append %s: bad head dim %d", k.Name, k.HeadDim)
	}
	if k.RotaryDim == 0 {
		k.RotaryDim = k.HeadDim
	}
	if k.RotaryDim < 2 || k.RotaryDim > k.HeadDim || k.RotaryDim%2 != 0 {
		return k, fmt.Errorf("qk norm rope kv append %s: bad rotary dim %d for head dim %d", k.Name, k.RotaryDim, k.HeadDim)
	}
	if k.PageSize < 1 {
		return k, fmt.Errorf("qk norm rope kv append %s: bad page size %d", k.Name, k.PageSize)
	}
	if k.MaxPages < 1 {
		return k, fmt.Errorf("qk norm rope kv append %s: bad max pages %d", k.Name, k.MaxPages)
	}
	if k.DType == 0 {
		k.DType = BF16
	}
	if k.DType != BF16 && k.DType != F32 {
		return k, fmt.Errorf("qk norm rope kv append %s: dtype must be BF16 or F32, got %v", k.Name, k.DType)
	}
	if k.Eps <= 0 {
		k.Eps = 1e-5
	}
	return k, nil
}

// Grid returns the launch grid and block dimensions for m tokens. Each
// token row has a CTA of 128 threads (4 warps) per 4 heads, so a single
// token's heads still spread across SMs. The kernel is correct for any
// gridDim.y: each CTA strides over the heads by 4·gridDim.y.
func (k QKNormRoPEKVAppend) Grid(m int) (grid, block [3]int) {
	return [3]int{m, (max(k.NumHeads, k.KVHeads) + 3) / 4, 1}, [3]int{128, 1, 1}
}

func (k QKNormRoPEKVAppend) params() []string {
	qType := k.DType.cType()
	kvType := qType
	if k.KVFP8 {
		kvType = "uint8_t"
	}
	p := []string{
		"const " + qType + "* __restrict__ qkv",
	}
	if k.QKNorm {
		p = append(p,
			"const "+qType+"* __restrict__ q_norm_weight",
			"const "+qType+"* __restrict__ k_norm_weight",
		)
	}
	p = append(p,
		"const float* __restrict__ cos_sin_table",
		"const int32_t* __restrict__ positions",
		"const int32_t* __restrict__ seq_ids",
		"const int32_t* __restrict__ seqlens",
		"const int32_t* __restrict__ kv_positions",
		"const int32_t* __restrict__ page_table",
	)
	if k.KVFP8 {
		p = append(p,
			"const float* __restrict__ k_scale",
			"const float* __restrict__ v_scale",
		)
	}
	p = append(p,
		qType+"* __restrict__ q_out",
		kvType+"* __restrict__ k_cache",
		kvType+"* __restrict__ v_cache",
	)
	return p
}

// CUDA returns the device kernel source (without [Prelude]).
func (k QKNormRoPEKVAppend) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	qType := k.DType.cType()
	loadVal := func(expr string) string {
		if k.DType == BF16 {
			return "eg_bf16_to_f32(" + expr + ")"
		}
		return expr
	}
	storeQ := func(dst, val string) string {
		if k.DType == BF16 {
			return dst + " = eg_f32_to_bf16(" + val + ")"
		}
		return dst + " = " + val
	}
	storeK := func(dst, val string) string {
		if k.KVFP8 {
			return dst + " = eg_f32_to_e4m3((" + val + ") / ks)"
		}
		if k.DType == BF16 {
			return dst + " = eg_f32_to_bf16(" + val + ")"
		}
		return dst + " = " + val
	}
	storeV := func(dst, val string) string {
		if k.KVFP8 {
			return dst + " = eg_f32_to_e4m3((" + val + ") / vs)"
		}
		if k.DType == BF16 {
			return dst + " = eg_f32_to_bf16(" + val + ")"
		}
		return dst + " = " + val
	}

	var b strings.Builder
	b.WriteString(`#ifndef EG_F32_TO_E4M3_HELPER
#define EG_F32_TO_E4M3_HELPER
EG_DEV uint8_t eg_f32_to_e4m3(float v) {
    return (uint8_t)(eg_f32x2_to_e4m3x2(v, 0.0f) & 0xffu);
}
#endif
`)
	fmt.Fprintf(&b, "// %s: fused QK RMSNorm, RoPE (%d/%d), and paged KV cache append (%d/%d heads, page %d).\n",
		k.Name, k.RotaryDim, k.HeadDim, k.NumHeads, k.KVHeads, k.PageSize)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, D = %d, DROPE = %d, HALF_ROPE = %d;\n",
		k.NumHeads, k.KVHeads, k.HeadDim, k.RotaryDim, k.RotaryDim/2)
	fmt.Fprintf(&b, "    constexpr int PAGE = %d, MAX_PAGES = %d, QKV_STRIDE = (HQ + 2 * HKV) * D;\n",
		k.PageSize, k.MaxPages)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))

	b.WriteString(`    const int m = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int pos = positions ? positions[m] : 0;
    const int seq_id = seq_ids ? seq_ids[m] : m;
    const int token_pos = kv_positions ? kv_positions[m] : (seqlens[seq_id] - 1);
    const float* const cs_row = cos_sin_table + (size_t)pos * DROPE;
`)
	fmt.Fprintf(&b, "    const %s* const qkv_row = qkv + (size_t)m * QKV_STRIDE;\n", qType)
	fmt.Fprintf(&b, "    const %s* const q_in = qkv_row;\n", qType)
	fmt.Fprintf(&b, "    const %s* const k_in = qkv_row + HQ * D;\n", qType)
	fmt.Fprintf(&b, "    const %s* const v_in = qkv_row + (HQ + HKV) * D;\n", qType)
	if k.KVFP8 {
		b.WriteString(`    const float ks = (k_scale && k_scale[0] != 0.0f) ? k_scale[0] : 1.0f;
    const float vs = (v_scale && v_scale[0] != 0.0f) ? v_scale[0] : 1.0f;
`)
	}
	b.WriteString(`    alignas(16) __shared__ float s_head[4][D];

    // 1. Process Q heads: optional per-head RMSNorm + RoPE -> q_out[m, h, d].
    for (int h = blockIdx.y * 4 + warp; h < HQ; h += 4 * gridDim.y) {
`)
	fmt.Fprintf(&b, "        const %s* const src = q_in + h * D;\n", qType)
	if k.QKNorm {
		b.WriteString("        float sq = 0.0f;\n")
		b.WriteString("        for (int d = lane; d < D; d += 32) {\n")
		fmt.Fprintf(&b, "            const float v = %s;\n", loadVal("src[d]"))
		b.WriteString("            s_head[warp][d] = v;\n")
		b.WriteString("            sq += v * v;\n")
		b.WriteString("        }\n")
		b.WriteString("        const float scale = rsqrtf(eg_warp_sum(sq) * (1.0f / (float)D) + EPS);\n")
		b.WriteString("        __syncwarp();\n")
		b.WriteString("        for (int d = lane; d < D; d += 32) {\n")
		fmt.Fprintf(&b, "            const float w = q_norm_weight ? %s : 1.0f;\n", loadVal("q_norm_weight[d]"))
		b.WriteString("            s_head[warp][d] = s_head[warp][d] * scale * w;\n")
		b.WriteString("        }\n")
		b.WriteString("        __syncwarp();\n")
	} else {
		b.WriteString("        for (int d = lane; d < D; d += 32) {\n")
		fmt.Fprintf(&b, "            s_head[warp][d] = %s;\n", loadVal("src[d]"))
		b.WriteString("        }\n")
		b.WriteString("        __syncwarp();\n")
	}

	fmt.Fprintf(&b, "        %s* const dst = q_out + ((size_t)m * HQ + h) * D;\n", qType)
	if k.InterleavedRoPE {
		b.WriteString(`        for (int i = lane; i < HALF_ROPE; i += 32) {
            const float x0 = s_head[warp][2 * i];
            const float x1 = s_head[warp][2 * i + 1];
            const float c = cs_row[2 * i];
            const float s = cs_row[2 * i + 1];
            const float y0 = x0 * c - x1 * s;
            const float y1 = x1 * c + x0 * s;
`)
		fmt.Fprintf(&b, "            %s;\n", storeQ("dst[2 * i]", "y0"))
		fmt.Fprintf(&b, "            %s;\n", storeQ("dst[2 * i + 1]", "y1"))
		b.WriteString("        }\n")
	} else {
		b.WriteString(`        for (int i = lane; i < HALF_ROPE; i += 32) {
            const float x0 = s_head[warp][i];
            const float x1 = s_head[warp][i + HALF_ROPE];
            const float c = cs_row[i];
            const float s = cs_row[i + HALF_ROPE];
            const float y0 = x0 * c - x1 * s;
            const float y1 = x1 * c + x0 * s;
`)
		fmt.Fprintf(&b, "            %s;\n", storeQ("dst[i]", "y0"))
		fmt.Fprintf(&b, "            %s;\n", storeQ("dst[i + HALF_ROPE]", "y1"))
		b.WriteString("        }\n")
	}
	b.WriteString("        for (int d = DROPE + lane; d < D; d += 32) {\n")
	fmt.Fprintf(&b, "            %s;\n", storeQ("dst[d]", "s_head[warp][d]"))
	b.WriteString("        }\n")
	b.WriteString("        __syncwarp();\n")
	b.WriteString("    }\n\n")

	// 2. Resolve paged KV cache slot and write rotated K and unpacked V.
	b.WriteString(`    if (token_pos < 0) return;
    const int page_id = page_table[(size_t)seq_id * MAX_PAGES + token_pos / PAGE];
    if (page_id < 0) return;
    const int page_offset = token_pos % PAGE;
    const size_t kv_slot = (size_t)page_id * PAGE + page_offset;

    for (int kh = blockIdx.y * 4 + warp; kh < HKV; kh += 4 * gridDim.y) {
`)
	fmt.Fprintf(&b, "        const %s* const k_src = k_in + kh * D;\n", qType)
	fmt.Fprintf(&b, "        const %s* const v_src = v_in + kh * D;\n", qType)
	if k.QKNorm {
		b.WriteString("        float sq = 0.0f;\n")
		b.WriteString("        for (int d = lane; d < D; d += 32) {\n")
		fmt.Fprintf(&b, "            const float v = %s;\n", loadVal("k_src[d]"))
		b.WriteString("            s_head[warp][d] = v;\n")
		b.WriteString("            sq += v * v;\n")
		b.WriteString("        }\n")
		b.WriteString("        const float scale = rsqrtf(eg_warp_sum(sq) * (1.0f / (float)D) + EPS);\n")
		b.WriteString("        __syncwarp();\n")
		b.WriteString("        for (int d = lane; d < D; d += 32) {\n")
		fmt.Fprintf(&b, "            const float w = k_norm_weight ? %s : 1.0f;\n", loadVal("k_norm_weight[d]"))
		b.WriteString("            s_head[warp][d] = s_head[warp][d] * scale * w;\n")
		b.WriteString("        }\n")
		b.WriteString("        __syncwarp();\n")
	} else {
		b.WriteString("        for (int d = lane; d < D; d += 32) {\n")
		fmt.Fprintf(&b, "            s_head[warp][d] = %s;\n", loadVal("k_src[d]"))
		b.WriteString("        }\n")
		b.WriteString("        __syncwarp();\n")
	}

	kvType := qType
	if k.KVFP8 {
		kvType = "uint8_t"
	}
	fmt.Fprintf(&b, "        %s* const k_dst = k_cache + (kv_slot * HKV + kh) * D;\n", kvType)
	fmt.Fprintf(&b, "        %s* const v_dst = v_cache + (kv_slot * HKV + kh) * D;\n", kvType)
	if k.InterleavedRoPE {
		b.WriteString(`        for (int i = lane; i < HALF_ROPE; i += 32) {
            const float x0 = s_head[warp][2 * i];
            const float x1 = s_head[warp][2 * i + 1];
            const float c = cs_row[2 * i];
            const float s = cs_row[2 * i + 1];
            const float y0 = x0 * c - x1 * s;
            const float y1 = x1 * c + x0 * s;
`)
		fmt.Fprintf(&b, "            %s;\n", storeK("k_dst[2 * i]", "y0"))
		fmt.Fprintf(&b, "            %s;\n", storeK("k_dst[2 * i + 1]", "y1"))
		b.WriteString("        }\n")
	} else {
		b.WriteString(`        for (int i = lane; i < HALF_ROPE; i += 32) {
            const float x0 = s_head[warp][i];
            const float x1 = s_head[warp][i + HALF_ROPE];
            const float c = cs_row[i];
            const float s = cs_row[i + HALF_ROPE];
            const float y0 = x0 * c - x1 * s;
            const float y1 = x1 * c + x0 * s;
`)
		fmt.Fprintf(&b, "            %s;\n", storeK("k_dst[i]", "y0"))
		fmt.Fprintf(&b, "            %s;\n", storeK("k_dst[i + HALF_ROPE]", "y1"))
		b.WriteString("        }\n")
	}
	b.WriteString("        for (int d = DROPE + lane; d < D; d += 32) {\n")
	fmt.Fprintf(&b, "            %s;\n", storeK("k_dst[d]", "s_head[warp][d]"))
	b.WriteString("        }\n")
	b.WriteString("        for (int d = lane; d < D; d += 32) {\n")
	fmt.Fprintf(&b, "            const float vv = %s;\n", loadVal("v_src[d]"))
	fmt.Fprintf(&b, "            %s;\n", storeV("v_dst[d]", "vv"))
	b.WriteString("        }\n")
	b.WriteString("        __syncwarp();\n")
	b.WriteString("    }\n}\n")
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(int m, <kernel params...>, cudaStream_t stream)
func (k QKNormRoPEKVAppend) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := paramNames(params)
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int m, %s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	_, block := k.Grid(1)
	fmt.Fprintf(&b, "    const dim3 grid(m, %d, 1), block(%d, 1, 1);\n", (max(k.NumHeads, k.KVHeads)+3)/4, block[0])
	fmt.Fprintf(&b, "    if (m > 0) EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", k.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (k QKNormRoPEKVAppend) Source() (string, error) {
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

// QKNormRoPEKVInputs holds the inputs and in-place paged KV cache buffers for
// [QKNormRoPEKVAppend.Reference].
type QKNormRoPEKVInputs struct {
	QKV         []float64 // [M, (NumHeads + 2*KVHeads) * HeadDim]
	QNormWeight []float64 // [HeadDim] when QKNorm is true
	KNormWeight []float64 // [HeadDim] when QKNorm is true
	CosSinTable []float32 // [maxPos * RotaryDim]
	Positions   []int32   // [M]: RoPE position per token
	SeqIDs      []int32   // [M]: sequence index per token (nil defaults to 0..M-1)
	SeqLens     []int32   // [batch]: sequence lengths (used when KVPositions is nil)
	KVPositions []int32   // [M]: explicit KV cache token position per token (optional)
	PageTable   []int32   // [batch, MaxPages]
	KScale      float32   // when KVFP8 (0 defaults to 1)
	VScale      float32   // when KVFP8 (0 defaults to 1)
	KCache      []float64 // [totalPages * PageSize, KVHeads, HeadDim], updated in place
	VCache      []float64 // [totalPages * PageSize, KVHeads, HeadDim], updated in place
}

// Reference computes the float64 reference for [QKNormRoPEKVAppend]: it returns
// q_out [M * NumHeads * HeadDim] (rounded to k.DType) and scatters the rotated
// K and unpacked V into in.KCache and in.VCache in place (rounded to E4M3 when
// k.KVFP8 is true, or k.DType otherwise).
func (k QKNormRoPEKVAppend) Reference(m int, in *QKNormRoPEKVInputs) ([]float64, error) {
	k, err := k.norm()
	if err != nil {
		return nil, err
	}
	if in == nil {
		return nil, fmt.Errorf("qk norm rope kv append %s reference: nil inputs", k.Name)
	}
	qkvStride := (k.NumHeads + 2*k.KVHeads) * k.HeadDim
	if len(in.QKV) != m*qkvStride {
		return nil, fmt.Errorf("qk norm rope kv append %s reference: len(qkv) = %d, want %d", k.Name, len(in.QKV), m*qkvStride)
	}
	if k.QKNorm {
		if len(in.QNormWeight) != k.HeadDim || len(in.KNormWeight) != k.HeadDim {
			return nil, fmt.Errorf("qk norm rope kv append %s reference: norm weight lengths (%d, %d), want %d",
				k.Name, len(in.QNormWeight), len(in.KNormWeight), k.HeadDim)
		}
	}
	if len(in.Positions) != m {
		return nil, fmt.Errorf("qk norm rope kv append %s reference: len(positions) = %d, want %d", k.Name, len(in.Positions), m)
	}
	if in.SeqIDs != nil && len(in.SeqIDs) != m {
		return nil, fmt.Errorf("qk norm rope kv append %s reference: len(seq_ids) = %d, want %d", k.Name, len(in.SeqIDs), m)
	}
	if in.KVPositions != nil && len(in.KVPositions) != m {
		return nil, fmt.Errorf("qk norm rope kv append %s reference: len(kv_positions) = %d, want %d", k.Name, len(in.KVPositions), m)
	}
	if in.KVPositions == nil && len(in.SeqLens) == 0 {
		return nil, fmt.Errorf("qk norm rope kv append %s reference: need seqlens or kv_positions", k.Name)
	}

	ks := in.KScale
	if ks == 0 {
		ks = 1
	}
	vs := in.VScale
	if vs == 0 {
		vs = 1
	}

	roundIn := func(v float64) float64 {
		return float64(k.DType.Round(float32(v)))
	}
	roundQOut := func(v float64) float64 {
		return float64(k.DType.Round(float32(v)))
	}
	roundK := func(v float64) float64 {
		if k.KVFP8 {
			return float64(FromE4M3Bits(E4M3Bits(float32(v) / ks)))
		}
		return float64(k.DType.Round(float32(v)))
	}
	roundV := func(v float64) float64 {
		if k.KVFP8 {
			return float64(FromE4M3Bits(E4M3Bits(float32(v) / vs)))
		}
		return float64(k.DType.Round(float32(v)))
	}

	rmsNorm := func(head, weight []float64) []float64 {
		out := make([]float64, k.HeadDim)
		var ss float64
		for d := range k.HeadDim {
			v := roundIn(head[d])
			out[d] = v
			ss += v * v
		}
		if !k.QKNorm {
			return out
		}
		inv := 1.0 / math.Sqrt(ss/float64(k.HeadDim)+float64(k.Eps))
		for d := range k.HeadDim {
			w := roundIn(weight[d])
			out[d] = out[d] * inv * w
		}
		return out
	}

	applyRoPE := func(vec []float64, pos int) ([]float64, error) {
		if pos < 0 || (pos+1)*k.RotaryDim > len(in.CosSinTable) {
			return nil, fmt.Errorf("position %d out of cos_sin_table range (len %d, rotary_dim %d)",
				pos, len(in.CosSinTable), k.RotaryDim)
		}
		cs := in.CosSinTable[pos*k.RotaryDim : (pos+1)*k.RotaryDim]
		out := make([]float64, k.HeadDim)
		copy(out, vec)
		half := k.RotaryDim / 2
		if k.InterleavedRoPE {
			for i := range half {
				x0 := vec[2*i]
				x1 := vec[2*i+1]
				c := float64(cs[2*i])
				s := float64(cs[2*i+1])
				out[2*i] = x0*c - x1*s
				out[2*i+1] = x1*c + x0*s
			}
		} else {
			for i := range half {
				x0 := vec[i]
				x1 := vec[i+half]
				c := float64(cs[i])
				s := float64(cs[i+half])
				out[i] = x0*c - x1*s
				out[i+half] = x1*c + x0*s
			}
		}
		return out, nil
	}

	qOut := make([]float64, m*k.NumHeads*k.HeadDim)
	for row := range m {
		pos := int(in.Positions[row])
		seqID := row
		if in.SeqIDs != nil {
			seqID = int(in.SeqIDs[row])
		}
		var tokPos int
		if in.KVPositions != nil {
			tokPos = int(in.KVPositions[row])
		} else {
			if seqID < 0 || seqID >= len(in.SeqLens) {
				return nil, fmt.Errorf("qk norm rope kv append %s reference: seq_id %d out of range", k.Name, seqID)
			}
			tokPos = int(in.SeqLens[seqID]) - 1
		}

		qkvRow := in.QKV[row*qkvStride : (row+1)*qkvStride]
		qIn := qkvRow[:k.NumHeads*k.HeadDim]
		kIn := qkvRow[k.NumHeads*k.HeadDim : (k.NumHeads+k.KVHeads)*k.HeadDim]
		vIn := qkvRow[(k.NumHeads+k.KVHeads)*k.HeadDim:]

		for h := range k.NumHeads {
			normed := rmsNorm(qIn[h*k.HeadDim:(h+1)*k.HeadDim], in.QNormWeight)
			rot, err := applyRoPE(normed, pos)
			if err != nil {
				return nil, fmt.Errorf("qk norm rope kv append %s reference: %w", k.Name, err)
			}
			dst := qOut[(row*k.NumHeads+h)*k.HeadDim : (row*k.NumHeads+h+1)*k.HeadDim]
			for d := range k.HeadDim {
				dst[d] = roundQOut(rot[d])
			}
		}

		if tokPos < 0 {
			continue
		}
		pageIdx := seqID*k.MaxPages + tokPos/k.PageSize
		if pageIdx < 0 || pageIdx >= len(in.PageTable) {
			return nil, fmt.Errorf("qk norm rope kv append %s reference: page_table index %d out of range", k.Name, pageIdx)
		}
		pageID := int(in.PageTable[pageIdx])
		if pageID < 0 {
			continue
		}
		slot := pageID*k.PageSize + tokPos%k.PageSize
		for kh := range k.KVHeads {
			normedK := rmsNorm(kIn[kh*k.HeadDim:(kh+1)*k.HeadDim], in.KNormWeight)
			rotK, err := applyRoPE(normedK, pos)
			if err != nil {
				return nil, fmt.Errorf("qk norm rope kv append %s reference: %w", k.Name, err)
			}
			kvBase := (slot*k.KVHeads + kh) * k.HeadDim
			if kvBase < 0 || kvBase+k.HeadDim > len(in.KCache) || kvBase+k.HeadDim > len(in.VCache) {
				return nil, fmt.Errorf("qk norm rope kv append %s reference: kv cache slot %d out of range", k.Name, slot)
			}
			vHead := vIn[kh*k.HeadDim : (kh+1)*k.HeadDim]
			for d := range k.HeadDim {
				in.KCache[kvBase+d] = roundK(rotK[d])
				in.VCache[kvBase+d] = roundV(roundIn(vHead[d]))
			}
		}
	}
	return qOut, nil
}
