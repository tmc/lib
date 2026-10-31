package hopper

import (
	"cmp"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"strings"
)

// QuestKVVQConfig configures Sub-2-Bit / 4-Bit Walsh-Hadamard KV Cache Vector
// Quantization (KV-VQ) fused with Quest bounding-box sparse page-selection
// paged attention.
type QuestKVVQConfig struct {
	Name       string
	Batch      int
	NumQHeads  int
	NumKVHeads int
	HeadDim    int // Power of 2 (e.g. 64, 128) for warp-butterfly FWHT
	PageSize   int
	TotalPages int
	TopKPages  int // Quest critical pages selected per query head
	KVBits     int // 2 or 4 bits per element after Walsh-Hadamard rotation
	Arch       Arch
}

func isPowerOfTwo(n int) bool {
	return n >= 2 && (n&(n-1)) == 0
}

func (cfg QuestKVVQConfig) norm() (QuestKVVQConfig, error) {
	if cfg.Name == "" {
		cfg.Name = "quest_kvvq_paged_attn"
	}
	if !cIdent(cfg.Name) {
		return cfg, fmt.Errorf("quest kvvq: bad name %q", cfg.Name)
	}
	if cfg.Batch < 1 {
		cfg.Batch = 1
	}
	if cfg.NumQHeads < 1 {
		cfg.NumQHeads = 32
	}
	if cfg.NumKVHeads < 1 {
		cfg.NumKVHeads = 8
	}
	if cfg.NumQHeads%cfg.NumKVHeads != 0 {
		return cfg, fmt.Errorf("quest kvvq %s: bad heads %d/%d", cfg.Name, cfg.NumQHeads, cfg.NumKVHeads)
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = 64
	}
	if !isPowerOfTwo(cfg.HeadDim) || cfg.HeadDim < 16 || cfg.HeadDim > 256 {
		return cfg, fmt.Errorf("quest kvvq %s: HeadDim %d must be a power of 2 in 16..256", cfg.Name, cfg.HeadDim)
	}
	if cfg.PageSize < 1 {
		cfg.PageSize = 16
	}
	if cfg.TotalPages < 1 {
		cfg.TotalPages = 64
	}
	if cfg.TopKPages < 1 {
		cfg.TopKPages = min(8, cfg.TotalPages)
	}
	if cfg.TopKPages > cfg.TotalPages {
		cfg.TopKPages = cfg.TotalPages
	}
	if cfg.KVBits == 0 {
		cfg.KVBits = 2
	}
	if cfg.KVBits != 2 && cfg.KVBits != 4 {
		return cfg, fmt.Errorf("quest kvvq %s: KVBits %d, want 2 or 4", cfg.Name, cfg.KVBits)
	}
	if cfg.Arch == "" {
		cfg.Arch = TargetSM121A
	}
	return cfg, nil
}

// FastWalshHadamardHost computes the normalized orthogonal Fast Walsh-Hadamard
// Transform (H_d / sqrt(d)) over x. If len(x) is a multiple of a power-of-2
// head dimension (or itself a power of 2 >= 2), it transforms the vector
// in O(n log n) operations. Because H_d / sqrt(d) is symmetric and orthogonal,
// it is an exact involution: FastWalshHadamardHost(FastWalshHadamardHost(x)) == x.
func FastWalshHadamardHost(x []float32) ([]float32, error) {
	d := len(x)
	if !isPowerOfTwo(d) {
		return nil, fmt.Errorf("fast walsh-hadamard: length %d is not a power of 2 >= 2", d)
	}
	buf := make([]float64, d)
	for i, v := range x {
		buf[i] = float64(v)
	}
	for h := 1; h < d; h <<= 1 {
		for i := 0; i < d; i += h << 1 {
			for j := i; j < i+h; j++ {
				u := buf[j]
				v := buf[j+h]
				buf[j] = u + v
				buf[j+h] = u - v
			}
		}
	}
	invSqrtD := 1.0 / math.Sqrt(float64(d))
	out := make([]float32, d)
	for i, v := range buf {
		out[i] = float32(v * invSqrtD)
	}
	return out, nil
}

// fwhtInPlace applies the normalized orthogonal Fast Walsh-Hadamard Transform
// to a single head vector of length headDim (must be a power of 2).
func fwhtSingleHead(vec []float32) ([]float32, error) {
	return FastWalshHadamardHost(vec)
}

// QuantizeKVPageHadamard computes the Quest per-page bounding box (kMin, kMax)
// in R^{headDim} across all tokens in kPage prior to rotation, applies the
// normalized Fast Walsh-Hadamard Transform (FWHT) across each headDim-sized
// token vector in kPage and vPage to suppress activation outliers, and packs
// the rotated coordinates into 2-bit (4 elements/byte) or 4-bit (2 elements/byte)
// codes with per-token absmax scales (scalesK, scalesV).
func QuantizeKVPageHadamard(kPage, vPage []float32, headDim, kvBits int) (packedK, packedV []byte, scalesK, scalesV []float32, kMin, kMax []float32, err error) {
	if !isPowerOfTwo(headDim) {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("quantize kv page hadamard: headDim %d must be a power of 2 >= 2", headDim)
	}
	if kvBits != 2 && kvBits != 4 {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("quantize kv page hadamard: kvBits %d must be 2 or 4", kvBits)
	}
	if len(kPage) == 0 || len(kPage)%headDim != 0 {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("quantize kv page hadamard: len(kPage)=%d must be a positive multiple of headDim=%d", len(kPage), headDim)
	}
	if len(vPage) != len(kPage) {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("quantize kv page hadamard: len(vPage)=%d != len(kPage)=%d", len(vPage), len(kPage))
	}

	numTokens := len(kPage) / headDim
	kMin = make([]float32, headDim)
	kMax = make([]float32, headDim)
	for d := range headDim {
		kMin[d] = kPage[d]
		kMax[d] = kPage[d]
	}
	for t := 1; t < numTokens; t++ {
		row := kPage[t*headDim : (t+1)*headDim]
		for d := range headDim {
			if row[d] < kMin[d] {
				kMin[d] = row[d]
			}
			if row[d] > kMax[d] {
				kMax[d] = row[d]
			}
		}
	}

	elemsPerByte := 8 / kvBits
	if (numTokens*headDim)%elemsPerByte != 0 {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("quantize kv page hadamard: total elements %d not divisible by %d", numTokens*headDim, elemsPerByte)
	}
	packedBytes := (numTokens * headDim) / elemsPerByte
	packedK = make([]byte, packedBytes)
	packedV = make([]byte, packedBytes)
	scalesK = make([]float32, numTokens)
	scalesV = make([]float32, numTokens)

	qMax := float32((int(1) << (kvBits - 1)) - 1) // 1.5 for 2-bit symmetric or 1/3 levels; use uniform mid-riser/symmetric
	if kvBits == 2 {
		qMax = 1.5 // codes 0,1,2,3 -> {-1.5, -0.5, +0.5, +1.5} * scale
	} else {
		qMax = 7.5 // codes 0..15 -> {-7.5, ..., +7.5} * scale
	}

	quantizeRow := func(rot []float32, packedDst []byte) float32 {
		var absMax float32
		for _, v := range rot {
			av := float32(math.Abs(float64(v)))
			if av > absMax {
				absMax = av
			}
		}
		scale := absMax / qMax
		if scale == 0 {
			scale = 1e-8
		}
		invScale := 1.0 / scale
		for d, v := range rot {
			// Map v / scale in [-qMax, +qMax] to integer code in [0, 2^kvBits - 1]:
			// dequant(code) = (float32(code) - qMax) * scale
			raw := float64(v*invScale + qMax)
			code := int(math.Round(raw))
			maxCode := (1 << kvBits) - 1
			if code < 0 {
				code = 0
			} else if code > maxCode {
				code = maxCode
			}
			byteIdx := d / elemsPerByte
			bitShift := (d % elemsPerByte) * kvBits
			packedDst[byteIdx] |= byte(code) << bitShift
		}
		return scale
	}

	bytesPerToken := headDim / elemsPerByte
	for t := range numTokens {
		kRow := kPage[t*headDim : (t+1)*headDim]
		vRow := vPage[t*headDim : (t+1)*headDim]
		kRot, err := fwhtSingleHead(kRow)
		if err != nil {
			return nil, nil, nil, nil, nil, nil, err
		}
		vRot, err := fwhtSingleHead(vRow)
		if err != nil {
			return nil, nil, nil, nil, nil, nil, err
		}
		scalesK[t] = quantizeRow(kRot, packedK[t*bytesPerToken:(t+1)*bytesPerToken])
		scalesV[t] = quantizeRow(vRot, packedV[t*bytesPerToken:(t+1)*bytesPerToken])
	}

	return packedK, packedV, scalesK, scalesV, kMin, kMax, nil
}

// DequantizeKVPageHadamard unpacks 2-bit or 4-bit quantized K and V pages and
// applies the inverse Fast Walsh-Hadamard Transform (identical to forward FWHT
// because H_d / sqrt(d) is self-inverse) to reconstruct the original domain.
func DequantizeKVPageHadamard(packedK, packedV []byte, scalesK, scalesV []float32, headDim, kvBits int) (kOut, vOut []float32, err error) {
	if !isPowerOfTwo(headDim) {
		return nil, nil, fmt.Errorf("dequantize kv page hadamard: headDim %d must be a power of 2", headDim)
	}
	if kvBits != 2 && kvBits != 4 {
		return nil, nil, fmt.Errorf("dequantize kv page hadamard: kvBits %d must be 2 or 4", kvBits)
	}
	numTokens := len(scalesK)
	if len(scalesV) != numTokens {
		return nil, nil, fmt.Errorf("dequantize kv page hadamard: scales mismatch %d vs %d", len(scalesK), len(scalesV))
	}
	elemsPerByte := 8 / kvBits
	bytesPerToken := headDim / elemsPerByte
	if len(packedK) != numTokens*bytesPerToken || len(packedV) != numTokens*bytesPerToken {
		return nil, nil, fmt.Errorf("dequantize kv page hadamard: invalid packed byte length")
	}
	qMax := float32(1.5)
	if kvBits == 4 {
		qMax = 7.5
	}
	mask := byte((1 << kvBits) - 1)

	dequantRow := func(packed []byte, scale float32) ([]float32, error) {
		rot := make([]float32, headDim)
		for d := range headDim {
			byteIdx := d / elemsPerByte
			bitShift := (d % elemsPerByte) * kvBits
			code := (packed[byteIdx] >> bitShift) & mask
			rot[d] = (float32(code) - qMax) * scale
		}
		return fwhtSingleHead(rot)
	}

	kOut = make([]float32, numTokens*headDim)
	vOut = make([]float32, numTokens*headDim)
	for t := range numTokens {
		kr, err := dequantRow(packedK[t*bytesPerToken:(t+1)*bytesPerToken], scalesK[t])
		if err != nil {
			return nil, nil, err
		}
		vr, err := dequantRow(packedV[t*bytesPerToken:(t+1)*bytesPerToken], scalesV[t])
		if err != nil {
			return nil, nil, err
		}
		copy(kOut[t*headDim:(t+1)*headDim], kr)
		copy(vOut[t*headDim:(t+1)*headDim], vr)
	}
	return kOut, vOut, nil
}

// SelectQuestTopKPages computes the Quest upper-bound attention score for each
// page p:
//
//	score(p) = sum_{d=0}^{headDim-1} max(q[d] * pageKMin[p][d], q[d] * pageKMax[p][d])
//
// and returns the indices and upper-bound scores of the top-K pages in
// descending score order (ties broken by lower page index).
func SelectQuestTopKPages(q []float32, pageKMin, pageKMax [][]float32, topK int) ([]int, []float32) {
	numPages := min(len(pageKMin), len(pageKMax))
	if numPages == 0 || topK <= 0 || len(q) == 0 {
		return nil, nil
	}
	if topK > numPages {
		topK = numPages
	}
	headDim := len(q)
	type pageScore struct {
		idx   int
		score float32
	}
	scored := make([]pageScore, 0, numPages)
	for p := range numPages {
		kMin := pageKMin[p]
		kMax := pageKMax[p]
		if len(kMin) < headDim || len(kMax) < headDim {
			continue
		}
		var sum float32
		for d := range headDim {
			prodMin := q[d] * kMin[d]
			prodMax := q[d] * kMax[d]
			if prodMin >= prodMax {
				sum += prodMin
			} else {
				sum += prodMax
			}
		}
		scored = append(scored, pageScore{idx: p, score: sum})
	}
	slices.SortFunc(scored, func(a, b pageScore) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return cmp.Compare(a.idx, b.idx)
	})
	if topK > len(scored) {
		topK = len(scored)
	}
	indices := make([]int, topK)
	scores := make([]float32, topK)
	for i := range topK {
		indices[i] = scored[i].idx
		scores[i] = scored[i].score
	}
	return indices, scores
}

// CUDA emits the fused Quest bounding-box sparse page-selection + Sub-2-Bit /
// 4-Bit Walsh-Hadamard KV-VQ PagedAttention CUDA kernel.
//
// Because the normalized Walsh-Hadamard transform is orthogonal,
// <q, H_d^T dequant(k_hat)> == <H_d q, dequant(k_hat)>. The kernel:
//  1. Evaluates the Quest per-page bounding box upper-bound score
//     sum_d max(q_d * k_min[p,d], q_d * k_max[p,d]) in registers and selects
//     the TopKPages critical pages without loading non-critical KV pages;
//  2. Rotates the query vector in registers via a 5-instruction warp-butterfly
//     Fast Walsh-Hadamard Transform using __shfl_xor_sync(0xffffffff, val, lane_mask);
//  3. Streams only the TopKPages packed 2-bit/4-bit KV pages, unpacks and
//     scales K and V in registers, accumulates online softmax over the rotated
//     inner products, and applies the inverse 5-instruction warp-butterfly
//     FWHT on the output accumulator before writing out.
func (cfg QuestKVVQConfig) CUDA() string {
	normed, err := cfg.norm()
	if err != nil {
		normed = QuestKVVQConfig{
			Name:       "quest_kvvq_paged_attn",
			Batch:      1,
			NumQHeads:  32,
			NumKVHeads: 8,
			HeadDim:    64,
			PageSize:   16,
			TotalPages: 64,
			TopKPages:  8,
			KVBits:     2,
			Arch:       TargetSM121A,
		}
	}
	g := normed.NumQHeads / normed.NumKVHeads
	elemsPerByte := 8 / normed.KVBits
	bytesPerVec := normed.HeadDim / elemsPerByte
	dpt := normed.HeadDim / 32
	log2D := bits.TrailingZeros(uint(normed.HeadDim))
	invSqrtD := float32(1.0 / math.Sqrt(float64(normed.HeadDim)))
	smScale := float32(1.0 / math.Sqrt(float64(normed.HeadDim)))
	qMax := float32(1.5)
	if normed.KVBits == 4 {
		qMax = 7.5
	}
	bitMask := (1 << normed.KVBits) - 1

	var b strings.Builder
	fmt.Fprintf(&b, "// %s: Quest bounding-box Top-%d/%d page-sparse + %d-bit Walsh-Hadamard KV-VQ PagedAttention (arch=%s, Hq=%d, Hkv=%d, D=%d, page=%d, log2D=%d).\n",
		normed.Name, normed.TopKPages, normed.TotalPages, normed.KVBits, normed.Arch,
		normed.NumQHeads, normed.NumKVHeads, normed.HeadDim, normed.PageSize, log2D)

	b.WriteString(`#ifndef EG_QUEST_KVVQ_FWHT_DEFINED
#define EG_QUEST_KVVQ_FWHT_DEFINED 1
// eg_fwht_warp32 executes the 5-instruction in-register warp-butterfly
// Fast Walsh-Hadamard Transform across 32 lanes via __shfl_xor_sync.
EG_DEV float eg_fwht_warp32(float val, int lane) {
#pragma unroll
    for (int lane_mask = 1; lane_mask < 32; lane_mask <<= 1) {
        const float partner = __shfl_xor_sync(0xffffffff, val, lane_mask);
        val = ((lane & lane_mask) == 0) ? (val + partner) : (partner - val);
    }
    return val;
}
#endif // EG_QUEST_KVVQ_FWHT_DEFINED

`)

	fmt.Fprintf(&b, `__global__ void __launch_bounds__(128) %s(
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_vq_cache,
    const uint8_t* __restrict__ v_vq_cache,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    const float* __restrict__ page_k_min,
    const float* __restrict__ page_k_max,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ num_pages_per_seq,
    int32_t* __restrict__ selected_pages_out,
    float* __restrict__ out) {
`, normed.Name)
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, D = %d, DPT = %d;\n",
		normed.NumQHeads, normed.NumKVHeads, g, normed.HeadDim, dpt)
	fmt.Fprintf(&b, "    constexpr int PAGE_SIZE = %d, TOTAL_PAGES = %d, TOPK_PAGES = %d;\n",
		normed.PageSize, normed.TotalPages, normed.TopKPages)
	fmt.Fprintf(&b, "    constexpr int KV_BITS = %d, ELEMS_PER_BYTE = %d, BYTES_PER_VEC = %d;\n",
		normed.KVBits, elemsPerByte, bytesPerVec)
	fmt.Fprintf(&b, "    constexpr uint32_t CODE_MASK = %du;\n", bitMask)
	fmt.Fprintf(&b, "    constexpr float Q_OFFSET = %s, FWHT_NORM = %s, SM_SCALE = %s;\n",
		floatLiteral(qMax), floatLiteral(invSqrtD), floatLiteral(smScale))
	b.WriteString(`    const int b = blockIdx.x, qh = blockIdx.y;
    if (b >= ` + fmt.Sprint(normed.Batch) + ` || qh >= HQ) return;
    const int kvh = qh / G;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    __shared__ float s_page_scores[TOTAL_PAGES];
    __shared__ int32_t s_top_pages[TOPK_PAGES];

    // Load query head q[b, qh, :] into warp registers.
    float q_reg[DPT];
#pragma unroll
    for (int dp = 0; dp < DPT; ++dp) {
        const int d = dp * 32 + lane;
        q_reg[dp] = q[((size_t)b * HQ + qh) * D + d];
    }

    // Stage 1: Quest bounding-box upper-bound page criticality scoring:
    //   score(p) = sum_{d=0}^{D-1} max(q[d] * k_min[p, d], q[d] * k_max[p, d])
    const int n_pages = min(num_pages_per_seq ? num_pages_per_seq[b] : TOTAL_PAGES, TOTAL_PAGES);
    for (int p = warp; p < TOTAL_PAGES; p += 4) {
        float ub = NEG_INF;
        if (p < n_pages) {
            const int phys_page = page_table ? page_table[b * TOTAL_PAGES + p] : p;
            const size_t bbox_off = ((size_t)phys_page * HKV + kvh) * D;
            float part = 0.0f;
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) {
                const int d = dp * 32 + lane;
                const float qv = q_reg[dp];
                const float pmin = qv * page_k_min[bbox_off + d];
                const float pmax = qv * page_k_max[bbox_off + d];
                part += fmaxf(pmin, pmax);
            }
            ub = eg_warp_sum(part);
        }
        if (lane == 0) {
            s_page_scores[p] = ub;
        }
    }
    __syncthreads();

    // Select TopKPages critical pages in warp 0.
    const int k_sel = min(TOPK_PAGES, n_pages);
    if (warp == 0) {
        for (int s = 0; s < k_sel; ++s) {
            float best_s = NEG_INF;
            int best_p = 0x7fffffff;
            for (int p = lane; p < n_pages; p += 32) {
                const float sc = s_page_scores[p];
                if (sc > best_s || (sc == best_s && p < best_p)) {
                    best_s = sc;
                    best_p = p;
                }
            }
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                const float os = __shfl_xor_sync(0xffffffff, best_s, mask);
                const int op = __shfl_xor_sync(0xffffffff, best_p, mask);
                if (os > best_s || (os == best_s && op < best_p)) {
                    best_s = os;
                    best_p = op;
                }
            }
            if (lane == 0) {
                s_top_pages[s] = best_p;
                s_page_scores[best_p] = NEG_INF;
                if (selected_pages_out) {
                    selected_pages_out[(b * HQ + qh) * TOPK_PAGES + s] = best_p;
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();

    // Stage 2: Rotate Q in registers via 5-instruction warp-butterfly FWHT
    // so <H_d q, k_hat> is evaluated directly in the Hadamard domain without
    // per-key shared-memory barriers.
    float q_rot[DPT];
#pragma unroll
    for (int dp = 0; dp < DPT; ++dp) {
        q_rot[dp] = eg_fwht_warp32(q_reg[dp], lane);
    }
    if constexpr (DPT == 2) {
        const float u0 = q_rot[0], u1 = q_rot[1];
        q_rot[0] = (u0 + u1) * FWHT_NORM;
        q_rot[1] = (u0 - u1) * FWHT_NORM;
    } else if constexpr (DPT == 4) {
        const float a0 = q_rot[0] + q_rot[1], a1 = q_rot[0] - q_rot[1];
        const float a2 = q_rot[2] + q_rot[3], a3 = q_rot[2] - q_rot[3];
        q_rot[0] = (a0 + a2) * FWHT_NORM;
        q_rot[1] = (a1 + a3) * FWHT_NORM;
        q_rot[2] = (a0 - a2) * FWHT_NORM;
        q_rot[3] = (a1 - a3) * FWHT_NORM;
    } else {
#pragma unroll
        for (int dp = 0; dp < DPT; ++dp) q_rot[dp] *= FWHT_NORM;
    }

    // Stage 3: PagedAttention over the selected Quest Top-K pages with
    // on-the-fly 2-bit / 4-bit KV-VQ dequantization in the Hadamard domain.
    if (warp == 0) {
        float m_max = NEG_INF;
        float l_sum = 0.0f;
        float o_rot[DPT];
#pragma unroll
        for (int dp = 0; dp < DPT; ++dp) o_rot[dp] = 0.0f;

        for (int s = 0; s < k_sel; ++s) {
            const int log_page = s_top_pages[s];
            const int phys_page = page_table ? page_table[b * TOTAL_PAGES + log_page] : log_page;
            for (int t = 0; t < PAGE_SIZE; ++t) {
                const size_t slot = ((size_t)phys_page * PAGE_SIZE + t) * HKV + kvh;
                const uint8_t* const k_packed = k_vq_cache + slot * BYTES_PER_VEC;
                const uint8_t* const v_packed = v_vq_cache + slot * BYTES_PER_VEC;
                const float ks = k_scales[slot];
                const float vs = v_scales[slot];

                float dot_part = 0.0f;
                float v_deq[DPT];
#pragma unroll
                for (int dp = 0; dp < DPT; ++dp) {
                    const int d = dp * 32 + lane;
                    const int byte_idx = d / ELEMS_PER_BYTE;
                    const int bit_shift = (d % ELEMS_PER_BYTE) * KV_BITS;
                    const uint32_t k_code = (k_packed[byte_idx] >> bit_shift) & CODE_MASK;
                    const uint32_t v_code = (v_packed[byte_idx] >> bit_shift) & CODE_MASK;
                    const float kv_rot = ((float)k_code - Q_OFFSET) * ks;
                    v_deq[dp] = ((float)v_code - Q_OFFSET) * vs;
                    dot_part += q_rot[dp] * kv_rot;
                }
                const float score = eg_warp_sum(dot_part) * SM_SCALE;
                const float m_prev = m_max;
                const float m_new = fmaxf(m_prev, score);
                const float base = (m_new == NEG_INF) ? 0.0f : m_new;
                const float corr = expf(m_prev - base);
                const float p_weight = expf(score - base);
                m_max = m_new;
                l_sum = l_sum * corr + p_weight;
#pragma unroll
                for (int dp = 0; dp < DPT; ++dp) {
                    o_rot[dp] = o_rot[dp] * corr + p_weight * v_deq[dp];
                }
            }
        }

        // Normalize softmax and apply inverse warp-butterfly FWHT (self-inverse
        // H_d / sqrt(d)) to transform the output accumulator back from the
        // Hadamard domain to the standard activation coordinates.
        const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
        float o_final[DPT];
#pragma unroll
        for (int dp = 0; dp < DPT; ++dp) {
            o_final[dp] = eg_fwht_warp32(o_rot[dp] * inv_l, lane);
        }
        if constexpr (DPT == 2) {
            const float u0 = o_final[0], u1 = o_final[1];
            o_final[0] = (u0 + u1) * FWHT_NORM;
            o_final[1] = (u0 - u1) * FWHT_NORM;
        } else if constexpr (DPT == 4) {
            const float a0 = o_final[0] + o_final[1], a1 = o_final[0] - o_final[1];
            const float a2 = o_final[2] + o_final[3], a3 = o_final[2] - o_final[3];
            o_final[0] = (a0 + a2) * FWHT_NORM;
            o_final[1] = (a1 + a3) * FWHT_NORM;
            o_final[2] = (a0 - a2) * FWHT_NORM;
            o_final[3] = (a1 - a3) * FWHT_NORM;
        } else {
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) o_final[dp] *= FWHT_NORM;
        }

#pragma unroll
        for (int dp = 0; dp < DPT; ++dp) {
            const int d = dp * 32 + lane;
            out[((size_t)b * HQ + qh) * D + d] = o_final[dp];
        }
    }
}
`)
	return b.String()
}
