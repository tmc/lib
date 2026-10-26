package hopper

import (
	"fmt"
	"math"
	"strings"
)

// MTPPagedAttention configures a multi-token / speculative tree-verification
// paged attention kernel where QTokensPerSeq draft tokens per sequence share a
// single KV page-table walk and apply a bitpacked causal or tree attention mask.
type MTPPagedAttention struct {
	Name          string
	NumHeads      int     // total query heads (H_q = KVHeads * G)
	KVHeads       int     // KV heads
	HeadDim       int     // D (e.g., 64, 128)
	PageSize      int     // tokens per page (e.g., 16, 64)
	MaxPages      int     // stride of page_table per batch row
	QTokensPerSeq int     // number of query/draft tokens per sequence (1..64, e.g., 4 or 8)
	Scale         float32 // softmax scale (defaults to 1/sqrt(HeadDim))
	Softcap       float32 // optional logit softcap (>0 applies cap*tanh(s/cap))
	KVFP8         bool    // E4M3 KV cache with float k_scale, v_scale
	Gate          bool    // fuse sigmoid(Gate) multiply on output
	Swizzle       bool    // 128B XOR shared-memory swizzle for K/V tiles
	WGMMA         bool    // warpgroup m64n64k32 FP8 WGMMA when G*QTokensPerSeq >= 8 && HeadDim%32 == 0 && KVFP8
}

// CausalDraftMask returns a bitpacked causal draft mask of length qTokens where
// bit j <= i is set for row i (0 <= i < qTokens).
func CausalDraftMask(qTokens int) []uint64 {
	if qTokens <= 0 {
		return nil
	}
	mask := make([]uint64, qTokens)
	for i := range qTokens {
		if i >= 63 {
			mask[i] = ^uint64(0)
		} else {
			mask[i] = (uint64(1) << (i + 1)) - 1
		}
	}
	return mask
}

// TreeDraftMask builds a bitpacked speculative draft tree mask from parent,
// where parent[0] == -1 is the root and parent[i] < i (-1 <= parent[i] < i) is
// the parent of draft node i. Bit j in row i is set iff node j is an ancestor
// of node i (or j == i).
func TreeDraftMask(parent []int) ([]uint64, error) {
	n := len(parent)
	if n == 0 || n > 64 {
		return nil, fmt.Errorf("tree draft mask: length %d outside 1..64", n)
	}
	if parent[0] != -1 {
		return nil, fmt.Errorf("tree draft mask: parent[0] = %d, want -1", parent[0])
	}
	mask := make([]uint64, n)
	mask[0] = 1
	for i := 1; i < n; i++ {
		p := parent[i]
		if p < -1 || p >= i {
			return nil, fmt.Errorf("tree draft mask: parent[%d] = %d outside -1..%d", i, p, i-1)
		}
		var m uint64
		if p >= 0 {
			m = mask[p]
		}
		mask[i] = m | (uint64(1) << i)
	}
	return mask, nil
}

func (a MTPPagedAttention) norm() (MTPPagedAttention, error) {
	if a.Name == "" {
		a.Name = "mtp_paged_attn"
	}
	if !cIdent(a.Name) {
		return a, fmt.Errorf("mtp paged attention: bad name %q", a.Name)
	}
	if a.NumHeads < 1 || a.KVHeads < 1 || a.NumHeads%a.KVHeads != 0 {
		return a, fmt.Errorf("mtp paged attention %s: bad heads %d/%d", a.Name, a.NumHeads, a.KVHeads)
	}
	if a.HeadDim < 16 || a.HeadDim > 256 || a.HeadDim%16 != 0 {
		return a, fmt.Errorf("mtp paged attention %s: bad head dim %d", a.Name, a.HeadDim)
	}
	if a.PageSize < 1 {
		return a, fmt.Errorf("mtp paged attention %s: bad page size %d", a.Name, a.PageSize)
	}
	if a.MaxPages < 1 {
		return a, fmt.Errorf("mtp paged attention %s: bad max pages %d", a.Name, a.MaxPages)
	}
	if a.QTokensPerSeq < 1 || a.QTokensPerSeq > 64 {
		return a, fmt.Errorf("mtp paged attention %s: bad QTokensPerSeq %d, want 1..64", a.Name, a.QTokensPerSeq)
	}
	g := a.NumHeads / a.KVHeads
	rows := g * a.QTokensPerSeq
	if rows > 64 {
		return a, fmt.Errorf("mtp paged attention %s: %d query rows per KV head (G=%d * QTokensPerSeq=%d), want at most 64", a.Name, rows, g, a.QTokensPerSeq)
	}
	if a.WGMMA {
		if !a.KVFP8 {
			return a, fmt.Errorf("mtp paged attention %s: WGMMA requires KVFP8", a.Name)
		}
		if a.HeadDim%32 != 0 {
			return a, fmt.Errorf("mtp paged attention %s: WGMMA requires HeadDim (%d) multiple of 32", a.Name, a.HeadDim)
		}
		if rows < 8 {
			return a, fmt.Errorf("mtp paged attention %s: WGMMA requires G*QTokensPerSeq >= 8, got %d", a.Name, rows)
		}
	}
	if a.Scale == 0 {
		a.Scale = 1 / float32(math.Sqrt(float64(a.HeadDim)))
	}
	if a.Softcap < 0 {
		return a, fmt.Errorf("mtp paged attention %s: negative softcap %g", a.Name, a.Softcap)
	}
	return a, nil
}

// Grid returns the launch grid and block dimensions for batch sequences.
// One CTA of 128 threads serves one (sequence, KV head) pair and walks the KV
// page table once for all G * QTokensPerSeq query vectors.
func (a MTPPagedAttention) Grid(batch int) (grid, block [3]int) {
	return [3]int{batch, a.KVHeads, 1}, [3]int{128, 1, 1}
}

// SharedBytes returns the dynamic shared memory size in bytes required when
// WGMMA is enabled, or 0 otherwise.
func (a MTPPagedAttention) SharedBytes() int {
	a, err := a.norm()
	if err != nil || !a.WGMMA {
		return 0
	}
	// sq and sk each hold 64 rows x max(128, HeadDim) bytes (aligned to 1024B).
	qkStride := max(128, a.HeadDim)
	qkBytes := ((64*qkStride + 1023) / 1024) * 1024
	vBytes := 64 * a.HeadDim * 4
	return 2*qkBytes + vBytes
}

func (a MTPPagedAttention) params() []string {
	kvType := "uint16_t"
	if a.KVFP8 {
		kvType = "uint8_t"
	}
	p := []string{
		"const uint16_t* __restrict__ q",
		"const " + kvType + "* __restrict__ k_cache",
		"const " + kvType + "* __restrict__ v_cache",
		"const int32_t* __restrict__ page_table",
		"const int32_t* __restrict__ seqlens",
		"const uint64_t* __restrict__ draft_mask",
	}
	if a.KVFP8 {
		p = append(p,
			"const float* __restrict__ k_scale",
			"const float* __restrict__ v_scale",
		)
	}
	if a.Gate {
		p = append(p, "const uint16_t* __restrict__ gate")
	}
	p = append(p, "uint16_t* __restrict__ out")
	return p
}

// CUDA returns the device kernel source (without [Prelude]).
func (a MTPPagedAttention) CUDA() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	if a.WGMMA {
		return a.cudaWGMMA(), nil
	}
	return a.cudaWarp(), nil
}

func (a MTPPagedAttention) cudaWarp() string {
	g := a.NumHeads / a.KVHeads
	rows := g * a.QTokensPerSeq
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: MTP speculative tree-verify paged attention, %d/%d heads, %d draft tokens/seq, head_dim %d, page %d.\n",
		a.Name, a.NumHeads, a.KVHeads, a.QTokensPerSeq, a.HeadDim, a.PageSize)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, QTOK = %d, ROWS = %d;\n",
		a.NumHeads, a.KVHeads, g, a.QTokensPerSeq, rows)
	fmt.Fprintf(&b, "    constexpr int D = %d, PAGE = %d, MAX_PAGES = %d, TILE_K = 16;\n",
		a.HeadDim, a.PageSize, a.MaxPages)
	fmt.Fprintf(&b, "    constexpr float SM_SCALE = %s, SOFTCAP = %s;\n",
		floatLiteral(a.Scale), floatLiteral(a.Softcap))

	if a.KVFP8 {
		b.WriteString("    const float ks = k_scale ? k_scale[0] : 1.0f;\n")
		b.WriteString("    const float vs = v_scale ? v_scale[0] : 1.0f;\n")
	} else {
		b.WriteString("    constexpr float ks = 1.0f, vs = 1.0f;\n")
	}
	b.WriteString(`    const int b = blockIdx.x, kvh = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int seqlen = seqlens[b];
    const int prefix_len = seqlen - QTOK;
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    alignas(16) __shared__ float sq[ROWS * D];
    alignas(16) __shared__ float sk[TILE_K * D];
    alignas(16) __shared__ float sv[TILE_K * D];
    __shared__ uint64_t smask[QTOK];

    if (tid < QTOK) {
        uint64_t m = 0;
        if (draft_mask != nullptr) {
            m = draft_mask[b * QTOK + tid];
        } else {
            m = (tid >= 63) ? ~0ULL : ((1ULL << (tid + 1)) - 1ULL);
        }
        smask[tid] = m;
    }
    for (int i = tid; i < ROWS * D; i += 128) {
        const int r = i / D, d = i % D;
        const int q_idx = r / G, h_local = r % G;
        const int h = kvh * G + h_local;
        const size_t q_off = ((size_t)(b * QTOK + q_idx) * HQ + h) * D + d;
        sq[r * D + d] = eg_bf16_to_f32(q[q_off]);
    }
    __syncthreads();
`)
	if a.Swizzle {
		b.WriteString("    auto swz = [](int tk, int d) { return tk * D + ((d ^ ((tk & 7) << 2)) & (D - 1)); };\n")
	} else {
		b.WriteString("    auto swz = [](int tk, int d) { return tk * D + d; };\n")
	}
	b.WriteString(`    constexpr int RPT = (ROWS + 3) / 4;
    constexpr int DPT = (D + 31) / 32;
    float m_max[RPT];
    float l_sum[RPT];
    float acc[RPT][DPT];
#pragma unroll
    for (int rp = 0; rp < RPT; ++rp) {
        m_max[rp] = NEG_INF;
        l_sum[rp] = 0.0f;
#pragma unroll
        for (int dp = 0; dp < DPT; ++dp) acc[rp][dp] = 0.0f;
    }

    for (int k0 = 0; k0 < seqlen; k0 += TILE_K) {
        for (int i = tid; i < TILE_K * D; i += 128) {
            const int tk = i / D, d = i % D;
            const int k_pos = k0 + tk;
            float kv_k = 0.0f, kv_v = 0.0f;
            if (k_pos < seqlen) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE];
                const int slot = page * PAGE + (k_pos % PAGE);
                const size_t kv_off = ((size_t)slot * HKV + kvh) * D + d;
`)
	if a.KVFP8 {
		b.WriteString(`                kv_k = eg_e4m3_to_f32(k_cache[kv_off]) * ks;
                kv_v = eg_e4m3_to_f32(v_cache[kv_off]) * vs;
`)
	} else {
		b.WriteString(`                kv_k = eg_bf16_to_f32(k_cache[kv_off]);
                kv_v = eg_bf16_to_f32(v_cache[kv_off]);
`)
	}
	b.WriteString(`            }
            const int idx = swz(tk, d);
            sk[idx] = kv_k;
            sv[idx] = kv_v;
        }
        __syncthreads();

        const int tk_end = min(TILE_K, seqlen - k0);
#pragma unroll
        for (int rp = 0; rp < RPT; ++rp) {
            const int r = warp + rp * 4;
            if (r < ROWS) {
                const int q_idx = r / G;
                const uint64_t mask_bits = smask[q_idx];
                for (int tk = 0; tk < tk_end; ++tk) {
                    const int k_pos = k0 + tk;
                    bool vis = true;
                    if (k_pos >= prefix_len) {
                        const int draft_col = k_pos - prefix_len;
                        vis = ((mask_bits >> draft_col) & 1ULL) != 0ULL;
                    }
                    float part = 0.0f;
#pragma unroll
                    for (int dp = 0; dp < DPT; ++dp) {
                        const int d = lane + dp * 32;
                        if (d < D) {
                            part += sq[r * D + d] * sk[swz(tk, d)];
                        }
                    }
                    float score = eg_warp_sum(part) * SM_SCALE;
                    if (SOFTCAP > 0.0f) {
                        score = SOFTCAP * tanhf(score / SOFTCAP);
                    }
                    if (!vis) {
                        score = NEG_INF;
                    }
                    const float m_prev = m_max[rp];
                    const float m_new = fmaxf(m_prev, score);
                    const float base = (m_new == NEG_INF) ? 0.0f : m_new;
                    const float corr = expf(m_prev - base);
                    const float p = vis ? expf(score - base) : 0.0f;
                    m_max[rp] = m_new;
                    l_sum[rp] = l_sum[rp] * corr + p;
#pragma unroll
                    for (int dp = 0; dp < DPT; ++dp) {
                        const int d = lane + dp * 32;
                        if (d < D) {
                            acc[rp][dp] = acc[rp][dp] * corr + p * sv[swz(tk, d)];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int rp = 0; rp < RPT; ++rp) {
        const int r = warp + rp * 4;
        if (r < ROWS) {
            const int q_idx = r / G, h_local = r % G;
            const int h = kvh * G + h_local;
            const float inv_l = (l_sum[rp] > 0.0f) ? (1.0f / l_sum[rp]) : 0.0f;
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) {
                const int d = lane + dp * 32;
                if (d < D) {
                    const size_t o_off = ((size_t)(b * QTOK + q_idx) * HQ + h) * D + d;
                    float val = acc[rp][dp] * inv_l;
`)
	if a.Gate {
		b.WriteString("                    val *= eg_sigmoid(eg_bf16_to_f32(gate[o_off]));\n")
	}
	b.WriteString(`                    out[o_off] = eg_f32_to_bf16(val);
                }
            }
        }
    }
}
`)
	return b.String()
}

func (a MTPPagedAttention) cudaWGMMA() string {
	g := a.NumHeads / a.KVHeads
	rows := g * a.QTokensPerSeq
	var b strings.Builder
	if a.Swizzle {
		b.WriteString(wgmmaSwizzleHelpers)
	}
	wg := a.Name + "_wgmma"
	b.WriteString(wgmmaE4M3Func(wg, 64))
	fmt.Fprintf(&b, "// %s: MTP speculative tree-verify paged attention (WGMMA m64n64k32 FP8), %d/%d heads, %d draft tokens/seq, head_dim %d, page %d.\n",
		a.Name, a.NumHeads, a.KVHeads, a.QTokensPerSeq, a.HeadDim, a.PageSize)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, QTOK = %d, ROWS = %d, MROWS = 64;\n",
		a.NumHeads, a.KVHeads, g, a.QTokensPerSeq, rows)
	fmt.Fprintf(&b, "    constexpr int D = %d, PAGE = %d, MAX_PAGES = %d, STEP = 64, KSTEPS = D / 32;\n",
		a.HeadDim, a.PageSize, a.MaxPages)
	fmt.Fprintf(&b, "    constexpr float SM_SCALE = %s, SOFTCAP = %s;\n",
		floatLiteral(a.Scale), floatLiteral(a.Softcap))
	b.WriteString(`    const float ks = k_scale ? k_scale[0] : 1.0f;
    const float vs = v_scale ? v_scale[0] : 1.0f;
    const float qk_scale = SM_SCALE * ks;
    const int b = blockIdx.x, kvh = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g_lane = lane / 4, c = lane % 4;
    const int seqlen = seqlens[b];
    const int prefix_len = seqlen - QTOK;
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    EG_DYNAMIC_SMEM(unsigned char, dsmem);
    unsigned char* const sq = dsmem;
    unsigned char* const sk = dsmem + 8192;
    float* const sv = reinterpret_cast<float*>(dsmem + 16384);
    __shared__ uint64_t smask[QTOK];

    if (tid < QTOK) {
        uint64_t m = 0;
        if (draft_mask != nullptr) {
            m = draft_mask[b * QTOK + tid];
        } else {
            m = (tid >= 63) ? ~0ULL : ((1ULL << (tid + 1)) - 1ULL);
        }
        smask[tid] = m;
    }
`)
	if a.Swizzle {
		b.WriteString("    auto kcore = [](int ks_idx, int r, int ch) { return (int)eg_swizzle_128b((uint32_t)(r * 128 + (ks_idx * 2 + ch) * 16)); };\n")
	} else {
		b.WriteString("    auto kcore = [](int ks_idx, int r, int ch) { return ((r / 8) * (KSTEPS * 2) + ks_idx * 2 + ch) * 128 + (r % 8) * 16; };\n")
	}
	b.WriteString(`    // Convert BF16 query rows [ROWS, D] into E4M3 core-matrix layout in sq.
    for (int idx = tid; idx < MROWS * KSTEPS * 2; idx += 128) {
        const int r = idx / (KSTEPS * 2);
        const int rem = idx % (KSTEPS * 2);
        const int ks_idx = rem / 2, ch = rem % 2;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) {
            const int q_idx = r / G, h_local = r % G;
            const int h = kvh * G + h_local;
            const size_t q_off = ((size_t)(b * QTOK + q_idx) * HQ + h) * D + ks_idx * 32 + ch * 16;
            uint8_t bytes[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float f0 = eg_bf16_to_f32(q[q_off + 2 * j]);
                const float f1 = eg_bf16_to_f32(q[q_off + 2 * j + 1]);
                const uint16_t p = eg_f32x2_to_e4m3x2(f0, f1);
                bytes[2 * j] = (uint8_t)(p & 0xffu);
                bytes[2 * j + 1] = (uint8_t)(p >> 8);
            }
            memcpy(&w, bytes, 16);
        }
        *reinterpret_cast<eg_u32x4*>(sq + kcore(ks_idx, r, ch)) = w;
    }
    eg_fence_proxy_async();
    __syncthreads();

    constexpr int DPT = (D + 3) / 4;
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f};
    float o[2][DPT] = {};

    for (int k0 = 0; k0 < seqlen; k0 += STEP) {
        // Stage 64 K rows into sk in E4M3 WGMMA layout and 64 V rows into sv as FP32.
        for (int idx = tid; idx < STEP * KSTEPS * 2; idx += 128) {
            const int tk = idx / (KSTEPS * 2);
            const int rem = idx % (KSTEPS * 2);
            const int ks_idx = rem / 2, ch = rem % 2;
            const int k_pos = k0 + tk;
            eg_u32x4 kw = {0, 0, 0, 0};
            if (k_pos < seqlen) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE];
                const int slot = page * PAGE + (k_pos % PAGE);
                const size_t kv_off = ((size_t)slot * HKV + kvh) * D + ks_idx * 32 + ch * 16;
                kw = eg_ldg16(k_cache + kv_off);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t vb = v_cache[kv_off + j];
                    sv[tk * D + ks_idx * 32 + ch * 16 + j] = eg_e4m3_to_f32(vb) * vs;
                }
            } else {
                for (int j = 0; j < 16; ++j) {
                    sv[tk * D + ks_idx * 32 + ch * 16 + j] = 0.0f;
                }
            }
            *reinterpret_cast<eg_u32x4*>(sk + kcore(ks_idx, tk, ch)) = kw;
        }
        eg_fence_proxy_async();
        __syncthreads();

        float s[32] = {};
        eg_wgmma_fence();
#pragma unroll
        for (int ks_idx = 0; ks_idx < KSTEPS; ++ks_idx) {
`)
	if a.Swizzle {
		fmt.Fprintf(&b, "            %s(s, eg_wgmma_desc_128b(sq + ks_idx * 32, 1024), eg_wgmma_desc_128b(sk + ks_idx * 32, 1024));\n", wg)
	} else {
		fmt.Fprintf(&b, "            %s(s, eg_wgmma_desc(sq + ks_idx * 256, 128, KSTEPS * 256), eg_wgmma_desc(sk + ks_idx * 256, 128, KSTEPS * 256));\n", wg)
	}
	b.WriteString(`        }
        eg_wgmma_commit();
        eg_wgmma_wait<0>();

#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = warp * 16 + g_lane + 8 * h;
            const int q_idx = (r < ROWS) ? (r / G) : 0;
            const uint64_t mask_bits = (r < ROWS) ? smask[q_idx] : 0ULL;
            float mx = NEG_INF;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tk = nt * 8 + 2 * c + e;
                    const int k_pos = k0 + tk;
                    bool vis = (r < ROWS) && (k_pos < seqlen);
                    if (vis && k_pos >= prefix_len) {
                        const int draft_col = k_pos - prefix_len;
                        vis = ((mask_bits >> draft_col) & 1ULL) != 0ULL;
                    }
                    float& x = s[nt * 4 + 2 * h + e];
                    float score = x * qk_scale;
                    if (SOFTCAP > 0.0f) {
                        score = SOFTCAP * tanhf(score / SOFTCAP);
                    }
                    x = vis ? score : NEG_INF;
                    mx = fmaxf(mx, x);
                }
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mnew = fmaxf(m[h], mx);
            const float base = (mnew == NEG_INF) ? 0.0f : mnew;
            const float corr = expf(m[h] - base);
            m[h] = mnew;
            l[h] *= corr;
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) {
                o[h][dp] *= corr;
            }
            float p_row[16];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float& x = s[nt * 4 + 2 * h + e];
                    const float p = (x == NEG_INF) ? 0.0f : expf(x - base);
                    x = p;
                    p_row[nt * 2 + e] = p;
                    l[h] += p;
                }
            }
            // Accumulate P * V across all 64 keys of the step using warp shuffles across the 4-lane group c=0..3.
#pragma unroll
            for (int src_c = 0; src_c < 4; ++src_c) {
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float p_val = __shfl_sync(0xffffffffu, p_row[nt * 2 + e], g_lane * 4 + src_c);
                        const int tk = nt * 8 + 2 * src_c + e;
#pragma unroll
                        for (int dp = 0; dp < DPT; ++dp) {
                            const int d = c + dp * 4;
                            if (d < D) {
                                o[h][dp] += p_val * sv[tk * D + d];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
        const int r = warp * 16 + g_lane + 8 * h;
        if (r < ROWS) {
            const int q_idx = r / G, h_local = r % G;
            const int head = kvh * G + h_local;
            const float inv_l = (l[h] > 0.0f) ? (1.0f / l[h]) : 0.0f;
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) {
                const int d = c + dp * 4;
                if (d < D) {
                    const size_t o_off = ((size_t)(b * QTOK + q_idx) * HQ + head) * D + d;
                    float val = o[h][dp] * inv_l;
`)
	if a.Gate {
		b.WriteString("                    val *= eg_sigmoid(eg_bf16_to_f32(gate[o_off]));\n")
	}
	b.WriteString(`                    out[o_off] = eg_f32_to_bf16(val);
                }
            }
        }
    }
}
`)
	return b.String()
}

// Launcher returns a host function
//
//	void Name_launch(int batch, <kernel params...>, cudaStream_t stream)
//
// that configures dynamic shared memory if needed and launches the kernel.
func (a MTPPagedAttention) Launcher() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	params := a.params()
	names := paramNames(params)
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int batch, %s, cudaStream_t stream) {\n", a.Name, strings.Join(params, ", "))
	smem := "0"
	if a.WGMMA {
		smem = "smem"
		dsmem := a.SharedBytes()
		fmt.Fprintf(&b, "    constexpr int smem = %d;\n", dsmem)
		fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", a.Name)
	}
	fmt.Fprintf(&b, "    const dim3 grid(batch, %d, 1), block(128, 1, 1);\n", a.KVHeads)
	fmt.Fprintf(&b, "    if (batch > 0) EG_LAUNCH(%s, grid, block, %s, stream, %s);\n}\n", a.Name, smem, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (a MTPPagedAttention) Source() (string, error) {
	src, err := a.CUDA()
	if err != nil {
		return "", err
	}
	l, err := a.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// MTPPagedCache holds the inputs for [MTPPagedAttention.Reference].
type MTPPagedCache struct {
	KBF16, VBF16 []uint16 // when !KVFP8: [totalPages * PageSize, KVHeads, HeadDim]
	KFP8, VFP8   []uint8  // when KVFP8:  [totalPages * PageSize, KVHeads, HeadDim]
	PageTable    []int32  // [batch, MaxPages]
	SeqLens      []int32  // [batch]
	DraftMask    []uint64 // nil (causal default) or [batch * QTokensPerSeq]
	KScale       float32  // when KVFP8 (0 defaults to 1)
	VScale       float32  // when KVFP8 (0 defaults to 1)
	Gate         []uint16 // when Gate: BF16 [batch * QTokensPerSeq * NumHeads * HeadDim]
}

// Reference computes the float64 reference output of [MTPPagedAttention] for
// BF16 queries q ([batch * QTokensPerSeq * NumHeads * HeadDim]).
func (a MTPPagedAttention) Reference(q []uint16, c MTPPagedCache) ([]float64, error) {
	a, err := a.norm()
	if err != nil {
		return nil, err
	}
	batch := len(c.SeqLens)
	qTotal := batch * a.QTokensPerSeq * a.NumHeads * a.HeadDim
	if len(q) != qTotal {
		return nil, fmt.Errorf("mtp paged attention %s reference: len(q) = %d, want %d", a.Name, len(q), qTotal)
	}
	if len(c.PageTable) != batch*a.MaxPages {
		return nil, fmt.Errorf("mtp paged attention %s reference: len(page_table) = %d, want %d", a.Name, len(c.PageTable), batch*a.MaxPages)
	}
	if c.DraftMask != nil && len(c.DraftMask) != batch*a.QTokensPerSeq {
		return nil, fmt.Errorf("mtp paged attention %s reference: len(draft_mask) = %d, want %d", a.Name, len(c.DraftMask), batch*a.QTokensPerSeq)
	}
	if a.Gate && len(c.Gate) != qTotal {
		return nil, fmt.Errorf("mtp paged attention %s reference: len(gate) = %d, want %d", a.Name, len(c.Gate), qTotal)
	}
	ks := c.KScale
	if ks == 0 {
		ks = 1
	}
	vs := c.VScale
	if vs == 0 {
		vs = 1
	}
	g := a.NumHeads / a.KVHeads
	out := make([]float64, qTotal)

	for b := range batch {
		seqlen := int(c.SeqLens[b])
		if seqlen < a.QTokensPerSeq {
			return nil, fmt.Errorf("mtp paged attention %s reference: seqlen[%d] = %d < QTokensPerSeq %d", a.Name, b, seqlen, a.QTokensPerSeq)
		}
		prefixLen := seqlen - a.QTokensPerSeq
		for qIdx := range a.QTokensPerSeq {
			var maskBits uint64
			if c.DraftMask != nil {
				maskBits = c.DraftMask[b*a.QTokensPerSeq+qIdx]
			} else if qIdx >= 63 {
				maskBits = ^uint64(0)
			} else {
				maskBits = (uint64(1) << (qIdx + 1)) - 1
			}
			for h := range a.NumHeads {
				kvh := h / g
				qOff := ((b*a.QTokensPerSeq+qIdx)*a.NumHeads + h) * a.HeadDim
				qv := q[qOff : qOff+a.HeadDim]

				logits := make([]float64, seqlen)
				vis := make([]bool, seqlen)
				mx := math.Inf(-1)
				for kPos := range seqlen {
					if kPos >= prefixLen {
						draftCol := kPos - prefixLen
						if ((maskBits >> draftCol) & 1) == 0 {
							logits[kPos] = math.Inf(-1)
							continue
						}
					}
					vis[kPos] = true
					page := int(c.PageTable[b*a.MaxPages+kPos/a.PageSize])
					slot := page*a.PageSize + (kPos % a.PageSize)
					kvOff := (slot*a.KVHeads + kvh) * a.HeadDim
					var dot float64
					for d := range a.HeadDim {
						var qf float64
						if a.WGMMA {
							qf = float64(FromE4M3Bits(E4M3Bits(FromBF16Bits(qv[d]))))
						} else {
							qf = float64(FromBF16Bits(qv[d]))
						}
						var kf float64
						if a.KVFP8 {
							kf = float64(FromE4M3Bits(c.KFP8[kvOff+d])) * float64(ks)
						} else {
							kf = float64(FromBF16Bits(c.KBF16[kvOff+d]))
						}
						dot += qf * kf
					}
					s := dot * float64(a.Scale)
					if a.Softcap > 0 {
						s = float64(a.Softcap) * math.Tanh(s/float64(a.Softcap))
					}
					logits[kPos] = s
					if s > mx {
						mx = s
					}
				}

				var sum float64
				for kPos := range seqlen {
					if !vis[kPos] || math.IsInf(mx, -1) {
						logits[kPos] = 0
						continue
					}
					logits[kPos] = math.Exp(logits[kPos] - mx)
					sum += logits[kPos]
				}

				oSlice := out[qOff : qOff+a.HeadDim]
				if sum > 0 {
					for kPos := range seqlen {
						if !vis[kPos] {
							continue
						}
						w := logits[kPos] / sum
						page := int(c.PageTable[b*a.MaxPages+kPos/a.PageSize])
						slot := page*a.PageSize + (kPos % a.PageSize)
						kvOff := (slot*a.KVHeads + kvh) * a.HeadDim
						for d := range a.HeadDim {
							var vf float64
							if a.KVFP8 {
								vf = float64(FromE4M3Bits(c.VFP8[kvOff+d])) * float64(vs)
							} else {
								vf = float64(FromBF16Bits(c.VBF16[kvOff+d]))
							}
							oSlice[d] += w * vf
						}
					}
				}
				if a.Gate {
					for d := range a.HeadDim {
						gf := float64(FromBF16Bits(c.Gate[qOff+d]))
						sig := 1.0 / (1.0 + math.Exp(-gf))
						oSlice[d] *= sig
					}
				}
				for d := range a.HeadDim {
					oSlice[d] = float64(FromBF16Bits(BF16Bits(float32(oSlice[d]))))
				}
			}
		}
	}
	return out, nil
}
