package hopper

import (
	"fmt"
	"math"
	"strings"
)

// MLAPagedAttention configures a Multi-head Latent Attention (MLA) kernel with
// matrix absorption over a paged compressed KV cache.
//
// With matrix absorption, the key up-projection is absorbed into the non-RoPE
// query vectors q_nope [batch, NumHeads, LatentDim] and the value up-projection
// is absorbed into the output projection after attention. All NumHeads query
// heads therefore share a single compressed latent KV head (KVHeads = 1) per
// token:
//   - c_kv [totalSlots, LatentDim]: compressed latent KV vector
//   - k_pe [totalSlots, RopeDim]:   decoupled shared RoPE key vector
//
// Each query head h in 0..NumHeads-1 has q_nope[b, h, LatentDim] and
// q_pe[b, h, RopeDim]. Its logit against token t is:
//
//	score(b, h, t) = Scale * (dot(q_nope[b, h], c_kv[b, t]) + dot(q_pe[b, h], k_pe[b, t]))
//
// and its latent attention output is:
//
//	out[b, h, d] = sum_t P(b, h, t) * c_kv[b, t, d]   for d in 0..LatentDim-1.
//
// Because all NumHeads query heads share the single (c_kv, k_pe) tile per
// sequence, a CTA loads c_kv and k_pe into shared memory once and computes
// Q_nope*C_KV^T + Q_pe*K_pe^T and P*C_KV across up to 64 query heads
// simultaneously, naturally filling a Hopper warpgroup WGMMA tile whenever
// NumHeads >= 8, LatentDim%32 == 0, and RopeDim%32 == 0.
type MLAPagedAttention struct {
	Name      string
	NumHeads  int     // H (all NumHeads share the single latent KV head!)
	LatentDim int     // D_c (kv_lora_rank, e.g., 64, 128, 512)
	RopeDim   int     // D_r (qk_rope_head_dim, e.g., 32, 64)
	PageSize  int     // tokens per page
	MaxPages  int     // page_table stride per batch row
	Scale     float32 // softmax scale (defaults to 1/sqrt(LatentDim + RopeDim))
	KVFP8     bool    // E4M3 c_kv and k_pe with c_scale, kpe_scale
	Swizzle   bool    // 128B XOR shared-memory swizzling
	WGMMA     bool    // Hopper warpgroup m64n64k32 FP8 WGMMA when KVFP8 && NumHeads >= 8 && LatentDim%32 == 0 && RopeDim%32 == 0
}

func (a MLAPagedAttention) norm() (MLAPagedAttention, error) {
	if a.Name == "" {
		a.Name = "mla_paged_attn"
	}
	if !cIdent(a.Name) {
		return a, fmt.Errorf("mla paged attention: bad name %q", a.Name)
	}
	if a.NumHeads < 1 || a.NumHeads > 128 {
		return a, fmt.Errorf("mla paged attention %s: bad NumHeads %d, want 1..128", a.Name, a.NumHeads)
	}
	if a.LatentDim < 16 || a.LatentDim > 512 || a.LatentDim%16 != 0 {
		return a, fmt.Errorf("mla paged attention %s: bad LatentDim %d, want multiple of 16 in 16..512", a.Name, a.LatentDim)
	}
	if a.RopeDim < 16 || a.RopeDim > 256 || a.RopeDim%16 != 0 {
		return a, fmt.Errorf("mla paged attention %s: bad RopeDim %d, want multiple of 16 in 16..256", a.Name, a.RopeDim)
	}
	if a.PageSize < 1 {
		return a, fmt.Errorf("mla paged attention %s: bad PageSize %d", a.Name, a.PageSize)
	}
	if a.MaxPages < 1 {
		return a, fmt.Errorf("mla paged attention %s: bad MaxPages %d", a.Name, a.MaxPages)
	}
	if a.WGMMA {
		if !a.KVFP8 {
			return a, fmt.Errorf("mla paged attention %s: WGMMA requires KVFP8", a.Name)
		}
		if a.NumHeads < 8 {
			return a, fmt.Errorf("mla paged attention %s: WGMMA requires NumHeads >= 8, got %d", a.Name, a.NumHeads)
		}
		if a.LatentDim%32 != 0 || a.RopeDim%32 != 0 {
			return a, fmt.Errorf("mla paged attention %s: WGMMA requires LatentDim (%d) and RopeDim (%d) multiples of 32", a.Name, a.LatentDim, a.RopeDim)
		}
	}
	if a.Scale == 0 {
		a.Scale = 1 / float32(math.Sqrt(float64(a.LatentDim+a.RopeDim)))
	}
	return a, nil
}

// headGroups returns the number of 64-head tiles needed to cover NumHeads.
func (a MLAPagedAttention) headGroups() int {
	return (a.NumHeads + 63) / 64
}

// Grid returns the launch grid and block dimensions for batch sequences.
// One CTA of 128 threads serves up to 64 query heads of one sequence and loads
// the sequence's shared c_kv and k_pe tiles once across all heads in the CTA.
func (a MLAPagedAttention) Grid(batch int) (grid, block [3]int) {
	return [3]int{batch, max(1, a.headGroups()), 1}, [3]int{128, 1, 1}
}

// SharedBytes returns the dynamic shared memory size in bytes required by the
// kernel, or 0 if static shared memory is used.
func (a MLAPagedAttention) SharedBytes() int {
	a, err := a.norm()
	if err != nil || !a.WGMMA {
		return 0
	}
	kcBlocks := (a.LatentDim/32 + 3) / 4
	krBlocks := (a.RopeDim/32 + 3) / 4
	cBytes := kcBlocks * 8192
	rBytes := krBlocks * 8192
	vBytes := 64 * a.LatentDim * 4
	return 2*cBytes + 2*rBytes + vBytes
}

func (a MLAPagedAttention) params() []string {
	kvType := "uint16_t"
	if a.KVFP8 {
		kvType = "uint8_t"
	}
	p := []string{
		"const uint16_t* __restrict__ q_nope",
		"const uint16_t* __restrict__ q_pe",
		"const " + kvType + "* __restrict__ c_kv_cache",
		"const " + kvType + "* __restrict__ k_pe_cache",
		"const int32_t* __restrict__ page_table",
		"const int32_t* __restrict__ seqlens",
	}
	if a.KVFP8 {
		p = append(p,
			"const float* __restrict__ c_scale",
			"const float* __restrict__ kpe_scale",
		)
	}
	p = append(p, "uint16_t* __restrict__ out")
	return p
}

// CUDA returns the device kernel source (without [Prelude]).
func (a MLAPagedAttention) CUDA() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	if a.WGMMA {
		return a.cudaWGMMA(), nil
	}
	return a.cudaWarp(), nil
}

func (a MLAPagedAttention) cudaWarp() string {
	rows := min(64, a.NumHeads)
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: Multi-head Latent Attention (matrix absorption), %d heads, latent_dim %d, rope_dim %d, page %d.\n",
		a.Name, a.NumHeads, a.LatentDim, a.RopeDim, a.PageSize)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, MAX_ROWS = %d, D_C = %d, D_R = %d;\n",
		a.NumHeads, rows, a.LatentDim, a.RopeDim)
	fmt.Fprintf(&b, "    constexpr int PAGE = %d, MAX_PAGES = %d, TILE_K = 16;\n",
		a.PageSize, a.MaxPages)
	fmt.Fprintf(&b, "    constexpr float SM_SCALE = %s;\n", floatLiteral(a.Scale))

	if a.KVFP8 {
		b.WriteString("    const float cs = c_scale ? c_scale[0] : 1.0f;\n")
		b.WriteString("    const float kpes = kpe_scale ? kpe_scale[0] : 1.0f;\n")
	} else {
		b.WriteString("    constexpr float cs = 1.0f, kpes = 1.0f;\n")
	}
	b.WriteString(`    const int b = blockIdx.x, hg = blockIdx.y;
    const int h0 = hg * 64;
    const int rows = min(MAX_ROWS, HQ - h0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int seqlen = seqlens[b];
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    alignas(16) __shared__ float sq_nope[MAX_ROWS * D_C];
    alignas(16) __shared__ float sq_pe[MAX_ROWS * D_R];
    alignas(16) __shared__ float sc_kv[TILE_K * D_C];
    alignas(16) __shared__ float sk_pe[TILE_K * D_R];

    for (int i = tid; i < rows * D_C; i += 128) {
        const int r = i / D_C, d = i % D_C;
        const size_t off = ((size_t)b * HQ + (h0 + r)) * D_C + d;
        sq_nope[r * D_C + d] = eg_bf16_to_f32(q_nope[off]);
    }
    for (int i = tid; i < rows * D_R; i += 128) {
        const int r = i / D_R, d = i % D_R;
        const size_t off = ((size_t)b * HQ + (h0 + r)) * D_R + d;
        sq_pe[r * D_R + d] = eg_bf16_to_f32(q_pe[off]);
    }
    __syncthreads();
`)
	if a.Swizzle {
		b.WriteString("    auto swz_c = [](int tk, int d) { return tk * D_C + (d ^ ((tk & 3) << 2)); };\n")
		b.WriteString("    auto swz_r = [](int tk, int d) { return tk * D_R + (d ^ ((tk & 3) << 2)); };\n")
	} else {
		b.WriteString("    auto swz_c = [](int tk, int d) { return tk * D_C + d; };\n")
		b.WriteString("    auto swz_r = [](int tk, int d) { return tk * D_R + d; };\n")
	}
	b.WriteString(`    constexpr int RPT = (MAX_ROWS + 3) / 4;
    constexpr int DPT_C = (D_C + 31) / 32;
    constexpr int DPT_R = (D_R + 31) / 32;
    float m_max[RPT];
    float l_sum[RPT];
    float acc[RPT][DPT_C];
#pragma unroll
    for (int rp = 0; rp < RPT; ++rp) {
        m_max[rp] = NEG_INF;
        l_sum[rp] = 0.0f;
#pragma unroll
        for (int dp = 0; dp < DPT_C; ++dp) acc[rp][dp] = 0.0f;
    }

    for (int k0 = 0; k0 < seqlen; k0 += TILE_K) {
        for (int i = tid; i < TILE_K * D_C; i += 128) {
            const int tk = i / D_C, d = i % D_C;
            const int k_pos = k0 + tk;
            float cv = 0.0f;
            if (k_pos < seqlen) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE];
                const int slot = page * PAGE + (k_pos % PAGE);
                const size_t c_off = (size_t)slot * D_C + d;
`)
	if a.KVFP8 {
		b.WriteString("                cv = eg_e4m3_to_f32(c_kv_cache[c_off]) * cs;\n")
	} else {
		b.WriteString("                cv = eg_bf16_to_f32(c_kv_cache[c_off]);\n")
	}
	b.WriteString(`            }
            sc_kv[swz_c(tk, d)] = cv;
        }
        for (int i = tid; i < TILE_K * D_R; i += 128) {
            const int tk = i / D_R, d = i % D_R;
            const int k_pos = k0 + tk;
            float kv = 0.0f;
            if (k_pos < seqlen) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE];
                const int slot = page * PAGE + (k_pos % PAGE);
                const size_t r_off = (size_t)slot * D_R + d;
`)
	if a.KVFP8 {
		b.WriteString("                kv = eg_e4m3_to_f32(k_pe_cache[r_off]) * kpes;\n")
	} else {
		b.WriteString("                kv = eg_bf16_to_f32(k_pe_cache[r_off]);\n")
	}
	b.WriteString(`            }
            sk_pe[swz_r(tk, d)] = kv;
        }
        __syncthreads();

        const int tk_end = min(TILE_K, seqlen - k0);
#pragma unroll
        for (int rp = 0; rp < RPT; ++rp) {
            const int r = warp + rp * 4;
            if (r < rows) {
                for (int tk = 0; tk < tk_end; ++tk) {
                    float part = 0.0f;
#pragma unroll
                    for (int dp = 0; dp < DPT_C; ++dp) {
                        const int d = lane + dp * 32;
                        if (d < D_C) {
                            part += sq_nope[r * D_C + d] * sc_kv[swz_c(tk, d)];
                        }
                    }
#pragma unroll
                    for (int dp = 0; dp < DPT_R; ++dp) {
                        const int d = lane + dp * 32;
                        if (d < D_R) {
                            part += sq_pe[r * D_R + d] * sk_pe[swz_r(tk, d)];
                        }
                    }
                    const float score = eg_warp_sum(part) * SM_SCALE;
                    const float m_prev = m_max[rp];
                    const float m_new = fmaxf(m_prev, score);
                    const float base = (m_new == NEG_INF) ? 0.0f : m_new;
                    const float corr = expf(m_prev - base);
                    const float p = expf(score - base);
                    m_max[rp] = m_new;
                    l_sum[rp] = l_sum[rp] * corr + p;
#pragma unroll
                    for (int dp = 0; dp < DPT_C; ++dp) {
                        const int d = lane + dp * 32;
                        if (d < D_C) {
                            acc[rp][dp] = acc[rp][dp] * corr + p * sc_kv[swz_c(tk, d)];
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
        if (r < rows) {
            const int h = h0 + r;
            const float inv_l = (l_sum[rp] > 0.0f) ? (1.0f / l_sum[rp]) : 0.0f;
#pragma unroll
            for (int dp = 0; dp < DPT_C; ++dp) {
                const int d = lane + dp * 32;
                if (d < D_C) {
                    const size_t o_off = ((size_t)b * HQ + h) * D_C + d;
                    out[o_off] = eg_f32_to_bf16(acc[rp][dp] * inv_l);
                }
            }
        }
    }
}
`)
	return b.String()
}

func (a MLAPagedAttention) cudaWGMMA() string {
	rows := min(64, a.NumHeads)
	kcBlocks := (a.LatentDim/32 + 3) / 4
	krBlocks := (a.RopeDim/32 + 3) / 4
	cBytes := kcBlocks * 8192
	rBytes := krBlocks * 8192
	var b strings.Builder
	if a.Swizzle {
		b.WriteString(wgmmaSwizzleHelpers)
	}
	wg := a.Name + "_wgmma"
	b.WriteString(wgmmaE4M3Func(wg, 64))
	fmt.Fprintf(&b, "// %s: Multi-head Latent Attention (WGMMA m64n64k32 FP8), %d heads, latent_dim %d, rope_dim %d, page %d.\n",
		a.Name, a.NumHeads, a.LatentDim, a.RopeDim, a.PageSize)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, MAX_ROWS = %d, MROWS = 64;\n",
		a.NumHeads, rows)
	fmt.Fprintf(&b, "    constexpr int D_C = %d, D_R = %d, KC_STEPS = D_C / 32, KR_STEPS = D_R / 32;\n",
		a.LatentDim, a.RopeDim)
	fmt.Fprintf(&b, "    constexpr int PAGE = %d, MAX_PAGES = %d, STEP = 64;\n",
		a.PageSize, a.MaxPages)
	fmt.Fprintf(&b, "    constexpr float SM_SCALE = %s;\n", floatLiteral(a.Scale))
	fmt.Fprintf(&b, "    constexpr int C_BYTES = %d, R_BYTES = %d;\n", cBytes, rBytes)
	b.WriteString(`    const float cs = c_scale ? c_scale[0] : 1.0f;
    const float kpes = kpe_scale ? kpe_scale[0] : 1.0f;
    const int b = blockIdx.x, hg = blockIdx.y;
    const int h0 = hg * 64;
    const int rows = min(MAX_ROWS, HQ - h0);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g_lane = lane / 4, c = lane % 4;
    const int seqlen = seqlens[b];
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    EG_DYNAMIC_SMEM(unsigned char, dsmem);
    unsigned char* const sq_nope = dsmem;
    unsigned char* const sc_kv = dsmem + C_BYTES;
    unsigned char* const sq_pe = dsmem + 2 * C_BYTES;
    unsigned char* const sk_pe = dsmem + 2 * C_BYTES + R_BYTES;
    float* const sv_c = reinterpret_cast<float*>(dsmem + 2 * C_BYTES + 2 * R_BYTES);
`)
	if a.Swizzle {
		b.WriteString("    auto ccore = [](int ks_idx, int r, int ch) { return (ks_idx / 4) * 8192 + (int)eg_swizzle_128b((uint32_t)(r * 128 + ((ks_idx % 4) * 2 + ch) * 16)); };\n")
		b.WriteString("    auto rcore = [](int ks_idx, int r, int ch) { return (ks_idx / 4) * 8192 + (int)eg_swizzle_128b((uint32_t)(r * 128 + ((ks_idx % 4) * 2 + ch) * 16)); };\n")
	} else {
		b.WriteString("    auto ccore = [](int ks_idx, int r, int ch) { return ((r / 8) * (KC_STEPS * 2) + ks_idx * 2 + ch) * 128 + (r % 8) * 16; };\n")
		b.WriteString("    auto rcore = [](int ks_idx, int r, int ch) { return ((r / 8) * (KR_STEPS * 2) + ks_idx * 2 + ch) * 128 + (r % 8) * 16; };\n")
	}
	b.WriteString(`    // Convert BF16 q_nope [rows, D_C] into E4M3 WGMMA layout in sq_nope.
    for (int idx = tid; idx < MROWS * KC_STEPS * 2; idx += 128) {
        const int r = idx / (KC_STEPS * 2);
        const int rem = idx % (KC_STEPS * 2);
        const int ks_idx = rem / 2, ch = rem % 2;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < rows) {
            const size_t q_off = ((size_t)b * HQ + (h0 + r)) * D_C + ks_idx * 32 + ch * 16;
            uint8_t bytes[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float f0 = eg_bf16_to_f32(q_nope[q_off + 2 * j]);
                const float f1 = eg_bf16_to_f32(q_nope[q_off + 2 * j + 1]);
                const uint16_t p = eg_f32x2_to_e4m3x2(f0, f1);
                bytes[2 * j] = (uint8_t)(p & 0xffu);
                bytes[2 * j + 1] = (uint8_t)(p >> 8);
            }
            memcpy(&w, bytes, 16);
        }
        *reinterpret_cast<eg_u32x4*>(sq_nope + ccore(ks_idx, r, ch)) = w;
    }
    // Convert BF16 q_pe [rows, D_R] into E4M3 WGMMA layout in sq_pe.
    for (int idx = tid; idx < MROWS * KR_STEPS * 2; idx += 128) {
        const int r = idx / (KR_STEPS * 2);
        const int rem = idx % (KR_STEPS * 2);
        const int ks_idx = rem / 2, ch = rem % 2;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < rows) {
            const size_t q_off = ((size_t)b * HQ + (h0 + r)) * D_R + ks_idx * 32 + ch * 16;
            uint8_t bytes[16];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float f0 = eg_bf16_to_f32(q_pe[q_off + 2 * j]);
                const float f1 = eg_bf16_to_f32(q_pe[q_off + 2 * j + 1]);
                const uint16_t p = eg_f32x2_to_e4m3x2(f0, f1);
                bytes[2 * j] = (uint8_t)(p & 0xffu);
                bytes[2 * j + 1] = (uint8_t)(p >> 8);
            }
            memcpy(&w, bytes, 16);
        }
        *reinterpret_cast<eg_u32x4*>(sq_pe + rcore(ks_idx, r, ch)) = w;
    }
    eg_fence_proxy_async();
    __syncthreads();

    constexpr int DPT_C = (D_C + 3) / 4;
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f};
    float o[2][DPT_C] = {};

    for (int k0 = 0; k0 < seqlen; k0 += STEP) {
        // Stage 64 c_kv rows into sc_kv (E4M3 WGMMA layout) and sv_c (dequantized FP32).
        for (int idx = tid; idx < STEP * KC_STEPS * 2; idx += 128) {
            const int tk = idx / (KC_STEPS * 2);
            const int rem = idx % (KC_STEPS * 2);
            const int ks_idx = rem / 2, ch = rem % 2;
            const int k_pos = k0 + tk;
            eg_u32x4 cw = {0, 0, 0, 0};
            if (k_pos < seqlen) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE];
                const int slot = page * PAGE + (k_pos % PAGE);
                const size_t c_off = (size_t)slot * D_C + ks_idx * 32 + ch * 16;
                cw = eg_ldg16(c_kv_cache + c_off);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t cb = c_kv_cache[c_off + j];
                    sv_c[tk * D_C + ks_idx * 32 + ch * 16 + j] = eg_e4m3_to_f32(cb) * cs;
                }
            } else {
                for (int j = 0; j < 16; ++j) {
                    sv_c[tk * D_C + ks_idx * 32 + ch * 16 + j] = 0.0f;
                }
            }
            *reinterpret_cast<eg_u32x4*>(sc_kv + ccore(ks_idx, tk, ch)) = cw;
        }
        // Stage 64 k_pe rows into sk_pe (E4M3 WGMMA layout).
        for (int idx = tid; idx < STEP * KR_STEPS * 2; idx += 128) {
            const int tk = idx / (KR_STEPS * 2);
            const int rem = idx % (KR_STEPS * 2);
            const int ks_idx = rem / 2, ch = rem % 2;
            const int k_pos = k0 + tk;
            eg_u32x4 rw = {0, 0, 0, 0};
            if (k_pos < seqlen) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE];
                const int slot = page * PAGE + (k_pos % PAGE);
                const size_t r_off = (size_t)slot * D_R + ks_idx * 32 + ch * 16;
                rw = eg_ldg16(k_pe_cache + r_off);
            }
            *reinterpret_cast<eg_u32x4*>(sk_pe + rcore(ks_idx, tk, ch)) = rw;
        }
        eg_fence_proxy_async();
        __syncthreads();

        float s_nope[32] = {};
        float s_pe[32] = {};
        eg_wgmma_fence();
#pragma unroll
        for (int ks_idx = 0; ks_idx < KC_STEPS; ++ks_idx) {
`)
	if a.Swizzle {
		fmt.Fprintf(&b, "            %s(s_nope, eg_wgmma_desc_128b(sq_nope + (ks_idx / 4) * 8192 + (ks_idx %% 4) * 32, 1024), eg_wgmma_desc_128b(sc_kv + (ks_idx / 4) * 8192 + (ks_idx %% 4) * 32, 1024));\n", wg)
	} else {
		fmt.Fprintf(&b, "            %s(s_nope, eg_wgmma_desc(sq_nope + ks_idx * 256, 128, KC_STEPS * 256), eg_wgmma_desc(sc_kv + ks_idx * 256, 128, KC_STEPS * 256));\n", wg)
	}
	b.WriteString(`        }
#pragma unroll
        for (int ks_idx = 0; ks_idx < KR_STEPS; ++ks_idx) {
`)
	if a.Swizzle {
		fmt.Fprintf(&b, "            %s(s_pe, eg_wgmma_desc_128b(sq_pe + (ks_idx / 4) * 8192 + (ks_idx %% 4) * 32, 1024), eg_wgmma_desc_128b(sk_pe + (ks_idx / 4) * 8192 + (ks_idx %% 4) * 32, 1024));\n", wg)
	} else {
		fmt.Fprintf(&b, "            %s(s_pe, eg_wgmma_desc(sq_pe + ks_idx * 256, 128, KR_STEPS * 256), eg_wgmma_desc(sk_pe + ks_idx * 256, 128, KR_STEPS * 256));\n", wg)
	}
	b.WriteString(`        }
        eg_wgmma_commit();
        eg_wgmma_wait<0>();

#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = warp * 16 + g_lane + 8 * h;
            float mx = NEG_INF;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int tk = nt * 8 + 2 * c + e;
                    const int k_pos = k0 + tk;
                    const int idx = nt * 4 + 2 * h + e;
                    const bool vis = (r < rows) && (k_pos < seqlen);
                    const float score = (s_nope[idx] * cs + s_pe[idx] * kpes) * SM_SCALE;
                    s_nope[idx] = vis ? score : NEG_INF;
                    mx = fmaxf(mx, s_nope[idx]);
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
            for (int dp = 0; dp < DPT_C; ++dp) {
                o[h][dp] *= corr;
            }
            float p_row[16];
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int idx = nt * 4 + 2 * h + e;
                    const float x = s_nope[idx];
                    const float p = (x == NEG_INF) ? 0.0f : expf(x - base);
                    p_row[nt * 2 + e] = p;
                    l[h] += p;
                }
            }
            // Accumulate P * C_KV across all 64 keys using warp shuffles across c=0..3.
#pragma unroll
            for (int src_c = 0; src_c < 4; ++src_c) {
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float p_val = __shfl_sync(0xffffffffu, p_row[nt * 2 + e], g_lane * 4 + src_c);
                        const int tk = nt * 8 + 2 * src_c + e;
#pragma unroll
                        for (int dp = 0; dp < DPT_C; ++dp) {
                            const int d = c + dp * 4;
                            if (d < D_C) {
                                o[h][dp] += p_val * sv_c[tk * D_C + d];
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
        if (r < rows) {
            const int head = h0 + r;
            const float inv_l = (l[h] > 0.0f) ? (1.0f / l[h]) : 0.0f;
#pragma unroll
            for (int dp = 0; dp < DPT_C; ++dp) {
                const int d = c + dp * 4;
                if (d < D_C) {
                    const size_t o_off = ((size_t)b * HQ + head) * D_C + d;
                    out[o_off] = eg_f32_to_bf16(o[h][dp] * inv_l);
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
func (a MLAPagedAttention) Launcher() (string, error) {
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
	fmt.Fprintf(&b, "    const dim3 grid(batch, %d, 1), block(128, 1, 1);\n", a.headGroups())
	fmt.Fprintf(&b, "    if (batch > 0) EG_LAUNCH(%s, grid, block, %s, stream, %s);\n}\n", a.Name, smem, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (a MLAPagedAttention) Source() (string, error) {
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

// MLAPagedCache holds the paged latent KV cache inputs for [MLAPagedAttention.Reference].
type MLAPagedCache struct {
	CKVBF16, KPEBF16 []uint16 // when !KVFP8: c_kv [totalSlots, LatentDim], k_pe [totalSlots, RopeDim]
	CKVFP8, KPEFP8   []uint8  // when KVFP8:  c_kv [totalSlots, LatentDim], k_pe [totalSlots, RopeDim]
	PageTable        []int32  // [batch, MaxPages]
	SeqLens          []int32  // [batch]
	CScale           float32  // when KVFP8 (0 defaults to 1)
	KPEScale         float32  // when KVFP8 (0 defaults to 1)
}

// Reference computes the float64 reference output of [MLAPagedAttention] for
// BF16 queries qNope ([batch, NumHeads, LatentDim]) and qPe ([batch, NumHeads, RopeDim]).
func (a MLAPagedAttention) Reference(qNope, qPe []uint16, c MLAPagedCache) ([]float64, error) {
	a, err := a.norm()
	if err != nil {
		return nil, err
	}
	batch := len(c.SeqLens)
	wantNope := batch * a.NumHeads * a.LatentDim
	wantPe := batch * a.NumHeads * a.RopeDim
	if len(qNope) != wantNope {
		return nil, fmt.Errorf("mla paged attention %s reference: len(qNope) = %d, want %d", a.Name, len(qNope), wantNope)
	}
	if len(qPe) != wantPe {
		return nil, fmt.Errorf("mla paged attention %s reference: len(qPe) = %d, want %d", a.Name, len(qPe), wantPe)
	}
	if len(c.PageTable) != batch*a.MaxPages {
		return nil, fmt.Errorf("mla paged attention %s reference: len(page_table) = %d, want %d", a.Name, len(c.PageTable), batch*a.MaxPages)
	}
	cs := c.CScale
	if cs == 0 {
		cs = 1
	}
	kpes := c.KPEScale
	if kpes == 0 {
		kpes = 1
	}
	out := make([]float64, wantNope)

	for b := range batch {
		seqlen := int(c.SeqLens[b])
		if seqlen < 1 {
			return nil, fmt.Errorf("mla paged attention %s reference: seqlen[%d] = %d < 1", a.Name, b, seqlen)
		}
		for h := range a.NumHeads {
			qnOff := (b*a.NumHeads + h) * a.LatentDim
			qrOff := (b*a.NumHeads + h) * a.RopeDim
			qn := qNope[qnOff : qnOff+a.LatentDim]
			qr := qPe[qrOff : qrOff+a.RopeDim]

			logits := make([]float64, seqlen)
			mx := math.Inf(-1)
			for t := range seqlen {
				page := int(c.PageTable[b*a.MaxPages+t/a.PageSize])
				slot := page*a.PageSize + (t % a.PageSize)
				cOff := slot * a.LatentDim
				rOff := slot * a.RopeDim

				var dotNope, dotPe float64
				for d := range a.LatentDim {
					var qf float64
					if a.WGMMA {
						qf = float64(FromE4M3Bits(E4M3Bits(FromBF16Bits(qn[d]))))
					} else {
						qf = float64(FromBF16Bits(qn[d]))
					}
					var cf float64
					if a.KVFP8 {
						cf = float64(FromE4M3Bits(c.CKVFP8[cOff+d])) * float64(cs)
					} else {
						cf = float64(FromBF16Bits(c.CKVBF16[cOff+d]))
					}
					dotNope += qf * cf
				}
				for d := range a.RopeDim {
					var qf float64
					if a.WGMMA {
						qf = float64(FromE4M3Bits(E4M3Bits(FromBF16Bits(qr[d]))))
					} else {
						qf = float64(FromBF16Bits(qr[d]))
					}
					var kf float64
					if a.KVFP8 {
						kf = float64(FromE4M3Bits(c.KPEFP8[rOff+d])) * float64(kpes)
					} else {
						kf = float64(FromBF16Bits(c.KPEBF16[rOff+d]))
					}
					dotPe += qf * kf
				}
				s := (dotNope + dotPe) * float64(a.Scale)
				logits[t] = s
				if s > mx {
					mx = s
				}
			}

			var sum float64
			for t := range seqlen {
				logits[t] = math.Exp(logits[t] - mx)
				sum += logits[t]
			}

			oSlice := out[qnOff : qnOff+a.LatentDim]
			if sum > 0 {
				for t := range seqlen {
					w := logits[t] / sum
					page := int(c.PageTable[b*a.MaxPages+t/a.PageSize])
					slot := page*a.PageSize + (t % a.PageSize)
					cOff := slot * a.LatentDim
					for d := range a.LatentDim {
						var cf float64
						if a.KVFP8 {
							cf = float64(FromE4M3Bits(c.CKVFP8[cOff+d])) * float64(cs)
						} else {
							cf = float64(FromBF16Bits(c.CKVBF16[cOff+d]))
						}
						oSlice[d] += w * cf
					}
				}
			}
			for d := range a.LatentDim {
				oSlice[d] = float64(FromBF16Bits(BF16Bits(float32(oSlice[d]))))
			}
		}
	}
	return out, nil
}
