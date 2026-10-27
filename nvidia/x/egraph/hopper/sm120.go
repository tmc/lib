package hopper

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// sm120Prelude provides workstation/edge Blackwell (sm_120a / sm_121a / sm_120f)
// warp-synchronous m16n8k64 NVFP4 block-scaled MMA, E2M1x2 dequantization, and
// clusterlaunchcontrol (try_cancel / query_cancel) helpers with host/device
// emulation fallbacks.
const sm120Prelude = `#ifndef EG_SM120_PRELUDE
#define EG_SM120_PRELUDE

EG_DEV float eg_sm120_decode_e2m1(uint32_t code) {
    uint32_t h[2];
    eg_e2m1x4_to_bf16x4(code & 0xfu, h);
    return eg_bf16_to_f32(h[0]);
}

// eg_sm120_dequant_nvfp4_pair widens the two E2M1 nibbles in packed_byte and
// scales them by the UE4M3 block scale byte scale_e4m3.
EG_DEV void eg_sm120_dequant_nvfp4_pair(uint32_t packed_byte, uint32_t scale_e4m3, float out[2]) {
    const float sf = eg_e4m3_to_f32(scale_e4m3 & 0xffu);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200) && defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint32_t f16x2;
    asm("{\n\t"
        ".reg .b8 q;\n\t"
        "cvt.u8.u32 q, %1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, q;\n\t"
        "}" : "=r"(f16x2) : "r"(packed_byte));
    const uint32_t lo = f16x2 & 0xffffu;
    const uint32_t hi = f16x2 >> 16;
    const uint32_t blo = (lo & 0x7fffu) ? (((lo & 0x7fffu) >> 3) + 0x3800u) | (lo & 0x8000u) : (lo & 0x8000u);
    const uint32_t bhi = (hi & 0x7fffu) ? (((hi & 0x7fffu) >> 3) + 0x3800u) | (hi & 0x8000u) : (hi & 0x8000u);
    out[0] = eg_bf16_to_f32(blo) * sf;
    out[1] = eg_bf16_to_f32(bhi) * sf;
#else
    uint32_t h[2];
    eg_e2m1x4_to_bf16x4(packed_byte & 0xffu, h);
    out[0] = eg_bf16_to_f32(h[0]) * sf;
    out[1] = eg_bf16_hi_to_f32(h[0]) * sf;
#endif
}

// eg_sm120_mma_nvfp4_16864 executes workstation/edge Blackwell (sm_120a / sm_121a)
// warp-synchronous m16n8k64 NVFP4 block-scaled tensor-core accumulation into
// warp registers d[0..3].
EG_DEV void eg_sm120_mma_nvfp4_16864(
    float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1200) && defined(__CUDA_ARCH_FEAT_SM120_ALL) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
#elif defined(EG_HOST_EMU)
    eg_emu_mma_nvfp4_16864(d, a, b, sfa, sfb);
#else
    const int lane = (int)(threadIdx.x & 31u);
    const int g = lane / 4, t = lane % 4;
    float out[4] = {d[0], d[1], d[2], d[3]};
    for (int i = 0; i < 4; ++i) {
        const int row = g + 8 * (i / 2);
        const int col = 2 * t + (i & 1);
        const int a_slane = (row % 8) * 4 + (row / 8);
        const int b_slane = col * 4;
        const uint32_t row_sfa = __shfl_sync(0xffffffffu, sfa, a_slane);
        const uint32_t col_sfb = __shfl_sync(0xffffffffu, sfb, b_slane);
        float acc = out[i];
        for (int k = 0; k < 64; ++k) {
            const int alane = (row % 8) * 4 + (k % 32) / 8;
            const int areg = (row / 8) + 2 * (k / 32);
            const uint32_t aword = (areg == 0) ? a[0] : (areg == 1) ? a[1] : (areg == 2) ? a[2] : a[3];
            const uint32_t got_a = __shfl_sync(0xffffffffu, aword, alane);
            const float av = eg_sm120_decode_e2m1((got_a >> (4 * (k % 8))) & 0xfu) *
                             eg_e4m3_to_f32((row_sfa >> (8 * (k / 16))) & 0xffu);

            const int blane = col * 4 + (k % 32) / 8;
            const uint32_t bword = (k < 32) ? b[0] : b[1];
            const uint32_t got_b = __shfl_sync(0xffffffffu, bword, blane);
            const float bv = eg_sm120_decode_e2m1((got_b >> (4 * (k % 8))) & 0xfu) *
                             eg_e4m3_to_f32((col_sfb >> (8 * (k / 16))) & 0xffu);
            acc += av * bv;
        }
        out[i] = acc;
    }
    for (int i = 0; i < 4; ++i) d[i] = out[i];
#endif
}

EG_DEV void eg_sm120_clc_mbar_init(uint64_t* mbar, uint32_t count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
#elif defined(EG_HOST_EMU) || defined(EG_EMU)
    eg_emu_mbar_init(mbar, (int)count);
#else
    (void)mbar;
    (void)count;
#endif
}

// eg_clc_try_cancel issues a Blackwell clusterlaunchcontrol.try_cancel request
// to steal an unlaunched CTA tile into 16-byte shared-memory response_smem.
EG_DEV void eg_clc_try_cancel(uint4* response_smem, uint64_t* mbar) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    const uint32_t resp_addr = (uint32_t)__cvta_generic_to_shared(response_smem);
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], 16;\n\t"
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];"
        :: "r"(resp_addr), "r"(mbar_addr) : "memory");
#elif defined(EG_EMU)
    eg_emu_clc_try_cancel_async(response_smem, mbar);
#else
    if (response_smem) *response_smem = {0u, 0u, 0u, 0u};
    (void)mbar;
#endif
}

EG_DEV void eg_sm120_clc_wait(uint64_t* mbar, uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "EG_SM120_CLC_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra EG_SM120_CLC_WAIT_%=;\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase) : "memory");
#elif defined(EG_EMU)
    eg_emu_clc_wait(mbar, (int)phase);
#else
    (void)mbar;
    (void)phase;
#endif
}

EG_DEV bool eg_sm120_clc_query_is_canceled(uint4 resp) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint32_t pred;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b128 r128;\n\t"
        "mov.b128 r128, {%1, %2, %3, %4};\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r128;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(pred)
        : "r"(resp.x), "r"(resp.y), "r"(resp.z), "r"(resp.w));
    return pred != 0;
#elif defined(EG_EMU)
    return eg_emu_clc_query_is_canceled(resp);
#else
    return resp.w != 0u;
#endif
}

EG_DEV uint32_t eg_sm120_clc_query_first_ctaid_x(uint4 resp) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint32_t ctaid_x;
    asm volatile(
        "{\n\t"
        ".reg .b128 r128;\n\t"
        "mov.b128 r128, {%1, %2, %3, %4};\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, r128;\n\t"
        "}"
        : "=r"(ctaid_x)
        : "r"(resp.x), "r"(resp.y), "r"(resp.z), "r"(resp.w));
    return ctaid_x;
#elif defined(EG_EMU)
    return eg_emu_clc_query_first_ctaid_x(resp);
#else
    return resp.x;
#endif
}

#endif // EG_SM120_PRELUDE
`

func validSM120Target(t string) bool {
	switch t {
	case "sm_120a", "sm_121a", "sm_120f":
		return true
	}
	return false
}

// SM120BlockScaledGEMM configures a workstation/edge Blackwell (sm_120a / sm_121a / sm_120f)
// NVFP4 block-scaled GEMM schedule that accumulates warp-synchronous
// mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3
// tiles directly in warp registers (without TMEM or tcgen05) and optionally steals
// output tiles dynamically across persistent CTAs via clusterlaunchcontrol.try_cancel.
type SM120BlockScaledGEMM struct {
	Name       string
	M, N, K    int
	BM, BN, BK int
	Target     string       // "sm_120a", "sm_121a", or "sm_120f" (defaults to "sm_121a")
	UseCLC     bool         // dynamic work-stealing via clusterlaunchcontrol.try_cancel / query_cancel
	Epilogue   TileEpilogue // TileLinear, TileSiLU, or TileGLU
	Out        DType        // F32 or BF16
}

func (g SM120BlockScaledGEMM) norm() (SM120BlockScaledGEMM, error) {
	if g.Name == "" {
		g.Name = "sm120_block_scaled_gemm"
	}
	if !cIdent(g.Name) {
		return g, fmt.Errorf("sm120 block scaled gemm: bad name %q", g.Name)
	}
	if g.M == 0 {
		g.M = 64
	}
	if g.N == 0 {
		g.N = 64
	}
	if g.K == 0 {
		g.K = 64
	}
	if g.BM == 0 {
		g.BM = 64
		if g.M > 0 && g.M < g.BM {
			g.BM = g.M
		}
	}
	if g.BN == 0 {
		g.BN = 64
		if g.N > 0 && g.N < g.BN {
			g.BN = g.N
		}
	}
	if g.BK == 0 {
		g.BK = 64
	}
	if g.Target == "" {
		g.Target = "sm_121a"
	}
	if !validSM120Target(g.Target) {
		return g, fmt.Errorf("sm120 block scaled gemm %s: unsupported target %q (want sm_120a, sm_121a, or sm_120f)", g.Name, g.Target)
	}
	if g.Out == 0 {
		g.Out = F32
	}
	if g.Out != F32 && g.Out != BF16 {
		return g, fmt.Errorf("sm120 block scaled gemm %s: out must be F32 or BF16, got %v", g.Name, g.Out)
	}
	if g.Epilogue != TileLinear && g.Epilogue != TileSiLU && g.Epilogue != TileGLU {
		return g, fmt.Errorf("sm120 block scaled gemm %s: unsupported epilogue %v", g.Name, g.Epilogue)
	}
	if g.M < 16 || g.M%16 != 0 {
		return g, fmt.Errorf("sm120 block scaled gemm %s: M=%d must be a positive multiple of 16", g.Name, g.M)
	}
	if g.N < 16 || g.N%16 != 0 {
		return g, fmt.Errorf("sm120 block scaled gemm %s: N=%d must be a positive multiple of 16", g.Name, g.N)
	}
	if g.K < 64 || g.K%64 != 0 {
		return g, fmt.Errorf("sm120 block scaled gemm %s: K=%d must be a positive multiple of 64", g.Name, g.K)
	}
	if g.BM < 16 || g.BM%16 != 0 || g.BN < 16 || g.BN%16 != 0 || g.BK < 64 || g.BK%64 != 0 {
		return g, fmt.Errorf("sm120 block scaled gemm %s: tile (%d, %d, %d) must be multiples of (16, 16, 64)", g.Name, g.BM, g.BN, g.BK)
	}
	if g.K%g.BK != 0 {
		return g, fmt.Errorf("sm120 block scaled gemm %s: K=%d must be divisible by BK=%d", g.Name, g.K, g.BK)
	}
	return g, nil
}

// TotalTiles returns the total number of (BM x BN) output tiles for M rows.
func (g SM120BlockScaledGEMM) TotalTiles(M int) int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	if M <= 0 {
		M = g.M
	}
	numM := (M + g.BM - 1) / g.BM
	numN := (g.N + g.BN - 1) / g.BN
	return numM * numN
}

// SharedBytes returns the shared memory size in bytes per CTA.
func (g SM120BlockScaledGEMM) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	stageBytes := (g.BM+g.BN)*(g.BK/2) + (g.BM+g.BN)*(g.BK/16)
	accBytes := g.BM * g.BN * 4
	return stageBytes + accBytes + 64
}

// Grid returns the launch grid dimensions for M activation rows.
func (g SM120BlockScaledGEMM) Grid(M int) [3]int {
	g, err := g.norm()
	if err != nil {
		return [3]int{}
	}
	if M <= 0 {
		M = g.M
	}
	numM := (M + g.BM - 1) / g.BM
	numN := (g.N + g.BN - 1) / g.BN
	if g.UseCLC {
		return [3]int{numM * numN, 1, 1}
	}
	return [3]int{numN, numM, 1}
}

// Block returns the CTA thread block dimensions (4 warps = 128 threads).
func (g SM120BlockScaledGEMM) Block() [3]int {
	return [3]int{128, 1, 1}
}

func (g SM120BlockScaledGEMM) params() []string {
	return []string{
		"const uint8_t* __restrict__ a_packed",
		"const uint8_t* __restrict__ a_scales",
		"const uint8_t* __restrict__ b_packed",
		"const uint8_t* __restrict__ b_scales",
		g.Out.cType() + "* __restrict__ out",
		fmt.Sprintf("int m_total = %d", g.M),
	}
}

// CUDA returns the self-contained CUDA C++ kernel source for g.
func (g SM120BlockScaledGEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(Prelude)
	b.WriteString("\n")
	b.WriteString(sm120Prelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: %s warp-scaled m16n8k64 NVFP4 GEMM (M=%d, N=%d, K=%d, tile=%dx%dx%d, clc=%v, epilogue=%s, out=%s).\n",
		g.Name, g.Target, g.M, g.N, g.K, g.BM, g.BN, g.BK, g.UseCLC, g.Epilogue, g.Out)
	fmt.Fprintf(&b, "__global__ void %s(\n    %s) {\n",
		g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = %d, KT = K / BK;\n",
		g.N, g.K, g.BM, g.BN, g.BK)
	if g.Epilogue == TileGLU {
		b.WriteString("    constexpr int NOUT = N / 2;\n")
		b.WriteString("    constexpr int HALF_BN = (BN / 2 < NOUT) ? (BN / 2) : NOUT;\n")
	} else {
		b.WriteString("    constexpr int NOUT = N;\n")
	}
	b.WriteString(`    constexpr int K_FP4 = K / 2, K_SF = K / 16;
    constexpr int BK_FP4 = BK / 2, BK_SF = BK / 16;
    constexpr int NUM_N = (N + BN - 1) / BN;
    constexpr int M_TILES = BM / 16, N_TILES = BN / 8;
    constexpr int SUBTILES = M_TILES * N_TILES;
    constexpr int SUB_PER_WARP = (SUBTILES + 3) / 4;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane / 4, t = lane % 4;
    const int num_m = (m_total + BM - 1) / BM;
    const int total_tiles = num_m * NUM_N;

    alignas(16) __shared__ uint8_t sa_fp4[BM * BK_FP4];
    alignas(16) __shared__ uint8_t sa_sf[BM * BK_SF];
    alignas(16) __shared__ uint8_t sb_fp4[BN * BK_FP4];
    alignas(16) __shared__ uint8_t sb_sf[BN * BK_SF];
    alignas(16) __shared__ float s_acc[BM * BN];
    __shared__ uint4 clc_response;
    __shared__ uint64_t clc_mbar;
    __shared__ int clc_next_tile;
`)
	if g.UseCLC {
		b.WriteString(`    if (tid == 0) {
        eg_sm120_clc_mbar_init(&clc_mbar, 1u);
    }
    __syncthreads();

    uint32_t clc_phase = 0u;
    for (int tile_idx = (int)(blockIdx.y * gridDim.x + blockIdx.x); tile_idx < total_tiles; ) {
        if (tid == 0) {
            eg_clc_try_cancel(&clc_response, &clc_mbar);
        }
`)
	} else {
		b.WriteString(`    const int grid_stride = (int)(gridDim.x * gridDim.y);
    for (int tile_idx = (int)(blockIdx.y * gridDim.x + blockIdx.x); tile_idx < total_tiles; tile_idx += grid_stride) {
`)
	}

	b.WriteString(`        const int tile_m = tile_idx / NUM_N;
        const int tile_n = tile_idx % NUM_N;
        const int bm0 = tile_m * BM;
        const int bn0 = tile_n * BN;
`)
	if g.Epilogue == TileGLU {
		b.WriteString(`        auto wrow = [&](int j) -> int {
            if (j < BN / 2) {
                const int c = tile_n * HALF_BN + j;
                return (j < HALF_BN && c < NOUT) ? c : -1;
            }
            const int j_up = j - (BN / 2);
            const int c = tile_n * HALF_BN + j_up;
            return (j_up < HALF_BN && c < NOUT) ? (NOUT + c) : -1;
        };
`)
	} else {
		b.WriteString(`        auto wrow = [&](int j) -> int {
            const int gn = bn0 + j;
            return (gn < N) ? gn : -1;
        };
`)
	}

	b.WriteString(`        float acc[SUB_PER_WARP][4] = {};
        for (int kt = 0; kt < KT; ++kt) {
            __syncthreads();
            for (int i = tid; i < BM * BK_FP4; i += 128) {
                const int r = i / BK_FP4, k = i % BK_FP4;
                const int gm = bm0 + r;
                sa_fp4[i] = (gm < m_total) ? a_packed[(size_t)gm * K_FP4 + kt * BK_FP4 + k] : (uint8_t)0u;
            }
            for (int i = tid; i < BM * BK_SF; i += 128) {
                const int r = i / BK_SF, k = i % BK_SF;
                const int gm = bm0 + r;
                sa_sf[i] = (gm < m_total) ? a_scales[(size_t)gm * K_SF + kt * BK_SF + k] : (uint8_t)0u;
            }
            for (int i = tid; i < BN * BK_FP4; i += 128) {
                const int r = i / BK_FP4, k = i % BK_FP4;
                const int gn = wrow(r);
                sb_fp4[i] = (gn >= 0) ? b_packed[(size_t)gn * K_FP4 + kt * BK_FP4 + k] : (uint8_t)0u;
            }
            for (int i = tid; i < BN * BK_SF; i += 128) {
                const int r = i / BK_SF, k = i % BK_SF;
                const int gn = wrow(r);
                sb_sf[i] = (gn >= 0) ? b_scales[(size_t)gn * K_SF + kt * BK_SF + k] : (uint8_t)0u;
            }
            __syncthreads();

#pragma unroll
            for (int ks = 0; ks < BK / 64; ++ks) {
#pragma unroll
                for (int s = 0; s < SUB_PER_WARP; ++s) {
                    const int sub = warp + s * 4;
                    if (sub < SUBTILES) {
                        const int mt = sub / N_TILES;
                        const int nt = sub % N_TILES;
                        const int m0 = mt * 16;
                        const int n0 = nt * 8;
                        uint32_t a_frag[4];
                        uint32_t b_frag[2];
                        memcpy(&a_frag[0], &sa_fp4[(m0 + g) * BK_FP4 + ks * 32 + t * 4], 4);
                        memcpy(&a_frag[1], &sa_fp4[(m0 + g + 8) * BK_FP4 + ks * 32 + t * 4], 4);
                        memcpy(&a_frag[2], &sa_fp4[(m0 + g) * BK_FP4 + ks * 32 + 16 + t * 4], 4);
                        memcpy(&a_frag[3], &sa_fp4[(m0 + g + 8) * BK_FP4 + ks * 32 + 16 + t * 4], 4);
                        memcpy(&b_frag[0], &sb_fp4[(n0 + g) * BK_FP4 + ks * 32 + t * 4], 4);
                        memcpy(&b_frag[1], &sb_fp4[(n0 + g) * BK_FP4 + ks * 32 + 16 + t * 4], 4);

                        const int a_sf_row = m0 + g + 8 * (t & 1);
                        uint32_t sfa = 0u, sfb = 0u;
                        memcpy(&sfa, &sa_sf[a_sf_row * BK_SF + ks * 4], 4);
                        memcpy(&sfb, &sb_sf[(n0 + g) * BK_SF + ks * 4], 4);

                        eg_sm120_mma_nvfp4_16864(acc[s], a_frag, b_frag, sfa, sfb);
                    }
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < SUB_PER_WARP; ++s) {
            const int sub = warp + s * 4;
            if (sub < SUBTILES) {
                const int mt = sub / N_TILES;
                const int nt = sub % N_TILES;
                for (int i = 0; i < 4; ++i) {
                    const int r = mt * 16 + g + 8 * (i / 2);
                    const int c = nt * 8 + 2 * t + (i & 1);
                    s_acc[r * BN + c] = acc[s][i];
                }
            }
        }
        __syncthreads();
`)

	switch g.Epilogue {
	case TileGLU:
		b.WriteString(`        for (int idx = tid; idx < BM * HALF_BN; idx += 128) {
            const int r = idx / HALF_BN, c = idx % HALF_BN;
            const int gm = bm0 + r;
            const int gn = tile_n * HALF_BN + c;
            if (gm < m_total && gn < NOUT) {
                const float gate = s_acc[r * BN + c];
                const float up = s_acc[r * BN + (BN / 2) + c];
                const float y = eg_silu(gate) * up;
`)
	case TileSiLU:
		b.WriteString(`        for (int idx = tid; idx < BM * BN; idx += 128) {
            const int r = idx / BN, c = idx % BN;
            const int gm = bm0 + r;
            const int gn = bn0 + c;
            if (gm < m_total && gn < NOUT) {
                const float y = eg_silu(s_acc[r * BN + c]);
`)
	default:
		b.WriteString(`        for (int idx = tid; idx < BM * BN; idx += 128) {
            const int r = idx / BN, c = idx % BN;
            const int gm = bm0 + r;
            const int gn = bn0 + c;
            if (gm < m_total && gn < NOUT) {
                const float y = s_acc[r * BN + c];
`)
	}
	if g.Out == BF16 {
		b.WriteString("                out[(size_t)gm * NOUT + gn] = eg_f32_to_bf16(y);\n")
	} else {
		b.WriteString("                out[(size_t)gm * NOUT + gn] = y;\n")
	}
	b.WriteString("            }\n        }\n")

	if g.UseCLC {
		b.WriteString(`        if (tid == 0) {
            eg_sm120_clc_wait(&clc_mbar, clc_phase);
            clc_phase ^= 1u;
            if (eg_sm120_clc_query_is_canceled(clc_response)) {
                clc_next_tile = (int)eg_sm120_clc_query_first_ctaid_x(clc_response);
            } else {
                clc_next_tile = total_tiles;
            }
        }
        __syncthreads();
        tile_idx = clc_next_tile;
    }
}
`)
	} else {
		b.WriteString(`    }
}
`)
	}
	return b.String(), nil
}

// Launcher returns a C++ host launch wrapper for g.
func (g SM120BlockScaledGEMM) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	block := g.Block()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(\n    const uint8_t* a_packed,\n    const uint8_t* a_scales,\n    const uint8_t* b_packed,\n    const uint8_t* b_scales,\n    %s* out,\n    int m_total,\n    cudaStream_t stream = nullptr) {\n",
		g.Name, g.Out.cType())
	if g.UseCLC {
		fmt.Fprintf(&b, "    const int num_m = (m_total + %d) / %d;\n", g.BM-1, g.BM)
		fmt.Fprintf(&b, "    const int num_n = (%d + %d) / %d;\n", g.N, g.BN-1, g.BN)
		fmt.Fprintf(&b, "    const dim3 grid(num_m * num_n, 1, 1), block(%d, %d, %d);\n", block[0], block[1], block[2])
	} else {
		fmt.Fprintf(&b, "    const int num_m = (m_total + %d) / %d;\n", g.BM-1, g.BM)
		fmt.Fprintf(&b, "    const int num_n = (%d + %d) / %d;\n", g.N, g.BN-1, g.BN)
		fmt.Fprintf(&b, "    const dim3 grid(num_n, num_m, 1), block(%d, %d, %d);\n", block[0], block[1], block[2])
	}
	fmt.Fprintf(&b, "    if (m_total > 0) EG_LAUNCH(%s, grid, block, 0, stream, a_packed, a_scales, b_packed, b_scales, out, m_total);\n}\n", g.Name)
	return b.String(), nil
}

// Source returns the self-contained CUDA C++ translation unit (Prelude, kernel,
// and host launcher) for g.
func (g SM120BlockScaledGEMM) Source() (string, error) {
	cuda, err := g.CUDA()
	if err != nil {
		return "", err
	}
	launcher, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return cuda + "\n" + launcher, nil
}

// Reference computes the host reference output for g over block-scaled NVFP4
// inputs aPacked [M, K/2], aScales [M, K/16], bPacked [N, K/2], and
// bScales [N, K/16].
func (g SM120BlockScaledGEMM) Reference(aPacked, aScales, bPacked, bScales []uint8, M int) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if M <= 0 {
		M = g.M
	}
	kFP4, kSF := g.K/2, g.K/16
	if len(aPacked) != M*kFP4 || len(aScales) != M*kSF || len(bPacked) != g.N*kFP4 || len(bScales) != g.N*kSF {
		return nil, fmt.Errorf("sm120 block scaled gemm %s reference: invalid input lengths (got aPacked=%d aScales=%d bPacked=%d bScales=%d for M=%d N=%d K=%d)",
			g.Name, len(aPacked), len(aScales), len(bPacked), len(bScales), M, g.N, g.K)
	}
	nout := g.N
	if g.Epilogue == TileGLU {
		nout = g.N / 2
	}
	dots := make([]float32, M*g.N)
	for r := range M {
		for n := range g.N {
			var acc float32
			for k := range g.K {
				ab := aPacked[r*kFP4+k/2]
				bb := bPacked[n*kFP4+k/2]
				ac := (ab >> (4 * (k & 1))) & 0xf
				bc := (bb >> (4 * (k & 1))) & 0xf
				asf := FromE4M3Bits(aScales[r*kSF+k/16])
				bsf := FromE4M3Bits(bScales[n*kSF+k/16])
				av := FromE2M1(ac) * asf
				bv := FromE2M1(bc) * bsf
				acc += av * bv
			}
			dots[r*g.N+n] = acc
		}
	}
	out := make([]float32, M*nout)
	for r := range M {
		for c := range nout {
			var y float32
			switch g.Epilogue {
			case TileGLU:
				gate := dots[r*g.N+c]
				up := dots[r*g.N+nout+c]
				silu := gate / (1 + float32(math.Exp(float64(-gate))))
				y = silu * up
			case TileSiLU:
				v := dots[r*g.N+c]
				y = v / (1 + float32(math.Exp(float64(-v))))
			default:
				y = dots[r*g.N+c]
			}
			out[r*nout+c] = g.Out.Round(y)
		}
	}
	return out, nil
}

// SM120PagedAttention configures a workstation/edge Blackwell (sm_120a / sm_121a)
// grouped-query paged attention schedule using register-accumulated online
// softmax (exp2f rescaling in registers, no TMEM required) over a paged KV
// cache table with either BF16 or block-scaled NVFP4 (with per-16-element UE4M3
// scales) KV storage.
type SM120PagedAttention struct {
	Name                  string
	NumQHeads, NumKVHeads int
	HeadDim               int
	PageSize, MaxPages    int
	SeqLen                int
	Scale                 float32
	Target                string       // "sm_120a" or "sm_121a" (defaults to "sm_121a")
	KVFormat              WeightFormat // WeightBF16 or WeightNVFP4

	// Splits, when > 0, divides each sequence's keys among Splits CTAs per
	// KV head (split-KV, as in flash-decoding), so the grid is (batch,
	// NumKVHeads, Splits). With seq_len keys, each split takes
	// max(MinSplitKeys, ⌈seq_len/Splits⌉ rounded up to 16) keys, so a short
	// sequence uses fewer splits and the rest write empty partials. The
	// kernel writes each split's unnormalized output, running maximum and
	// sum to ws ([SM120PagedAttention.WorkspaceFloats]) in place of out, and
	// the Name_combine kernel ([SM120PagedAttention.CombineGrid]) rescales
	// the partials to their common maximum and writes out. Splits reads the
	// cache in 16-byte vectors and resolves each key's page once per tile;
	// it needs WeightBF16.
	Splits       int
	MinSplitKeys int // at least this many keys per split: a multiple of 16; 0 means 64
}

func (a SM120PagedAttention) norm() (SM120PagedAttention, error) {
	if a.Name == "" {
		a.Name = "sm120_paged_attention"
	}
	if !cIdent(a.Name) {
		return a, fmt.Errorf("sm120 paged attention: bad name %q", a.Name)
	}
	if a.NumQHeads == 0 {
		a.NumQHeads = 8
	}
	if a.NumKVHeads == 0 {
		a.NumKVHeads = 2
	}
	if a.HeadDim == 0 {
		a.HeadDim = 64
	}
	if a.PageSize == 0 {
		a.PageSize = 16
	}
	if a.SeqLen == 0 {
		a.SeqLen = 64
	}
	if a.MaxPages == 0 {
		a.MaxPages = max((a.SeqLen+a.PageSize-1)/a.PageSize, 1)
	}
	if a.Scale == 0 {
		a.Scale = 1 / float32(math.Sqrt(float64(a.HeadDim)))
	}
	if a.Target == "" {
		a.Target = "sm_121a"
	}
	if !validSM120Target(a.Target) {
		return a, fmt.Errorf("sm120 paged attention %s: unsupported target %q (want sm_120a, sm_121a, or sm_120f)", a.Name, a.Target)
	}
	if a.KVFormat == 0 {
		a.KVFormat = WeightBF16
	}
	if a.KVFormat != WeightBF16 && a.KVFormat != WeightNVFP4 {
		return a, fmt.Errorf("sm120 paged attention %s: unsupported KVFormat %v (want WeightBF16 or WeightNVFP4)", a.Name, a.KVFormat)
	}
	if a.NumQHeads < 1 || a.NumKVHeads < 1 || a.NumQHeads%a.NumKVHeads != 0 {
		return a, fmt.Errorf("sm120 paged attention %s: bad heads %d/%d", a.Name, a.NumQHeads, a.NumKVHeads)
	}
	if a.HeadDim < 16 || a.HeadDim > 256 || a.HeadDim%16 != 0 {
		return a, fmt.Errorf("sm120 paged attention %s: HeadDim=%d must be a multiple of 16 in 16..256", a.Name, a.HeadDim)
	}
	if a.PageSize < 1 || a.MaxPages < 1 || a.SeqLen < 1 {
		return a, fmt.Errorf("sm120 paged attention %s: PageSize=%d MaxPages=%d SeqLen=%d must be positive", a.Name, a.PageSize, a.MaxPages, a.SeqLen)
	}
	if a.Splits < 0 {
		return a, fmt.Errorf("sm120 paged attention %s: negative Splits %d", a.Name, a.Splits)
	}
	if a.Splits > 0 {
		if a.KVFormat != WeightBF16 {
			return a, fmt.Errorf("sm120 paged attention %s: Splits needs WeightBF16 KV, have %v", a.Name, a.KVFormat)
		}
		if a.MinSplitKeys == 0 {
			a.MinSplitKeys = 64
		}
		if a.MinSplitKeys < 16 || a.MinSplitKeys%16 != 0 {
			return a, fmt.Errorf("sm120 paged attention %s: MinSplitKeys=%d must be a positive multiple of 16", a.Name, a.MinSplitKeys)
		}
	}
	return a, nil
}

// WorkspaceFloats returns the floats of workspace ws a launch over batch
// sequences needs with Splits > 0, or 0 without splits: D + 2 per
// (sequence, query head, split).
func (a SM120PagedAttention) WorkspaceFloats(batch int) int {
	a, err := a.norm()
	if err != nil || a.Splits == 0 {
		return 0
	}
	return batch * a.NumQHeads * a.Splits * (a.HeadDim + 2)
}

// CombineGrid returns the grid and block of the Name_combine kernel for
// batch sequences: one CTA per (sequence, query head).
func (a SM120PagedAttention) CombineGrid(batch int) (grid, block [3]int) {
	return [3]int{batch * a.NumQHeads, 1, 1}, [3]int{128, 1, 1}
}

// SharedBytes returns the shared memory size in bytes per CTA.
func (a SM120PagedAttention) SharedBytes() int {
	a, err := a.norm()
	if err != nil {
		return 0
	}
	const tileK = 16
	return 2 * tileK * a.HeadDim * 4
}

// Grid returns the launch grid dimensions for batch sequences.
func (a SM120PagedAttention) Grid(batch int) [3]int {
	a, err := a.norm()
	if err != nil {
		return [3]int{}
	}
	if batch <= 0 {
		batch = 1
	}
	return [3]int{batch, a.NumKVHeads, max(a.Splits, 1)}
}

// Block returns the CTA thread block dimensions (4 warps = 128 threads).
func (a SM120PagedAttention) Block() [3]int {
	return [3]int{128, 1, 1}
}

func (a SM120PagedAttention) params() []string {
	out := "float* __restrict__ out"
	if a.Splits > 0 {
		out = "float* __restrict__ ws"
	}
	return []string{
		"const float* __restrict__ q",
		"const uint8_t* __restrict__ k_pages",
		"const uint8_t* __restrict__ v_pages",
		"const uint8_t* __restrict__ k_scales",
		"const uint8_t* __restrict__ v_scales",
		"const int32_t* __restrict__ page_table",
		out,
		fmt.Sprintf("int seq_len = %d", a.SeqLen),
	}
}

// CUDA returns the self-contained CUDA C++ kernel source for a.
func (a SM120PagedAttention) CUDA() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	g := a.NumQHeads / a.NumKVHeads
	var b strings.Builder
	b.WriteString(Prelude)
	b.WriteString("\n")
	b.WriteString(sm120Prelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: %s register-accumulated online softmax paged attention (%d/%d heads, D=%d, page=%d, max_pages=%d, kv=%s).\n",
		a.Name, a.Target, a.NumQHeads, a.NumKVHeads, a.HeadDim, a.PageSize, a.MaxPages, a.KVFormat)
	fmt.Fprintf(&b, "__global__ void %s(\n    %s) {\n",
		a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, D = %d;\n",
		a.NumQHeads, a.NumKVHeads, g, a.HeadDim)
	fmt.Fprintf(&b, "    constexpr int PAGE_SIZE = %d, MAX_PAGES = %d, TILE_K = 16;\n",
		a.PageSize, a.MaxPages)
	fmt.Fprintf(&b, "    constexpr int RPT = (G + 3) / 4, DPT = (D + 31) / 32;\n")
	fmt.Fprintf(&b, "    constexpr float SM_SCALE = %s;\n", floatLiteral(a.Scale))
	b.WriteString(`    const float scale_log2 = SM_SCALE * 1.4426950408889634f;
    const int b = (int)blockIdx.x;
    const int kvh = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    alignas(16) __shared__ float sk[TILE_K * D];
    alignas(16) __shared__ float sv[TILE_K * D];

    float q_reg[RPT][DPT] = {};
    float o_reg[RPT][DPT] = {};
    float m_reg[RPT];
    float l_reg[RPT];
#pragma unroll
    for (int rp = 0; rp < RPT; ++rp) {
        m_reg[rp] = NEG_INF;
        l_reg[rp] = 0.0f;
        const int r = warp + rp * 4;
        if (r < G) {
            const int qh = kvh * G + r;
            const size_t q_base = ((size_t)b * HQ + qh) * D;
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) {
                const int d = lane + dp * 32;
                if (d < D) {
                    q_reg[rp][dp] = q[q_base + d];
                }
            }
        }
    }

`)
	const loop = `    for (int k0 = 0; k0 < seq_len; k0 += TILE_K) {
        __syncthreads();
`
	switch {
	case a.Splits > 0:
		fmt.Fprintf(&b, "    constexpr int SPLITS = %d, MIN_SPLIT_KEYS = %d;\n", a.Splits, a.MinSplitKeys)
		b.WriteString(sm120SplitTile)
	case a.KVFormat == WeightNVFP4:
		b.WriteString(loop)
		b.WriteString(`        constexpr int D_FP4 = D / 2, D_SF = D / 16;
        for (int idx = tid; idx < TILE_K * D_FP4; idx += 128) {
            const int tk = idx / D_FP4, byte_idx = idx % D_FP4;
            const int d0 = byte_idx * 2;
            const int k_pos = k0 + tk;
            float k_pair[2] = {0.0f, 0.0f};
            float v_pair[2] = {0.0f, 0.0f};
            if (k_pos < seq_len) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE_SIZE];
                const int slot = page * PAGE_SIZE + (k_pos % PAGE_SIZE);
                const size_t row = (size_t)slot * HKV + kvh;
                const uint8_t kb = k_pages[row * D_FP4 + byte_idx];
                const uint8_t vb = v_pages[row * D_FP4 + byte_idx];
                const uint8_t ks = k_scales[row * D_SF + d0 / 16];
                const uint8_t vs = v_scales[row * D_SF + d0 / 16];
                eg_sm120_dequant_nvfp4_pair(kb, ks, k_pair);
                eg_sm120_dequant_nvfp4_pair(vb, vs, v_pair);
            }
            sk[tk * D + d0] = k_pair[0];
            sk[tk * D + d0 + 1] = k_pair[1];
            sv[tk * D + d0] = v_pair[0];
            sv[tk * D + d0 + 1] = v_pair[1];
        }
`)
	default:
		b.WriteString(loop)
		b.WriteString(`        const uint16_t* const k_bf16 = reinterpret_cast<const uint16_t*>(k_pages);
        const uint16_t* const v_bf16 = reinterpret_cast<const uint16_t*>(v_pages);
        for (int idx = tid; idx < TILE_K * D; idx += 128) {
            const int tk = idx / D, d = idx % D;
            const int k_pos = k0 + tk;
            float kv_k = 0.0f, kv_v = 0.0f;
            if (k_pos < seq_len) {
                const int page = page_table[b * MAX_PAGES + k_pos / PAGE_SIZE];
                const int slot = page * PAGE_SIZE + (k_pos % PAGE_SIZE);
                const size_t kv_off = ((size_t)slot * HKV + kvh) * D + d;
                kv_k = eg_bf16_to_f32(k_bf16[kv_off]);
                kv_v = eg_bf16_to_f32(v_bf16[kv_off]);
            }
            sk[tk * D + d] = kv_k;
            sv[tk * D + d] = kv_v;
        }
`)
	}

	score := `        __syncthreads();
        const int tk_end = min(TILE_K, seq_len - k0);
#pragma unroll
        for (int rp = 0; rp < RPT; ++rp) {
            const int r = warp + rp * 4;
            if (r < G) {
                for (int tk = 0; tk < tk_end; ++tk) {
                    float part = 0.0f;
#pragma unroll
                    for (int dp = 0; dp < DPT; ++dp) {
                        const int d = lane + dp * 32;
                        if (d < D) {
                            part += q_reg[rp][dp] * sk[tk * D + d];
                        }
                    }
                    const float score_log2 = eg_warp_sum(part) * scale_log2;
                    const float m_prev = m_reg[rp];
                    const float m_new = fmaxf(m_prev, score_log2);
                    const float corr = (m_prev == NEG_INF) ? 0.0f : exp2f(m_prev - m_new);
                    const float p = exp2f(score_log2 - m_new);
                    m_reg[rp] = m_new;
                    l_reg[rp] = l_reg[rp] * corr + p;
#pragma unroll
                    for (int dp = 0; dp < DPT; ++dp) {
                        const int d = lane + dp * 32;
                        if (d < D) {
                            o_reg[rp][dp] = o_reg[rp][dp] * corr + p * sv[tk * D + d];
                        }
                    }
                }
            }
        }
    }

`
	if a.Splits > 0 {
		b.WriteString(strings.Replace(score, "seq_len - k0", "k_end - k0", 1))
		b.WriteString(sm120SplitStore)
		fmt.Fprintf(&b, sm120Combine, a.Name, a.HeadDim, a.Splits)
		return b.String(), nil
	}
	b.WriteString(score)
	b.WriteString(`#pragma unroll
    for (int rp = 0; rp < RPT; ++rp) {
        const int r = warp + rp * 4;
        if (r < G) {
            const int qh = kvh * G + r;
            const float inv_l = (l_reg[rp] > 0.0f) ? (1.0f / l_reg[rp]) : 0.0f;
            const size_t o_base = ((size_t)b * HQ + qh) * D;
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) {
                const int d = lane + dp * 32;
                if (d < D) {
                    out[o_base + d] = o_reg[rp][dp] * inv_l;
                }
            }
        }
    }
}
`)

	return b.String(), nil
}

// sm120SplitTile opens a split CTA's key loop over [k_begin, k_end) and
// stages each tile: one thread per key resolves its cache slot, then K and
// V rows load as 16-byte vectors of 8 BF16 values.
const sm120SplitTile = `    const int split = (int)blockIdx.z;
    int split_keys = (seq_len + SPLITS - 1) / SPLITS;
    split_keys = max(MIN_SPLIT_KEYS, (split_keys + TILE_K - 1) / TILE_K * TILE_K);
    const int k_begin = split * split_keys;
    const int k_end = min(seq_len, k_begin + split_keys);
    const uint16_t* const k_bf16 = reinterpret_cast<const uint16_t*>(k_pages);
    const uint16_t* const v_bf16 = reinterpret_cast<const uint16_t*>(v_pages);
    __shared__ int sslot[TILE_K];
    for (int k0 = k_begin; k0 < k_end; k0 += TILE_K) {
        __syncthreads();
        if (tid < TILE_K) {
            const int k_pos = k0 + tid;
            sslot[tid] = k_pos < k_end ? page_table[b * MAX_PAGES + k_pos / PAGE_SIZE] * PAGE_SIZE + k_pos % PAGE_SIZE : -1;
        }
        __syncthreads();
        constexpr int DV = D / 8;
        for (int idx = tid; idx < TILE_K * DV; idx += 128) {
            const int tk = idx / DV, d0 = (idx % DV) * 8;
            const int slot = sslot[tk];
            uint4 kw = {0u, 0u, 0u, 0u}, vw = {0u, 0u, 0u, 0u};
            if (slot >= 0) {
                const size_t kv_off = ((size_t)slot * HKV + kvh) * D + d0;
                kw = *reinterpret_cast<const uint4*>(k_bf16 + kv_off);
                vw = *reinterpret_cast<const uint4*>(v_bf16 + kv_off);
            }
            const uint32_t kk[4] = {kw.x, kw.y, kw.z, kw.w};
            const uint32_t vv[4] = {vw.x, vw.y, vw.z, vw.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sk[tk * D + d0 + 2 * j] = __uint_as_float(kk[j] << 16);
                sk[tk * D + d0 + 2 * j + 1] = __uint_as_float(kk[j] & 0xffff0000u);
                sv[tk * D + d0 + 2 * j] = __uint_as_float(vv[j] << 16);
                sv[tk * D + d0 + 2 * j + 1] = __uint_as_float(vv[j] & 0xffff0000u);
            }
        }
`

// sm120SplitStore writes a split CTA's partial for each of its query
// heads: the unnormalized output, then the running maximum (in log2 units)
// and sum. An empty split writes maximum -inf and sum 0.
const sm120SplitStore = `#pragma unroll
    for (int rp = 0; rp < RPT; ++rp) {
        const int r = warp + rp * 4;
        if (r < G) {
            const int qh = kvh * G + r;
            float* const part = ws + (((size_t)b * HQ + qh) * SPLITS + split) * (D + 2);
#pragma unroll
            for (int dp = 0; dp < DPT; ++dp) {
                const int d = lane + dp * 32;
                if (d < D) {
                    part[d] = o_reg[rp][dp];
                }
            }
            if (lane == 0) {
                part[D] = m_reg[rp];
                part[D + 1] = l_reg[rp];
            }
        }
    }
}
`

// sm120Combine is the combine kernel, formatted with the attention
// kernel's name, D and Splits. One CTA per (sequence, query head) rescales
// every non-empty split's output and sum from its own maximum to the
// largest and divides.
const sm120Combine = `
__global__ void %[1]s_combine(const float* __restrict__ ws, float* __restrict__ out) {
    constexpr int D = %[2]d, SPLITS = %[3]d;
    const float* const part = ws + (size_t)blockIdx.x * SPLITS * (D + 2);
    float m = -__uint_as_float(0x7f800000u);
    for (int s = 0; s < SPLITS; ++s) {
        if (part[s * (D + 2) + D + 1] > 0.0f) {
            m = fmaxf(m, part[s * (D + 2) + D]);
        }
    }
    for (int d = (int)threadIdx.x; d < D; d += (int)blockDim.x) {
        float o = 0.0f, l = 0.0f;
        for (int s = 0; s < SPLITS; ++s) {
            const float* const ps = part + s * (D + 2);
            if (ps[D + 1] > 0.0f) {
                const float w = exp2f(ps[D] - m);
                o += ps[d] * w;
                l += ps[D + 1] * w;
            }
        }
        out[(size_t)blockIdx.x * D + d] = l > 0.0f ? o / l : 0.0f;
    }
}
`

// Launcher returns a C++ host launch wrapper for a. With Splits it takes
// the workspace ws after page_table and launches the combine kernel too.
func (a SM120PagedAttention) Launcher() (string, error) {
	a, err := a.norm()
	if err != nil {
		return "", err
	}
	block := a.Block()
	var b strings.Builder
	if a.Splits > 0 {
		fmt.Fprintf(&b, "inline void %s_launch(\n    int batch,\n    const float* q,\n    const uint8_t* k_pages,\n    const uint8_t* v_pages,\n    const uint8_t* k_scales,\n    const uint8_t* v_scales,\n    const int32_t* page_table,\n    float* ws,\n    float* out,\n    int seq_len,\n    cudaStream_t stream = nullptr) {\n",
			a.Name)
		fmt.Fprintf(&b, "    const dim3 grid(batch, %d, %d), block(%d, %d, %d);\n", a.NumKVHeads, a.Splits, block[0], block[1], block[2])
		fmt.Fprintf(&b, "    if (batch <= 0 || seq_len <= 0) return;\n")
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, q, k_pages, v_pages, k_scales, v_scales, page_table, ws, seq_len);\n", a.Name)
		fmt.Fprintf(&b, "    EG_LAUNCH(%s_combine, dim3(batch * %d, 1, 1), dim3(128, 1, 1), 0, stream, ws, out);\n}\n", a.Name, a.NumQHeads)
		return b.String(), nil
	}
	fmt.Fprintf(&b, "inline void %s_launch(\n    int batch,\n    const float* q,\n    const uint8_t* k_pages,\n    const uint8_t* v_pages,\n    const uint8_t* k_scales,\n    const uint8_t* v_scales,\n    const int32_t* page_table,\n    float* out,\n    int seq_len,\n    cudaStream_t stream = nullptr) {\n",
		a.Name)
	fmt.Fprintf(&b, "    const dim3 grid(batch, %d, 1), block(%d, %d, %d);\n", a.NumKVHeads, block[0], block[1], block[2])
	fmt.Fprintf(&b, "    if (batch > 0 && seq_len > 0) EG_LAUNCH(%s, grid, block, 0, stream, q, k_pages, v_pages, k_scales, v_scales, page_table, out, seq_len);\n}\n", a.Name)
	return b.String(), nil
}

// Source returns the self-contained CUDA C++ translation unit (Prelude, kernel,
// and host launcher) for a.
func (a SM120PagedAttention) Source() (string, error) {
	cuda, err := a.CUDA()
	if err != nil {
		return "", err
	}
	launcher, err := a.Launcher()
	if err != nil {
		return "", err
	}
	return cuda + "\n" + launcher, nil
}

// Reference computes the host reference output for [SM120PagedAttention] over
// paged KV cache tables with either WeightBF16 or WeightNVFP4 (with UE4M3
// per-16-element scales).
func (a SM120PagedAttention) Reference(q []float32, kPages, vPages []uint8, kScales, vScales []uint8, pageTable []int32, seqLen int) ([]float32, error) {
	a, err := a.norm()
	if err != nil {
		return nil, err
	}
	if seqLen <= 0 {
		seqLen = a.SeqLen
	}
	qPerBatch := a.NumQHeads * a.HeadDim
	if len(q) == 0 || len(q)%qPerBatch != 0 {
		return nil, fmt.Errorf("sm120 paged attention %s reference: len(q)=%d must be a positive multiple of %d", a.Name, len(q), qPerBatch)
	}
	batch := len(q) / qPerBatch
	needPages := (seqLen + a.PageSize - 1) / a.PageSize
	if needPages > a.MaxPages {
		return nil, fmt.Errorf("sm120 paged attention %s reference: seqLen=%d needs %d pages, exceeds MaxPages=%d", a.Name, seqLen, needPages, a.MaxPages)
	}
	ptStride := a.MaxPages
	if batch == 1 && len(pageTable) >= needPages && len(pageTable) < a.MaxPages {
		ptStride = len(pageTable)
	} else if len(pageTable) < batch*a.MaxPages {
		return nil, fmt.Errorf("sm120 paged attention %s reference: len(pageTable)=%d, want at least %d", a.Name, len(pageTable), batch*a.MaxPages)
	}

	g := a.NumQHeads / a.NumKVHeads
	out := make([]float32, len(q))
	dFP4 := a.HeadDim / 2
	dSF := a.HeadDim / 16

	readKV := func(pages, scales []uint8, slot, kvh, d int) (float64, error) {
		row := slot*a.NumKVHeads + kvh
		switch a.KVFormat {
		case WeightBF16:
			off := (row*a.HeadDim + d) * 2
			if off+2 > len(pages) {
				return 0, fmt.Errorf("sm120 paged attention %s reference: bf16 page offset %d out of bounds (len=%d)", a.Name, off, len(pages))
			}
			u16 := binary.LittleEndian.Uint16(pages[off : off+2])
			return float64(FromBF16Bits(u16)), nil
		case WeightNVFP4:
			bOff := row*dFP4 + d/2
			sOff := row*dSF + d/16
			if bOff >= len(pages) || sOff >= len(scales) {
				return 0, fmt.Errorf("sm120 paged attention %s reference: nvfp4 offset (%d, %d) out of bounds", a.Name, bOff, sOff)
			}
			nibble := (pages[bOff] >> (4 * (d & 1))) & 0xf
			sf := FromE4M3Bits(scales[sOff])
			return float64(FromE2M1(nibble) * sf), nil
		default:
			return 0, fmt.Errorf("sm120 paged attention %s reference: unsupported KVFormat %v", a.Name, a.KVFormat)
		}
	}

	for b := range batch {
		for qh := range a.NumQHeads {
			kvh := qh / g
			qBase := (b*a.NumQHeads + qh) * a.HeadDim
			qVec := q[qBase : qBase+a.HeadDim]

			logits := make([]float64, seqLen)
			maxLogit := math.Inf(-1)
			for kPos := range seqLen {
				page := int(pageTable[b*ptStride+kPos/a.PageSize])
				slot := page*a.PageSize + (kPos % a.PageSize)
				var dot float64
				for d := range a.HeadDim {
					kv, err := readKV(kPages, kScales, slot, kvh, d)
					if err != nil {
						return nil, err
					}
					dot += float64(qVec[d]) * kv
				}
				s := dot * float64(a.Scale)
				logits[kPos] = s
				if s > maxLogit {
					maxLogit = s
				}
			}

			var sumExp float64
			for kPos := range seqLen {
				logits[kPos] = math.Exp(logits[kPos] - maxLogit)
				sumExp += logits[kPos]
			}

			for d := range a.HeadDim {
				var acc float64
				for kPos := range seqLen {
					page := int(pageTable[b*ptStride+kPos/a.PageSize])
					slot := page*a.PageSize + (kPos % a.PageSize)
					vv, err := readKV(vPages, vScales, slot, kvh, d)
					if err != nil {
						return nil, err
					}
					acc += (logits[kPos] / sumExp) * vv
				}
				out[qBase+d] = float32(acc)
			}
		}
	}
	return out, nil
}
