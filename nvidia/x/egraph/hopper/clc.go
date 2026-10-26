package hopper

import (
	"fmt"
	"strings"
)

const clcHelpers = `// Blackwell sm_100a clusterlaunchcontrol (PTX ISA 8.6+) work-stealing helpers.
EG_DEV void eg_clc_mbar_init(uint64_t* mbar, uint32_t count) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    eg_emu_mbar_init(mbar, (int)count);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
#else
    (void)mbar;
    (void)count;
#endif
}

// eg_clc_try_cancel_async issues an asynchronous clusterlaunchcontrol.try_cancel
// request to steal an unlaunched CTA tile ID into 16-byte shared-memory
// response_smem, signaling completion on mbar.
EG_DEV void eg_clc_try_cancel_async(uint4* response_smem, uint64_t* mbar) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    eg_emu_clc_try_cancel_async(response_smem, mbar);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const uint32_t resp_addr = (uint32_t)__cvta_generic_to_shared(response_smem);
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], 16;\n\t"
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];"
        :: "r"(resp_addr), "r"(mbar_addr) : "memory");
#else
    (void)response_smem;
    (void)mbar;
#endif
}

// eg_clc_wait waits for the mbarrier phase parity of an asynchronous
// clusterlaunchcontrol.try_cancel request to complete.
EG_DEV void eg_clc_wait(uint64_t* mbar, uint32_t phase) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    eg_emu_clc_wait(mbar, (int)phase);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "EG_CLC_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra EG_CLC_WAIT_%=;\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase) : "memory");
#else
    (void)mbar;
    (void)phase;
#endif
}

// eg_clc_query_is_canceled decodes the 128-bit try_cancel response and reports
// whether a valid tile was stolen.
EG_DEV bool eg_clc_query_is_canceled(uint4 resp) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return eg_emu_clc_query_is_canceled(resp);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t pred;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b128 %0;\n\t"
        "mov.b128 %0, {%1, %2, %3, %4};\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, %0;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(pred)
        : "r"(resp.x), "r"(resp.y), "r"(resp.z), "r"(resp.w));
    return pred != 0;
#else
    (void)resp;
    return false;
#endif
}

// eg_clc_query_first_ctaid_x extracts the stolen CTA's first_ctaid.x from a
// canceled 128-bit clusterlaunchcontrol response.
EG_DEV uint32_t eg_clc_query_first_ctaid_x(uint4 resp) {
#if defined(EG_HOST_EMU) || defined(EG_EMU)
    return eg_emu_clc_query_first_ctaid_x(resp);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t ctaid_x;
    asm volatile(
        "{\n\t"
        ".reg .b128 %1;\n\t"
        "mov.b128 %1, {%1, %2, %3, %4};\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, %1;\n\t"
        "}"
        : "=r"(ctaid_x)
        : "r"(resp.x), "r"(resp.y), "r"(resp.z), "r"(resp.w));
    return ctaid_x;
#else
    (void)resp;
    return 0;
#endif
}
`

// CLCHelpers returns the device C++ wrappers for Blackwell sm_100a
// clusterlaunchcontrol.try_cancel and clusterlaunchcontrol.query_cancel.
func CLCHelpers() string {
	return clcHelpers
}

// CLCPersistentTileGEMM configures a Blackwell sm_100a persistent GEMM kernel
// that steals output tiles dynamically using clusterlaunchcontrol.try_cancel
// instead of global atomicAdd counters.
type CLCPersistentTileGEMM struct {
	Name       string
	N, K       int
	BM, BN, BK int
	ActiveCTAs int
	Weight     WeightFormat // WeightBF16, WeightE4M3, WeightNVFP4
	Out        DType
	Epilogue   Lambda
}

func (g CLCPersistentTileGEMM) norm() (CLCPersistentTileGEMM, error) {
	if g.Name == "" {
		g.Name = "clc_tile_gemm"
	}
	if !cIdent(g.Name) {
		return g, fmt.Errorf("clc tile gemm: bad name %q", g.Name)
	}
	if g.BM == 0 {
		g.BM = 64
	}
	if g.BN == 0 {
		g.BN = 64
	}
	if g.BK == 0 {
		g.BK = 64
	}
	if g.ActiveCTAs == 0 {
		g.ActiveCTAs = 4
	}
	if g.Weight == 0 {
		g.Weight = WeightBF16
	}
	if g.Out == 0 {
		g.Out = F32
	}
	if g.Epilogue.root == nil {
		g.Epilogue = Identity
	}
	if g.N <= 0 || g.K <= 0 || g.BM <= 0 || g.BN <= 0 || g.BK <= 0 {
		return g, fmt.Errorf("clc tile gemm %s: dimensions N=%d K=%d BM=%d BN=%d BK=%d must be positive",
			g.Name, g.N, g.K, g.BM, g.BN, g.BK)
	}
	if g.N%g.BN != 0 || g.K%g.BK != 0 {
		return g, fmt.Errorf("clc tile gemm %s: N=%d must be a multiple of BN=%d and K=%d of BK=%d",
			g.Name, g.N, g.BN, g.K, g.BK)
	}
	if g.Weight == WeightNVFP4 && g.K%16 != 0 {
		return g, fmt.Errorf("clc tile gemm %s: WeightNVFP4 requires K=%d to be a multiple of 16", g.Name, g.K)
	}
	if g.ActiveCTAs <= 0 {
		return g, fmt.Errorf("clc tile gemm %s: ActiveCTAs=%d must be positive", g.Name, g.ActiveCTAs)
	}
	if g.Weight != WeightBF16 && g.Weight != WeightE4M3 && g.Weight != WeightNVFP4 {
		return g, fmt.Errorf("clc tile gemm %s: unsupported weight format %v", g.Name, g.Weight)
	}
	if g.Out != F32 && g.Out != BF16 && g.Out != E4M3 {
		return g, fmt.Errorf("clc tile gemm %s: unsupported output dtype %v", g.Name, g.Out)
	}
	if g.Epilogue.Arity() > 1 {
		return g, fmt.Errorf("clc tile gemm %s: epilogue uses $%d, have 1 accumulator", g.Name, g.Epilogue.Arity()-1)
	}
	return g, nil
}

// TotalTiles returns the total number of BM x BN output tiles for m rows.
func (g CLCPersistentTileGEMM) TotalTiles(m int) int {
	g, err := g.norm()
	if err != nil || m <= 0 {
		return 0
	}
	numM := (m + g.BM - 1) / g.BM
	numN := g.N / g.BN
	return numM * numN
}

// SharedBytes returns the shared memory bytes used by one CTA.
func (g CLCPersistentTileGEMM) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	return 4*(g.BM*g.BK+g.BN*g.BK) + 64
}

func (g CLCPersistentTileGEMM) params() []string {
	xType := "const uint16_t* __restrict__ x"
	if g.Weight == WeightE4M3 {
		xType = "const uint8_t* __restrict__ x"
	}
	p := []string{"int m_total", xType}
	switch g.Weight {
	case WeightNVFP4:
		p = append(p, "const uint8_t* __restrict__ w", "const uint8_t* __restrict__ w_sf", "const float* __restrict__ w_gs")
	case WeightE4M3:
		p = append(p, "const uint8_t* __restrict__ w")
	default:
		p = append(p, "const uint16_t* __restrict__ w")
	}
	p = append(p, g.Out.cType()+"* __restrict__ out")
	return p
}

// CUDA returns the device kernel source (without [Prelude] or [CLCHelpers]).
func (g CLCPersistentTileGEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	epilogueExpr, err := g.Epilogue.CUDA("v")
	if err != nil {
		return "", err
	}

	var b strings.Builder
	fmt.Fprintf(&b, "// %s: Blackwell sm_100a CLC persistent GEMM out = %s over x[M, %d] * W[%d, %d]^T (%v), %dx%dx%d tiles, %d active CTAs.\n",
		g.Name, g.Epilogue, g.K, g.N, g.K, g.Weight, g.BM, g.BN, g.BK, g.ActiveCTAs)
	fmt.Fprintf(&b, "__global__ void %s(\n    %s) {\n", g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = %d, KT = K / BK;\n",
		g.N, g.K, g.BM, g.BN, g.BK)
	b.WriteString(`    constexpr int NUM_N = N / BN;
    constexpr int MAX_ELEMS = (BM * BN + 31) / 32;
    const int tid = (int)threadIdx.x;
    const int bdim = (int)blockDim.x;
    const int num_m = (m_total + BM - 1) / BM;
    const int total_tiles = num_m * NUM_N;
`)
	if g.Weight == WeightNVFP4 {
		b.WriteString("    const float gs = w_gs[0];\n")
	}
	b.WriteString(`    __shared__ float sa[BM * BK];
    __shared__ float sb[BN * BK];
    __shared__ uint4 clc_response;
    __shared__ uint64_t clc_mbar;
    __shared__ int clc_next_tile;

    if (tid == 0) {
        eg_clc_mbar_init(&clc_mbar, 1);
    }
    __syncthreads();

    uint32_t clc_phase = 0;
    for (int tile_idx = (int)blockIdx.x; tile_idx < total_tiles; ) {
        const int tile_m = tile_idx / NUM_N;
        const int tile_n = tile_idx % NUM_N;
        const int bm0 = tile_m * BM;
        const int bn0 = tile_n * BN;

        if (tid == 0) {
            eg_clc_try_cancel_async(&clc_response, &clc_mbar);
        }

        double acc[MAX_ELEMS] = {};
        for (int kt = 0; kt < KT; ++kt) {
            __syncthreads();
            for (int c = tid; c < BM * BK; c += bdim) {
                const int r = c / BK, k = c % BK;
                const int gm = bm0 + r, gk = kt * BK + k;
`)
	if g.Weight == WeightE4M3 {
		b.WriteString("                sa[c] = (gm < m_total) ? eg_e4m3_to_f32(x[(size_t)gm * K + gk]) : 0.0f;\n")
	} else {
		b.WriteString("                sa[c] = (gm < m_total) ? eg_bf16_to_f32(x[(size_t)gm * K + gk]) : 0.0f;\n")
	}
	b.WriteString(`            }
            for (int c = tid; c < BN * BK; c += bdim) {
                const int n = c / BK, k = c % BK;
                const int gn = bn0 + n, gk = kt * BK + k;
                if (gn < N) {
`)
	switch g.Weight {
	case WeightNVFP4:
		b.WriteString(`                    const uint8_t packed = w[(size_t)gn * (K / 2) + gk / 2];
                    const uint32_t nibble = (gk & 1) ? (packed >> 4) : (packed & 0xfu);
                    uint32_t h[2];
                    eg_e2m1x4_to_bf16x4(nibble, h);
                    const float code = eg_bf16_to_f32(h[0]);
                    const float sf = eg_e4m3_to_f32(w_sf[(size_t)gn * (K / 16) + gk / 16]);
                    sb[c] = code * (sf * gs);
`)
	case WeightE4M3:
		b.WriteString("                    sb[c] = eg_e4m3_to_f32(w[(size_t)gn * K + gk]);\n")
	default:
		b.WriteString("                    sb[c] = eg_bf16_to_f32(w[(size_t)gn * K + gk]);\n")
	}
	b.WriteString(`                } else {
                    sb[c] = 0.0f;
                }
            }
            __syncthreads();
            for (int e = 0, idx = tid; idx < BM * BN; idx += bdim, ++e) {
                const int r = idx / BN, c = idx % BN;
                double s = acc[e];
                for (int k = 0; k < BK; ++k) {
                    s += (double)sa[r * BK + k] * (double)sb[c * BK + k];
                }
                acc[e] = s;
            }
        }

        for (int e = 0, idx = tid; idx < BM * BN; idx += bdim, ++e) {
            const int gm = bm0 + idx / BN;
            const int gn = bn0 + idx % BN;
            if (gm < m_total && gn < N) {
                const float v = (float)acc[e];
`)
	fmt.Fprintf(&b, "                const float y = %s;\n", epilogueExpr)
	switch g.Out {
	case BF16:
		b.WriteString("                out[(size_t)gm * N + gn] = eg_f32_to_bf16(y);\n")
	case E4M3:
		b.WriteString("                out[(size_t)gm * N + gn] = (uint8_t)(eg_f32x2_to_e4m3x2(y, 0.0f) & 0xffu);\n")
	default:
		b.WriteString("                out[(size_t)gm * N + gn] = y;\n")
	}
	b.WriteString(`            }
        }

        if (tid == 0) {
            eg_clc_wait(&clc_mbar, clc_phase);
            clc_phase ^= 1u;
            if (eg_clc_query_is_canceled(clc_response)) {
                clc_next_tile = (int)eg_clc_query_first_ctaid_x(clc_response);
            } else {
                clc_next_tile = total_tiles;
            }
        }
        __syncthreads();
        tile_idx = clc_next_tile;
    }
}
`)
	return b.String(), nil
}

// Launcher returns a host launcher function for g.
func (g CLCPersistentTileGEMM) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	params := g.params()[1:]
	names := []string{"m"}
	for _, p := range params {
		f := strings.Fields(p)
		names = append(names, f[len(f)-1])
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int m, %s, cudaStream_t stream) {\n", g.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    const dim3 grid(%d, 1, 1), block(128);\n", g.ActiveCTAs)
	fmt.Fprintf(&b, "    if (m > 0) EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", g.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns the complete CUDA translation unit for g, including [Prelude],
// [CLCHelpers], the persistent kernel, and its host launcher.
func (g CLCPersistentTileGEMM) Source() (string, error) {
	cuda, err := g.CUDA()
	if err != nil {
		return "", err
	}
	launcher, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + CLCHelpers() + "\n" + cuda + "\n" + launcher, nil
}

// Reference computes the [m, N] output in float64 with the epilogue evaluated
// in FP32 and rounded to g.Out.
func (g CLCPersistentTileGEMM) Reference(m int, x []uint16, w GEMMWeights) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if m <= 0 || len(x) != m*g.K {
		return nil, fmt.Errorf("clc tile gemm %s reference: bad inputs (m=%d, len(x)=%d, K=%d)", g.Name, m, len(x), g.K)
	}
	out := make([]float32, m*g.N)
	rows := make([]int, m)
	for i := range rows {
		rows[i] = i
	}
	xRow := func(i int) []uint16 { return x[i*g.K : (i+1)*g.K] }
	if err := gemmRows(g.N, g.K, g.Weight, w, xRow, rows, g.Epilogue, nil, nil, g.Out, out); err != nil {
		return nil, fmt.Errorf("clc tile gemm %s reference: %w", g.Name, err)
	}
	return out, nil
}
