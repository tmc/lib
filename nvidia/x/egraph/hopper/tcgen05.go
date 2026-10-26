package hopper

import (
	"fmt"
	"math"
	"strings"
)

// TileEpilogue selects the elementwise epilogue applied to TCGen05GEMM accumulators.
type TileEpilogue uint8

const (
	TileLinear TileEpilogue = iota // identity: y = acc
	TileSiLU                       // SiLU: y = acc / (1 + exp(-acc))
	TileGLU                        // SwiGLU over paired (gate, up) columns: y = silu(gate) * up
)

func (e TileEpilogue) String() string {
	switch e {
	case TileLinear:
		return "linear"
	case TileSiLU:
		return "silu"
	case TileGLU:
		return "glu"
	}
	return fmt.Sprintf("TileEpilogue(%d)", uint8(e))
}

func (e TileEpilogue) valid() bool {
	return e == TileLinear || e == TileSiLU || e == TileGLU
}

// TCGen05GEMM configures a Blackwell sm_100a Tensor Memory (TMEM) GEMM kernel
// using tcgen05.alloc, tcgen05.mma.cta_group::1, tcgen05.ld, and tcgen05.dealloc.
type TCGen05GEMM struct {
	Name     string
	N, K     int
	BM, BN   int          // e.g., BM=128, BN=128 or 256 (accumulators live in TMEM, not registers!)
	BK       int          // reduction tile step (default 64)
	Weight   WeightFormat // WeightBF16, WeightE4M3, WeightNVFP4
	Out      DType
	Epilogue TileEpilogue // TileLinear, TileSiLU, TileGLU
}

const tcgen05DeviceHelpers = `// Blackwell sm_100a Tensor Memory (TMEM) and tcgen05 helpers.
EG_DEV void eg_tmem_alloc(uint32_t* smem_addr_out, uint32_t num_cols) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    if (threadIdx.x == 0) eg_emu_tmem_alloc(smem_addr_out, num_cols);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if ((threadIdx.x >> 5) == 0) {
        const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_addr_out);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(s), "r"(num_cols) : "memory");
    }
#endif
}

EG_DEV void eg_tmem_dealloc(uint32_t tmem_addr, uint32_t num_cols) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    if (threadIdx.x == 0) eg_emu_tmem_dealloc(tmem_addr, num_cols);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if ((threadIdx.x >> 5) == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr), "r"(num_cols) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_mma_bf16(uint32_t tmem_d, const void* smem_a, const void* smem_b, int M, int N, int K, int accum_enable) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    if (threadIdx.x == 0) eg_emu_tcgen05_mma_bf16(tmem_d, smem_a, smem_b, M, N, K, accum_enable);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a, (uint32_t)(K * 2), (uint32_t)(8 * K * 2));
        const uint64_t db = eg_wgmma_desc(smem_b, (uint32_t)(K * 2), (uint32_t)(8 * K * 2));
        const uint32_t idesc = ((uint32_t)M << 24) | ((uint32_t)N << 16) | (uint32_t)K;
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_mma_e4m3(uint32_t tmem_d, const void* smem_a, const void* smem_b, int M, int N, int K, int accum_enable) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    if (threadIdx.x == 0) eg_emu_tcgen05_mma_e4m3(tmem_d, smem_a, smem_b, M, N, K, accum_enable);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a, (uint32_t)K, (uint32_t)(8 * K));
        const uint64_t db = eg_wgmma_desc(smem_b, (uint32_t)K, (uint32_t)(8 * K));
        const uint32_t idesc = ((uint32_t)M << 24) | ((uint32_t)N << 16) | (uint32_t)K;
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_mma_nvfp4(
    uint32_t tmem_d,
    const uint8_t* smem_a_fp4, const uint8_t* smem_a_scale,
    const uint8_t* smem_b_fp4, const uint8_t* smem_b_scale,
    int M, int N, int K, int accum_enable) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    if (threadIdx.x == 0) {
        eg_emu_tcgen05_mma_nvfp4(tmem_d, smem_a_fp4, smem_a_scale, smem_b_fp4, smem_b_scale, M, N, K, accum_enable);
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint64_t da = eg_wgmma_desc(smem_a_fp4, (uint32_t)(K / 2), (uint32_t)(8 * (K / 2)));
        const uint64_t db = eg_wgmma_desc(smem_b_fp4, (uint32_t)(K / 2), (uint32_t)(8 * (K / 2)));
        const uint32_t sfa = (uint32_t)__cvta_generic_to_shared(smem_a_scale);
        const uint32_t sfb = (uint32_t)__cvta_generic_to_shared(smem_b_scale);
        const uint32_t idesc = ((uint32_t)M << 24) | ((uint32_t)N << 16) | (uint32_t)K;
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;\n}\n"
            :: "r"(tmem_d), "l"(da), "l"(db), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accum_enable) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_commit(uint64_t* mbar) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    (void)mbar;
    if (threadIdx.x == 0) eg_emu_tcgen05_commit();
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (threadIdx.x == 0) {
        const uint32_t s = (uint32_t)__cvta_generic_to_shared(mbar);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(s) : "memory");
    }
#endif
}

EG_DEV void eg_tcgen05_wait() {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    eg_emu_tcgen05_wait();
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

EG_DEV void eg_tcgen05_ld_f32(float* reg_out, uint32_t tmem_addr, int row, int col_offset, int count) {
#if defined(EG_EMU) || defined(EG_HOST_EMU)
    eg_emu_tcgen05_ld_f32(reg_out, tmem_addr, row, col_offset, count);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    for (int c = 0; c < count; c += 4) {
        const uint32_t addr = ((uint32_t)(row & 0x7f) << 16) | ((tmem_addr + (uint32_t)col_offset + (uint32_t)c) & 0xffffu);
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
            : "=f"(reg_out[c]), "=f"(reg_out[c + 1]), "=f"(reg_out[c + 2]), "=f"(reg_out[c + 3])
            : "r"(addr) : "memory");
    }
#endif
}

`

func (g TCGen05GEMM) norm() (TCGen05GEMM, error) {
	if g.Name == "" {
		g.Name = "tcgen05_gemm"
	}
	if !cIdent(g.Name) {
		return g, fmt.Errorf("tcgen05 gemm: bad name %q", g.Name)
	}
	if g.BM == 0 {
		g.BM = 128
	}
	if g.BN == 0 {
		g.BN = 128
	}
	if g.BK == 0 {
		g.BK = 64
	}
	if g.Out == 0 {
		g.Out = F32
	}
	if g.BM != 64 && g.BM != 128 {
		return g, fmt.Errorf("tcgen05 gemm %s: BM %d must be 64 or 128", g.Name, g.BM)
	}
	if g.BN < 32 || g.BN > 512 || g.BN&(g.BN-1) != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: BN %d must be a power of two in [32, 512]", g.Name, g.BN)
	}
	if g.BK < 16 || g.BK%16 != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: BK %d must be a positive multiple of 16", g.Name, g.BK)
	}
	if g.N < g.BN || g.N%g.BN != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: N %d must be a positive multiple of BN %d", g.Name, g.N, g.BN)
	}
	if g.K < g.BK || g.K%g.BK != 0 {
		return g, fmt.Errorf("tcgen05 gemm %s: K %d must be a positive multiple of BK %d", g.Name, g.K, g.BK)
	}
	switch g.Weight {
	case WeightBF16, WeightE4M3, WeightNVFP4:
	default:
		return g, fmt.Errorf("tcgen05 gemm %s: unsupported weight format %v", g.Name, g.Weight)
	}
	if g.Out != F32 && g.Out != BF16 {
		return g, fmt.Errorf("tcgen05 gemm %s: out must be F32 or BF16, got %v", g.Name, g.Out)
	}
	if !g.Epilogue.valid() {
		return g, fmt.Errorf("tcgen05 gemm %s: invalid epilogue %v", g.Name, g.Epilogue)
	}
	if g.Epilogue == TileGLU && (g.BN < 64 || (g.N/2)%(g.BN/2) != 0) {
		return g, fmt.Errorf("tcgen05 gemm %s: GLU requires BN >= 64 and N/2 divisible by BN/2", g.Name)
	}
	return g, nil
}

// SharedBytes returns the dynamic shared memory bytes per CTA needed by g.
func (g TCGen05GEMM) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	switch g.Weight {
	case WeightBF16:
		return (g.BM + g.BN) * g.BK * 2
	case WeightE4M3:
		return (g.BM + g.BN) * g.BK
	case WeightNVFP4:
		return (g.BM+g.BN)*(g.BK/2) + (g.BM+g.BN)*(g.BK/16)
	}
	return 0
}

// Grid returns the launch grid and block dimensions for m activation rows.
func (g TCGen05GEMM) Grid(m int) (grid, block [3]int) {
	g, err := g.norm()
	if err != nil {
		return
	}
	return [3]int{g.N / g.BN, (m + g.BM - 1) / g.BM, 1}, [3]int{128, 1, 1}
}

func (g TCGen05GEMM) params() []string {
	switch g.Weight {
	case WeightNVFP4:
		return []string{
			"int m_total",
			"const uint8_t* __restrict__ a_fp4",
			"const uint8_t* __restrict__ a_scale",
			"const uint8_t* __restrict__ b_fp4",
			"const uint8_t* __restrict__ b_scale",
			g.Out.cType() + "* __restrict__ out",
		}
	case WeightE4M3:
		return []string{
			"int m_total",
			"const uint8_t* __restrict__ x",
			"const uint8_t* __restrict__ w",
			g.Out.cType() + "* __restrict__ out",
		}
	default:
		return []string{
			"int m_total",
			"const uint16_t* __restrict__ x",
			"const uint16_t* __restrict__ w",
			g.Out.cType() + "* __restrict__ out",
		}
	}
}

// CUDA returns the kernel source without [Prelude].
func (g TCGen05GEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(tcgen05DeviceHelpers)
	fmt.Fprintf(&b, "// %s: Blackwell sm_100a TMEM GEMM (%v, epilogue=%s, %dx%dx%d tile).\n",
		g.Name, g.Weight, g.Epilogue, g.BM, g.BN, g.BK)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n",
		g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = %d, KT = K / BK;\n",
		g.N, g.K, g.BM, g.BN, g.BK)
	if g.Epilogue == TileGLU {
		b.WriteString("    constexpr int NOUT = N / 2;\n")
	} else {
		b.WriteString("    constexpr int NOUT = N;\n")
	}
	b.WriteString(`    const int tid = threadIdx.x;
    const int bm0 = blockIdx.y * BM;
    const int bn0 = blockIdx.x * BN;
    if (bm0 >= m_total) return;
    __shared__ uint32_t s_tmem_addr;
    __shared__ uint64_t s_tmem_mbar;
    eg_tmem_alloc(&s_tmem_addr, BN);
    __syncthreads();
    const uint32_t tmem_d = s_tmem_addr;
    EG_DYNAMIC_SMEM(unsigned char, smem);
`)
	if g.Epilogue == TileGLU {
		b.WriteString(`    auto wrow = [&](int j) {
        const int c = blockIdx.x * (BN / 2) + j % (BN / 2);
        return j < BN / 2 ? c : NOUT + c;
    };
`)
	} else {
		b.WriteString("    auto wrow = [&](int j) { return bn0 + j; };\n")
	}

	switch g.Weight {
	case WeightBF16:
		b.WriteString(`    uint16_t* const sa = reinterpret_cast<uint16_t*>(smem);
    uint16_t* const sb = sa + BM * BK;
    for (int kt = 0; kt < KT; ++kt) {
        for (int i = tid; i < BM * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gm = bm0 + r;
            sa[i] = (gm < m_total) ? x[(size_t)gm * K + kt * BK + k] : (uint16_t)0u;
        }
        for (int i = tid; i < BN * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gn = wrow(r);
            sb[i] = w[(size_t)gn * K + kt * BK + k];
        }
        eg_fence_proxy_async();
        __syncthreads();
        eg_tcgen05_mma_bf16(tmem_d, sa, sb, BM, BN, BK, kt > 0 ? 1 : 0);
        __syncthreads();
    }
`)
	case WeightE4M3:
		b.WriteString(`    uint8_t* const sa = reinterpret_cast<uint8_t*>(smem);
    uint8_t* const sb = sa + BM * BK;
    for (int kt = 0; kt < KT; ++kt) {
        for (int i = tid; i < BM * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gm = bm0 + r;
            sa[i] = (gm < m_total) ? x[(size_t)gm * K + kt * BK + k] : (uint8_t)0u;
        }
        for (int i = tid; i < BN * BK; i += 128) {
            const int r = i / BK, k = i % BK;
            const int gn = wrow(r);
            sb[i] = w[(size_t)gn * K + kt * BK + k];
        }
        eg_fence_proxy_async();
        __syncthreads();
        eg_tcgen05_mma_e4m3(tmem_d, sa, sb, BM, BN, BK, kt > 0 ? 1 : 0);
        __syncthreads();
    }
`)
	case WeightNVFP4:
		b.WriteString(`    constexpr int BK_FP4 = BK / 2, BK_SF = BK / 16;
    constexpr int K_FP4 = K / 2, K_SF = K / 16;
    uint8_t* const sa_fp4 = reinterpret_cast<uint8_t*>(smem);
    uint8_t* const sa_sf = sa_fp4 + BM * BK_FP4;
    uint8_t* const sb_fp4 = sa_sf + BM * BK_SF;
    uint8_t* const sb_sf = sb_fp4 + BN * BK_FP4;
    for (int kt = 0; kt < KT; ++kt) {
        for (int i = tid; i < BM * BK_FP4; i += 128) {
            const int r = i / BK_FP4, k = i % BK_FP4;
            const int gm = bm0 + r;
            sa_fp4[i] = (gm < m_total) ? a_fp4[(size_t)gm * K_FP4 + kt * BK_FP4 + k] : (uint8_t)0u;
        }
        for (int i = tid; i < BM * BK_SF; i += 128) {
            const int r = i / BK_SF, k = i % BK_SF;
            const int gm = bm0 + r;
            sa_sf[i] = (gm < m_total) ? a_scale[(size_t)gm * K_SF + kt * BK_SF + k] : (uint8_t)0u;
        }
        for (int i = tid; i < BN * BK_FP4; i += 128) {
            const int r = i / BK_FP4, k = i % BK_FP4;
            const int gn = wrow(r);
            sb_fp4[i] = b_fp4[(size_t)gn * K_FP4 + kt * BK_FP4 + k];
        }
        for (int i = tid; i < BN * BK_SF; i += 128) {
            const int r = i / BK_SF, k = i % BK_SF;
            const int gn = wrow(r);
            sb_sf[i] = b_scale[(size_t)gn * K_SF + kt * BK_SF + k];
        }
        eg_fence_proxy_async();
        __syncthreads();
        eg_tcgen05_mma_nvfp4(tmem_d, sa_fp4, sa_sf, sb_fp4, sb_sf, BM, BN, BK, kt > 0 ? 1 : 0);
        __syncthreads();
    }
`)
	}

	b.WriteString(`    eg_tcgen05_commit(&s_tmem_mbar);
    __syncthreads();
    eg_tcgen05_wait();
    __syncthreads();
    if (tid < BM && bm0 + tid < m_total) {
        const int gm = bm0 + tid;
`)
	switch g.Epilogue {
	case TileGLU:
		b.WriteString(`        const int bn_out = blockIdx.x * (BN / 2);
        for (int c = 0; c < BN / 2; c += 4) {
            float gate[4], up[4], yv[4];
            eg_tcgen05_ld_f32(gate, tmem_d, tid, c, 4);
            eg_tcgen05_ld_f32(up, tmem_d, tid, (BN / 2) + c, 4);
            for (int j = 0; j < 4; ++j) {
                yv[j] = eg_silu(gate[j]) * up[j];
            }
            eg_store<4>(out + (size_t)gm * NOUT + bn_out + c, yv);
        }
`)
	case TileSiLU:
		b.WriteString(`        for (int c = 0; c < BN; c += 4) {
            float acc[4], yv[4];
            eg_tcgen05_ld_f32(acc, tmem_d, tid, c, 4);
            for (int j = 0; j < 4; ++j) {
                yv[j] = eg_silu(acc[j]);
            }
            eg_store<4>(out + (size_t)gm * NOUT + bn0 + c, yv);
        }
`)
	default: // TileLinear
		b.WriteString(`        for (int c = 0; c < BN; c += 4) {
            float acc[4];
            eg_tcgen05_ld_f32(acc, tmem_d, tid, c, 4);
            eg_store<4>(out + (size_t)gm * NOUT + bn0 + c, acc);
        }
`)
	}
	b.WriteString(`    }
    __syncthreads();
    eg_tmem_dealloc(tmem_d, BN);
}
`)
	return b.String(), nil
}

// Launcher returns the host launch wrapper for g.
func (g TCGen05GEMM) Launcher() (string, error) {
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
	fmt.Fprintf(&b, "    constexpr int smem = %d;\n", g.SharedBytes())
	fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", g.Name)
	fmt.Fprintf(&b, "    const dim3 grid(%d, (m + %d) / %d, 1), block(128);\n", g.N/g.BN, g.BM-1, g.BM)
	fmt.Fprintf(&b, "    if (grid.y > 0) EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", g.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the Blackwell tcgen05 kernel, and its launcher.
func (g TCGen05GEMM) Source() (string, error) {
	src, err := g.CUDA()
	if err != nil {
		return "", err
	}
	l, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

func evalTileEpilogue(e TileEpilogue, gate, up float32) float32 {
	switch e {
	case TileSiLU:
		return gate / (1 + float32(math.Exp(float64(-gate))))
	case TileGLU:
		s := gate / (1 + float32(math.Exp(float64(-gate))))
		return s * up
	default:
		return gate
	}
}

// Reference computes the float64 reference output for WeightBF16 or WeightE4M3
// with FP32 epilogue and Out rounding. For WeightE4M3, x holds E4M3 bytes in
// the low 8 bits of each uint16 element (matching TileGEMM.Reference).
func (g TCGen05GEMM) Reference(m int, x []uint16, w GEMMWeights) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if m < 1 || len(x) != m*g.K {
		return nil, fmt.Errorf("tcgen05 gemm %s reference: bad x length %d for m=%d K=%d", g.Name, len(x), m, g.K)
	}
	nout := g.N
	if g.Epilogue == TileGLU {
		nout = g.N / 2
	}
	dots := make([]float32, m*g.N)
	for n := 0; n < g.N; n++ {
		for r := 0; r < m; r++ {
			var sum float64
			for k := 0; k < g.K; k++ {
				var av, bv float64
				switch g.Weight {
				case WeightBF16:
					av = float64(FromBF16Bits(x[r*g.K+k]))
					bv = float64(FromBF16Bits(w.BF16[n*g.K+k]))
				case WeightE4M3:
					av = float64(FromE4M3Bits(uint8(x[r*g.K+k])))
					bv = float64(FromE4M3Bits(w.E4M3[n*g.K+k]))
				default:
					return nil, fmt.Errorf("tcgen05 gemm %s reference: use ReferenceNVFP4 for WeightNVFP4", g.Name)
				}
				sum += av * bv
			}
			dots[r*g.N+n] = float32(sum)
		}
	}
	out := make([]float32, m*nout)
	for r := 0; r < m; r++ {
		for c := 0; c < nout; c++ {
			var y float32
			if g.Epilogue == TileGLU {
				y = evalTileEpilogue(TileGLU, dots[r*g.N+c], dots[r*g.N+nout+c])
			} else {
				y = evalTileEpilogue(g.Epilogue, dots[r*g.N+c], 0)
			}
			out[r*nout+c] = g.Out.Round(y)
		}
	}
	return out, nil
}

// ReferenceNVFP4 computes the float64 reference output for WeightNVFP4 with
// block-scaled E2M1 operands (aFP4 [m, K/2], aScale [m, K/16], bFP4 [N, K/2],
// bScale [N, K/16]), FP32 epilogue, and Out rounding.
func (g TCGen05GEMM) ReferenceNVFP4(m int, aFP4, aScale, bFP4, bScale []uint8) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if g.Weight != WeightNVFP4 {
		return nil, fmt.Errorf("tcgen05 gemm %s: ReferenceNVFP4 requires WeightNVFP4", g.Name)
	}
	kFP4, kSF := g.K/2, g.K/16
	if len(aFP4) != m*kFP4 || len(aScale) != m*kSF || len(bFP4) != g.N*kFP4 || len(bScale) != g.N*kSF {
		return nil, fmt.Errorf("tcgen05 gemm %s: bad nvfp4 slice lengths", g.Name)
	}
	nout := g.N
	if g.Epilogue == TileGLU {
		nout = g.N / 2
	}
	dots := make([]float32, m*g.N)
	for n := 0; n < g.N; n++ {
		for r := 0; r < m; r++ {
			var sum float64
			for k := 0; k < g.K; k++ {
				ab := aFP4[r*kFP4+k/2]
				bb := bFP4[n*kFP4+k/2]
				ac := (ab >> (4 * (k & 1))) & 0xf
				bc := (bb >> (4 * (k & 1))) & 0xf
				asf := FromE4M3Bits(aScale[r*kSF+k/16])
				bsf := FromE4M3Bits(bScale[n*kSF+k/16])
				av := float64(FromE2M1(ac) * asf)
				bv := float64(FromE2M1(bc) * bsf)
				sum += av * bv
			}
			dots[r*g.N+n] = float32(sum)
		}
	}
	out := make([]float32, m*nout)
	for r := 0; r < m; r++ {
		for c := 0; c < nout; c++ {
			var y float32
			if g.Epilogue == TileGLU {
				y = evalTileEpilogue(TileGLU, dots[r*g.N+c], dots[r*g.N+nout+c])
			} else {
				y = evalTileEpilogue(g.Epilogue, dots[r*g.N+c], 0)
			}
			out[r*nout+c] = g.Out.Round(y)
		}
	}
	return out, nil
}
