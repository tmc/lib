package hopper

import (
	"fmt"
	"strings"
)

// TileGEMM computes out[m, n] = Epilogue(Σ_k x[m, k] · W[n, k], extras...)
// for many BF16 activation rows with mma.sync m16n8k16 tensor-core tiles
// and FP32 accumulation: prefill projections and batched MoE experts.
//
// Each block of 8 warps (2 × 4) owns a BM × BN output tile and streams
// BK = 64 slices of x and W through a Stages-deep cp.async ring in shared
// memory. BF16 weights feed ldmatrix directly; NVFP4 weights stay packed in
// shared memory and widen to BF16 fragments in registers (e2m1 · block
// scale is exact in BF16), with the global scale applied in the epilogue.
//
// With Grouped, W holds Experts matrices back to back and rows
// offsets[e] .. offsets[e+1] of x use matrix e (the expert-sorted layout of
// a MoE). With Gather, row m of the operand is x[rows[m]] while row m of
// out stays m, so tokens need not be copied into expert order.
type TileGEMM struct {
	Name     string
	N, K     int
	Weight   WeightFormat // WeightBF16 or WeightNVFP4
	Out      DType        // BF16 or F32
	Epilogue Lambda       // $0 is the accumulator; zero value means $0
	Extras   []Operand
	Grouped  bool
	// GLU pairs W's two halves: out has N/2 columns and the epilogue's $0
	// and $1 are the accumulators of columns c and c + N/2 (W13 rows
	// [:N/2] gate, [N/2:] up), so (mul (silu $0) $1) is SwiGLU
	// (moe8_moe_w4a16/reference.py:103-104). Extras follow as $2, ....
	GLU bool
	// WGMMA uses sm_90a warpgroup MMA (wgmma.mma_async m64nBNk16) from
	// shared-memory descriptors instead of mma.sync: two warpgroups each
	// own 64 rows of a 128 × BN tile.
	WGMMA  bool
	Gather bool
	// SumSq adds a parameter float ss[m_total, Nout/32] after out (Nout =
	// N, or N/2 with GLU): ss[m][j] is the sum of squares of the FP32
	// epilogue values of row m in columns 32j .. 32j+31, before rounding to
	// Out, summed in a fixed order. Σ_j ss[m][j] / Nout is the mean square
	// a following RMS norm needs (the MoE post-expert norm,
	// moe8_moe_w4a16/reference.py:88-90), so the norm need not reread out
	// to find it.
	SumSq bool
	// Repacked selects the RepackNVFP4WGMMA weight layout when WGMMA and
	// Weight == WeightNVFP4: w holds uint32 fragment words [N/128, K/64, 256, 4]
	// and w_sf holds E4M3 scale bytes [N/128, K/64, 256, 8], which the kernel
	// widens to core-matrix shared memory with 9-instruction dequant9 or
	// nibble-shift + bf16x2 scale multiply.
	Repacked bool
	// DequantWords selects the bitopt sub-byte unpack width when Repacked is
	// true: 0 or 1 uses 1-word per-subtile unpacking, and 2 uses the 2-word
	// bitopt.NVFP4Program2Word (eg_dequant_2w, 16 instructions across 2 words
	// via RepackNVFP4Dequant2Word).
	DequantWords int
	// Swizzle128B stages WGMMA shared-memory tiles in 128-byte XOR-swizzled
	// lines (offset ^ (((offset >> 7) & 7) << 4)) and emits 128B-swizzled
	// WGMMA descriptors (layout mode 1, bits [63:62] = 1). Without WGMMA,
	// for BF16 weights, it stores the mma.sync tiles' 64-element rows
	// unpadded, with row r's 16-byte chunk c at chunk c ^ (r & 7), so that
	// ldmatrix stays free of bank conflicts in 8/9 of the shared memory:
	// a 3-stage 128x128 tile then fits in 99 KB.
	Swizzle128B bool
	// GroupM, when > 1, orders CTA (tile_m, tile_n) coordinates in GroupM-row
	// M-bands so concurrently scheduled blocks share A and B tiles in L2 cache.
	GroupM int
	// StreamK, when > 0, launches a persistent Stream-K grid of StreamK
	// blocks (one or two per SM) in place of one block per tile. The
	// tiles × K/BK k-steps, in tile-major order, are divided evenly: block
	// b takes iterations [b·T/StreamK, (b+1)·T/StreamK) of T, so a tile
	// may be split across blocks. Thread 0 computes each tile segment's
	// bounds and broadcasts them through shared memory, so that the
	// k-loop around the wgmma is warpgroup-uniform and ptxas does not
	// serialize it (C7519). A block that covers part of a tile stores its
	// FP32 partial in the workspace ws (see [TileGEMM.WorkspaceBytes]);
	// the last to arrive sums the tile's partials in block order and runs
	// the epilogue, so the result does not depend on which block finishes
	// last. Grouped is not supported.
	StreamK int
	// WarpSpecialized, with WGMMA and Swizzle128B, runs 384-thread blocks:
	// one producer warpgroup fills a Stages-deep shared-memory ring with
	// cp.async.bulk copies that complete on full[stage] mbarriers, and two
	// consumer warpgroups wait on full[stage], run wgmma and release the
	// stage on empty[stage]. With WeightNVFP4 (which needs Repacked) the
	// product is computed transposed: the weights are wgmma's A operand,
	// dequantized in registers from RepackNVFP4WGMMA fragments, each consumer
	// warpgroup owning 64 of the tile's BN = 128 weight rows, and BM (64 or
	// 128) tokens of x are the B operand. SumSq is not supported with
	// WeightNVFP4. StreamK and DequantWords are not supported.
	WarpSpecialized bool
	// BlackwellFP4, when Weight == WeightNVFP4 and !WGMMA, emits Blackwell
	// (sm_100a / sm_120a / sm_121a GB10) hardware cvt.rn.f16x2.e2m1x2 nibble
	// widening and block-scaled mma.sync.aligned.m16n8k64.kind::mxf4nvf4
	// helpers alongside F16 m16n8k16 tensor-core accumulation.
	BlackwellFP4 bool

	BM, BN int // 64 or 128 (BN 256 with WGMMA); 0 means 128
	Stages int // 2 to 4; 0 means 3
}

const blackwellFP4Helpers = `// eg_e2m1x2_to_f16x2 widens the two E2M1 nibbles in the low byte of b to F16x2
// (cvt.rn.f16x2.e2m1x2 on sm_100a / sm_120a / sm_121a Blackwell).
EG_DEV uint32_t eg_e2m1x2_to_f16x2(uint32_t b) {
#if defined(EG_HOST_EMU)
    return eg_emu_e2m1x2_to_f16x2(b);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030 || (__CUDA_ARCH__ == 1200 && defined(__CUDA_ARCH_FEAT_SM120_ALL)))
    uint32_t r;
    asm("{\n.reg .b8 q;\ncvt.u8.u32 q, %1;\ncvt.rn.f16x2.e2m1x2 %0, q;\n}" : "=r"(r) : "r"(b));
    return r;
#else
    uint32_t h[2];
    eg_e2m1x4_to_bf16x4(b & 0xffu, h);
    return eg_pack_f16x2(eg_bf16_to_f32(h[0]), eg_bf16_hi_to_f32(h[0]));
#endif
}

// eg_e2m1x4_to_f16x4 widens four E2M1 nibbles in bits 0..15 of q to two F16x2 words.
EG_DEV void eg_e2m1x4_to_f16x4(uint32_t q, uint32_t out[2]) {
    out[0] = eg_e2m1x2_to_f16x2(q & 0xffu);
    out[1] = eg_e2m1x2_to_f16x2((q >> 8) & 0xffu);
}

// eg_e2m1x8_to_bf16 widens eight E2M1 nibbles in q to four BF16x2 words, using
// hardware cvt.rn.f16x2.e2m1x2 on Blackwell (__CUDA_ARCH__ >= 1000).
EG_DEV void eg_e2m1x8_to_bf16(uint32_t q, uint32_t out[4]) {
#if defined(EG_HOST_EMU) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t f2 = eg_e2m1x2_to_f16x2((q >> (8 * i)) & 0xffu);
        const uint32_t lo = f2 & 0xffffu, hi = f2 >> 16;
        // E2M1 values in F16 have biased exponent 14..17 (or 0 for ±0), so
        // shifting the 10-bit F16 mantissa >>3 and adding (127-15)<<7 = 0x3800
        // converts non-zero F16 to BF16 in register without FP32 round-trip.
        const uint32_t blo = (lo & 0x7fffu) ? (((lo & 0x7fffu) >> 3) + 0x3800u) | (lo & 0x8000u) : (lo & 0x8000u);
        const uint32_t bhi = (hi & 0x7fffu) ? (((hi & 0x7fffu) >> 3) + 0x3800u) | (hi & 0x8000u) : (hi & 0x8000u);
        out[i] = blo | (bhi << 16);
    }
#else
    eg_e2m1x4_to_bf16x4(q & 0xffffu, out);
    eg_e2m1x4_to_bf16x4(q >> 16, out + 2);
#endif
}

// eg_dequant_nvfp4 widens four E2M1 nibbles in bits 0..15 of q with
// cvt.rn.f16x2.e2m1x2 and multiplies by block scale byte s_e4m3 into two BF16x2 words.
EG_DEV void eg_dequant_nvfp4(uint32_t q, uint32_t s_e4m3, uint32_t out[2]) {
    uint32_t h[4];
    eg_e2m1x8_to_bf16(q & 0xffffu, h);
    const float sf = eg_e4m3_to_f32(s_e4m3);
    out[0] = eg_pack_bf16x2(eg_bf16_to_f32(h[0]) * sf, eg_bf16_hi_to_f32(h[0]) * sf);
    out[1] = eg_pack_bf16x2(eg_bf16_to_f32(h[1]) * sf, eg_bf16_hi_to_f32(h[1]) * sf);
}

// eg_mma_nvfp4_16864 issues Blackwell block-scaled mma.sync.aligned.m16n8k64
// (kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3).
EG_DEV void eg_mma_nvfp4_16864(float d[4], const uint32_t a[4], const uint32_t b[2], uint32_t sfa, uint32_t sfb) {
#if defined(EG_HOST_EMU)
    eg_emu_mma_nvfp4_16864(d, a, b, sfa, sfb);
#else
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0, 0}, {%11}, {0, 0};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
#endif
}

`

func (g TileGEMM) norm() (TileGEMM, error) {
	if !cIdent(g.Name) {
		return g, fmt.Errorf("tile gemm: bad name %q", g.Name)
	}
	if g.BM == 0 {
		g.BM = 128
	}
	if g.BN == 0 {
		g.BN = 128
	}
	if g.Stages == 0 {
		g.Stages = 3
	}
	swapAB := g.WarpSpecialized && g.Weight == WeightNVFP4
	if g.WarpSpecialized && (!g.WGMMA || !g.Swizzle128B) {
		return g, fmt.Errorf("tile gemm %s: WarpSpecialized requires WGMMA and Swizzle128B", g.Name)
	}
	if g.WarpSpecialized && (g.StreamK > 0 || g.DequantWords > 0) {
		return g, fmt.Errorf("tile gemm %s: WarpSpecialized does not support StreamK or DequantWords", g.Name)
	}
	if swapAB && (!g.Repacked || g.SumSq) {
		return g, fmt.Errorf("tile gemm %s: WarpSpecialized WeightNVFP4 requires Repacked and does not support SumSq", g.Name)
	}
	if g.WGMMA && g.BM != 128 && !(swapAB && g.BM == 64) {
		return g, fmt.Errorf("tile gemm %s: WGMMA needs BM 128", g.Name)
	}
	if g.Swizzle128B && !g.WGMMA && g.Weight != WeightBF16 {
		return g, fmt.Errorf("tile gemm %s: Swizzle128B without WGMMA requires BF16 weights", g.Name)
	}
	if g.GroupM < 0 {
		return g, fmt.Errorf("tile gemm %s: negative GroupM %d", g.Name, g.GroupM)
	}
	if g.StreamK < 0 {
		return g, fmt.Errorf("tile gemm %s: negative StreamK %d", g.Name, g.StreamK)
	}
	if g.StreamK > 0 && g.Grouped {
		return g, fmt.Errorf("tile gemm %s: StreamK does not support Grouped", g.Name)
	}
	if g.Repacked && (!g.WGMMA || g.Weight != WeightNVFP4 || g.BN != 128) {
		return g, fmt.Errorf("tile gemm %s: Repacked requires WGMMA, WeightNVFP4 and BN 128", g.Name)
	}
	if g.DequantWords < 0 || g.DequantWords > 2 || (g.DequantWords > 0 && !g.Repacked) {
		return g, fmt.Errorf("tile gemm %s: DequantWords %d requires Repacked and must be 0, 1 or 2", g.Name, g.DequantWords)
	}
	if g.BlackwellFP4 && (g.Weight != WeightNVFP4 || g.WGMMA) {
		return g, fmt.Errorf("tile gemm %s: BlackwellFP4 requires WeightNVFP4 without WGMMA", g.Name)
	}
	if g.BM != 64 && g.BM != 128 || g.BN != 64 && g.BN != 128 && !(g.WGMMA && g.BN == 256) || g.Stages < 2 || g.Stages > 4 {
		return g, fmt.Errorf("tile gemm %s: unsupported tile %dx%d, %d stages", g.Name, g.BM, g.BN, g.Stages)
	}
	if g.N < 1 || g.N%g.BN != 0 || g.K < 64 || g.K%64 != 0 || (g.Weight == WeightE4M3 && g.Swizzle128B && (g.K < 128 || g.K%128 != 0)) {
		return g, fmt.Errorf("tile gemm %s: N %d must be a multiple of %d and K %d of 64 (128 for Swizzle128B E4M3)", g.Name, g.N, g.BN, g.K)
	}
	if g.Weight == WeightE4M3 && !g.WGMMA {
		return g, fmt.Errorf("tile gemm %s: WeightE4M3 requires WGMMA", g.Name)
	}
	if g.Weight != WeightBF16 && g.Weight != WeightNVFP4 && g.Weight != WeightE4M3 {
		return g, fmt.Errorf("tile gemm %s: weight %v unsupported", g.Name, g.Weight)
	}
	if g.Out != BF16 && g.Out != F32 {
		return g, fmt.Errorf("tile gemm %s: out must be BF16 or F32", g.Name)
	}
	if g.Epilogue.root == nil {
		g.Epilogue = Identity
	}
	acc := 1
	if g.GLU {
		acc = 2
		if (g.N/2)%(g.BN/2) != 0 {
			return g, fmt.Errorf("tile gemm %s: GLU half width %d is not a multiple of %d", g.Name, g.N/2, g.BN/2)
		}
	}
	if g.Epilogue.Arity() > acc+len(g.Extras) {
		return g, fmt.Errorf("tile gemm %s: epilogue uses $%d, have %d accumulators and %d extras", g.Name, g.Epilogue.Arity()-1, acc, len(g.Extras))
	}
	seen := map[string]bool{"x": true, "rows": true, "w": true, "w_sf": true, "w_gs": true, "offsets": true, "out": true, "ss": true, "ws": true, "m_total": true}
	for _, e := range g.Extras {
		if !cIdent(e.Name) || seen[e.Name] || e.DType != F32 && e.DType != BF16 || e.Broadcast < PerCol || e.Broadcast > Full {
			return g, fmt.Errorf("tile gemm %s: bad extra %+v", g.Name, e)
		}
		seen[e.Name] = true
	}
	return g, nil
}

// WorkspaceBytes reports the bytes of workspace ws that the kernel needs
// when StreamK > 0, or 0 without Stream-K: two BM × BN FP32 partial tiles
// per block, then StreamK int32 arrival counters. The counters, the last
// 4·StreamK bytes, must be zero before the first launch; the kernel leaves
// them zero, so a CUDA graph can replay it.
func (g TileGEMM) WorkspaceBytes() int {
	g, err := g.norm()
	if err != nil || g.StreamK == 0 {
		return 0
	}
	return 4 * (2*g.StreamK*g.BM*g.BN + g.StreamK)
}

// SharedBytes returns the dynamic shared memory one block uses.
func (g TileGEMM) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	if g.WarpSpecialized {
		stage := (g.BM + g.BN) * 128
		if g.Weight == WeightNVFP4 {
			stage = (g.BM*128 + 256*(16+8) + 1023) / 1024 * 1024
		}
		// The stages, their full and empty mbarriers, and slack to align the
		// ring to 1024 bytes.
		return g.Stages*stage + 2*g.Stages*8 + 1024
	}
	if g.WGMMA {
		if g.Weight == WeightNVFP4 {
			if g.Repacked {
				return g.Stages * (g.BM*64*2 + g.BN*64*2 + 256*(16+8))
			}
			stage := g.BM*64*2 + g.BN*64*2 + g.BN*(32+4)
			if g.Swizzle128B {
				stage = (stage + 1023) / 1024 * 1024
			}
			return g.Stages * stage
		}
		if g.Weight == WeightE4M3 && !g.Swizzle128B {
			return g.Stages * (g.BM + g.BN) * 64
		}
		return g.Stages * (g.BM + g.BN) * 64 * 2
	}
	if g.Swizzle128B {
		return g.Stages * (g.BM + g.BN) * 64 * 2
	}
	b := g.BN * 72 * 2
	if g.Weight == WeightNVFP4 {
		b = g.BN * (48 + 4)
	}
	return g.Stages * (g.BM*72*2 + b)
}

func (g TileGEMM) params() []string {
	xType := "const uint16_t* __restrict__ x"
	if g.Weight == WeightE4M3 {
		xType = "const uint8_t* __restrict__ x"
	}
	p := []string{"int m_total", xType}
	if g.Gather {
		p = append(p, "const int32_t* __restrict__ rows")
	}
	switch g.Weight {
	case WeightNVFP4:
		if g.WGMMA && g.Repacked {
			p = append(p, "const uint32_t* __restrict__ w", "const uint8_t* __restrict__ w_sf", "const float* __restrict__ w_gs")
		} else {
			p = append(p, "const uint8_t* __restrict__ w", "const uint8_t* __restrict__ w_sf", "const float* __restrict__ w_gs")
		}
	case WeightE4M3:
		p = append(p, "const uint8_t* __restrict__ w")
	default:
		p = append(p, "const uint16_t* __restrict__ w")
	}
	if g.Grouped {
		p = append(p, "const int32_t* __restrict__ offsets")
	}
	for _, e := range g.Extras {
		p = append(p, "const "+e.DType.cType()+"* __restrict__ "+e.Name)
	}
	p = append(p, g.Out.cType()+"* __restrict__ out")
	if g.SumSq {
		p = append(p, "float* __restrict__ ss")
	}
	if g.StreamK > 0 {
		p = append(p, "float* __restrict__ ws")
	}
	return p
}

// Grid returns the grid and block for m rows (the most rows of any expert
// when Grouped) and experts experts (1 unless Grouped).
func (g TileGEMM) Grid(m, experts int) (grid, block [3]int) {
	g, err := g.norm()
	if err != nil {
		return
	}
	if g.StreamK > 0 {
		return [3]int{g.StreamK, 1, 1}, [3]int{256, 1, 1}
	}
	threads := 256
	if g.WarpSpecialized {
		threads = 384
	}
	return [3]int{g.N / g.BN, (m + g.BM - 1) / g.BM, experts}, [3]int{threads, 1, 1}
}

// CUDA returns the kernel source (without [Prelude]):
//
//	__global__ void Name(int m_total, x, [rows,] w..., [offsets,] extras..., out)
//
// launched with [TileGEMM.SharedBytes] of dynamic shared memory.
func (g TileGEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	if g.WarpSpecialized {
		return g.cudaWarpSpecialized()
	}
	if g.WGMMA {
		return g.cudaWGMMA()
	}
	bx := "blockIdx.x"
	if g.GroupM > 1 || g.StreamK > 0 {
		bx = "tile_n"
	}
	var b strings.Builder
	if g.BlackwellFP4 {
		b.WriteString(blackwellFP4Helpers)
	}
	fmt.Fprintf(&b, "// %s: out = %s over x[M, %d] · W[%d, %d]ᵀ (%v), %dx%d tiles, %d stages.\n",
		g.Name, g.Epilogue, g.K, g.N, g.K, g.Weight, g.BM, g.BN, g.Stages)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n", g.Name, strings.Join(g.params(), ",\n    "))
	ld := 72
	if g.Swizzle128B {
		ld = 64
	}
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = 64, STAGES = %d, LD = %d;\n", g.N, g.K, g.BM, g.BN, g.Stages, ld)
	// chunk(r, c) is the offset in a shared-memory row r of its 16-byte chunk c.
	chunk := func(r, c string) string {
		if g.Swizzle128B {
			return fmt.Sprintf("((%s) ^ ((%s) & 7)) * 8", c, r)
		}
		return fmt.Sprintf("(%s) * 8", c)
	}
	if g.GLU {
		b.WriteString("    constexpr int NOUT = N / 2;\n")
	} else {
		b.WriteString("    constexpr int NOUT = N;\n")
	}
	b.WriteString(`    constexpr int WM = BM / 2, WN = BN / 4, MT = WM / 16, NT = WN / 8, KT = K / BK;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / 4, wn = warp % 4;
`)
	if g.Grouped {
		b.WriteString("    const int e = blockIdx.z, row0 = offsets[e], row1 = offsets[e + 1];\n")
	} else {
		b.WriteString("    const int e = 0, row0 = 0, row1 = m_total;\n")
	}
	tileSetup(&b, g.GroupM, g.StreamK)
	b.WriteString("    EG_DYNAMIC_SMEM(uint16_t, sa); // [STAGES][BM][LD], then the W ring\n")
	if g.GLU {
		fmt.Fprintf(&b, `    // Tile row j: warp column j / WN holds WN/2 gate rows, then the matching up rows.
    auto wrow = [&](int j) {
        const int o = j %% WN, c = %s * (BN / 2) + (j / WN) * (WN / 2) + o %% (WN / 2);
        return o < WN / 2 ? c : NOUT + c;
    };
`, bx)
	} else {
		b.WriteString("    auto wrow = [&](int j) { return bn0 + j; };\n")
	}
	if g.Weight == WeightNVFP4 {
		b.WriteString(`    uint8_t* const sbq = reinterpret_cast<uint8_t*>(sa + STAGES * BM * LD); // [STAGES][BN][48]
    uint8_t* const sbs = sbq + STAGES * BN * 48;                                 // [STAGES][BN][4]
    const uint8_t* const we = w + (size_t)e * N * (K / 2);
    const uint8_t* const se = w_sf + (size_t)e * N * (K / 16);
    const float gs = w_gs[e];
`)
	} else {
		b.WriteString(`    uint16_t* const sb = sa + STAGES * BM * LD; // [STAGES][BN][LD]
    const uint16_t* const we = w + (size_t)e * N * K;
`)
	}
	xrow := "min(bm0 + r, row1 - 1)"
	if g.Gather {
		// Read the gathered row indices once, before the first copy, so
		// the loop's copies wait on no global load.
		xrow = "srow[r]"
		b.WriteString(`    __shared__ int srow[BM];
    for (int r = tid; r < BM; r += 256) srow[r] = rows[min(bm0 + r, row1 - 1)];
    __syncthreads();
`)
	}
	if g.Swizzle128B {
		fmt.Fprintf(&b, `    auto load = [&](int st, int kt) {
        for (int c = tid; c < BM * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(&sa[(st * BM + r) * LD + %s], x + (size_t)%s * K + kt * BK + ch * 8);
        }
`, chunk("r", "ch"), xrow)
	} else {
		fmt.Fprintf(&b, `    auto load = [&](int st, int kt) {
        for (int c = tid; c < BM * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(&sa[(st * BM + r) * LD + ch * 8], x + (size_t)%s * K + kt * BK + ch * 8);
        }
`, xrow)
	}
	if g.Weight == WeightNVFP4 {
		b.WriteString(`        for (int c = tid; c < BN * 2; c += 256) {
            const int r = c / 2, ch = c % 2;
            eg_cp_async16(&sbq[(st * BN + r) * 48 + ch * 16], we + (size_t)wrow(r) * (K / 2) + kt * 32 + ch * 16);
        }
        for (int r = tid; r < BN; r += 256) eg_cp_async4(&sbs[(st * BN + r) * 4], se + (size_t)wrow(r) * (K / 16) + kt * 4);
`)
	} else if g.Swizzle128B {
		fmt.Fprintf(&b, `        for (int c = tid; c < BN * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(&sb[(st * BN + r) * LD + %s], we + (size_t)wrow(r) * K + kt * BK + ch * 8);
        }
`, chunk("r", "ch"))
	} else {
		b.WriteString(`        for (int c = tid; c < BN * 8; c += 256) {
            const int r = c / 8, ch = c % 8;
            eg_cp_async16(&sb[(st * BN + r) * LD + ch * 8], we + (size_t)wrow(r) * K + kt * BK + ch * 8);
        }
`)
	}
	b.WriteString(`    };
    float acc[MT][NT][4] = {};
`)
	if g.StreamK > 0 {
		b.WriteString(`    const int sk_kt = sk_k1 - sk_k0;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < sk_kt) load(s, sk_k0 + s);
        eg_cp_async_commit();
    }
    for (int kt = sk_k0; kt < sk_k1; ++kt) {
        const int step = kt - sk_k0;
        eg_cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < sk_k1) load((step + STAGES - 1) % STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        const int st = step % STAGES;
`)
	} else {
		b.WriteString(`#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        eg_cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        eg_cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        const int st = kt % STAGES;
`)
	}
	if g.Swizzle128B {
		fmt.Fprintf(&b, `#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t af[MT][4], bf[NT][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const int ar = wm * WM + mt * 16 + lane %% 16;
                eg_ldmatrix_x4(af[mt], &sa[(st * BM + ar) * LD + %s]);
            }
`, chunk("ar", "ks * 2 + lane / 16"))
	} else {
		b.WriteString(`#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t af[MT][4], bf[NT][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                eg_ldmatrix_x4(af[mt], &sa[(st * BM + wm * WM + mt * 16 + lane % 16) * LD + ks * 16 + (lane / 16) * 8]);
`)
	}
	if g.Weight == WeightNVFP4 {
		if g.BlackwellFP4 {
			b.WriteString(`#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                // b0 holds k = 2t, 2t+1 (byte t of the k16 step), b1 k + 8 (byte t + 4).
                // sfb packs the 4 E4M3 block-16 scales for the k=64 slice in bytes 0..3.
                const int r = st * BN + wn * WN + nt * 8 + lane / 4;
                const uint8_t* const qb = &sbq[r * 48 + ks * 8 + lane % 4];
                const uint32_t sfb = *reinterpret_cast<const uint32_t*>(&sbs[r * 4]);
                const uint32_t s_e4m3 = (sfb >> (8 * ks)) & 0xffu;
                eg_dequant_nvfp4(qb[0] | (uint32_t)qb[4] << 8, s_e4m3, bf[nt]);
            }
`)
		} else {
			b.WriteString(`#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                // b0 holds k = 2t, 2t+1 (byte t of the k16 step), b1 k + 8 (byte t + 4).
                const int r = st * BN + wn * WN + nt * 8 + lane / 4;
                const uint8_t* const qb = &sbq[r * 48 + ks * 8 + lane % 4];
                uint32_t h[2];
                eg_e2m1x4_to_bf16x4(qb[0] | (uint32_t)qb[4] << 8, h);
                const float sf = eg_e4m3_to_f32(sbs[r * 4 + ks]);
                bf[nt][0] = eg_pack_bf16x2(eg_bf16_to_f32(h[0]) * sf, eg_bf16_hi_to_f32(h[0]) * sf);
                bf[nt][1] = eg_pack_bf16x2(eg_bf16_to_f32(h[1]) * sf, eg_bf16_hi_to_f32(h[1]) * sf);
            }
`)
		}
	} else if g.Swizzle128B {
		fmt.Fprintf(&b, `#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                uint32_t t4[4];
                const int br = wn * WN + np * 16 + lane %% 8 + 8 * (lane / 16);
                eg_ldmatrix_x4(t4, &sb[(st * BN + br) * LD + %s]);
                bf[2 * np][0] = t4[0]; bf[2 * np][1] = t4[1];
                bf[2 * np + 1][0] = t4[2]; bf[2 * np + 1][1] = t4[3];
            }
`, chunk("br", "ks * 2 + (lane / 8) % 2"))
	} else {
		b.WriteString(`#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                uint32_t t4[4];
                eg_ldmatrix_x4(t4, &sb[(st * BN + wn * WN + np * 16 + lane % 8 + 8 * (lane / 16)) * LD + ks * 16 + 8 * ((lane / 8) % 2)]);
                bf[2 * np][0] = t4[0]; bf[2 * np][1] = t4[1];
                bf[2 * np + 1][0] = t4[2]; bf[2 * np + 1][1] = t4[3];
            }
`)
	}
	b.WriteString(`#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) eg_mma_bf16(acc[mt][nt], af[mt], bf[nt], acc[mt][nt]);
        }
    }
    eg_cp_async_wait<0>();
`)
	if g.StreamK > 0 {
		b.WriteString(`    if (is_partial) {
        float* const sk_part = ws + (size_t)(2 * (int)blockIdx.x + (sk_k0 == 0 ? 1 : 0)) * (BM * BN);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    sk_part[((mt * NT + nt) * 4 + i) * 256 + tid] = acc[mt][nt][i];
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int ta = tile_idx * KT, tb = ta + KT;
            int cta_lo = (int)(((int64_t)ta * STREAM_K) / total_iters);
            while (cta_lo + 1 < STREAM_K && (int)(((int64_t)(cta_lo + 1) * total_iters) / STREAM_K) <= ta) ++cta_lo;
            int cta_hi = cta_lo;
            while (cta_hi + 1 < STREAM_K && (int)(((int64_t)(cta_hi + 1) * total_iters) / STREAM_K) < tb) ++cta_hi;
            const int old = atomicAdd(&counters[cta_lo], sk_k1 - sk_k0);
            sk_ctrl[1] = (old + (sk_k1 - sk_k0) == KT) ? 1 : 0;
            sk_ctrl[2] = cta_lo;
            sk_ctrl[3] = cta_hi;
        }
        __syncthreads();
        if (!sk_ctrl[1]) continue;
        __threadfence();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    acc[mt][nt][i] = 0.0f;
        for (int c = sk_ctrl[2]; c <= sk_ctrl[3]; ++c) {
            const int c0 = (int)(((int64_t)c * total_iters) / STREAM_K);
            const int c1 = (int)(((int64_t)(c + 1) * total_iters) / STREAM_K);
            if (c0 >= c1) continue;
            const float* const c_part = ws + (size_t)(2 * c + (c0 <= tile_idx * KT ? 1 : 0)) * (BM * BN);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        acc[mt][nt][i] += eg_ld_cg(&c_part[((mt * NT + nt) * 4 + i) * 256 + tid]);
        }
        __syncthreads();
        if (tid == 0) counters[sk_ctrl[2]] = 0;
    }
`)
	}
	nts, col := "NT", "bn0 + wn * WN"
	if g.GLU {
		nts, col = "NT / 2", bx+" * (BN / 2) + wn * (WN / 2)"
	}
	if g.SumSq {
		b.WriteString("    float sq[MT][2] = {};\n")
	}
	fmt.Fprintf(&b, `#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < %s; ++nt) {
            float yv[2]; // a row's two adjacent columns, stored together
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = bm0 + wm * WM + mt * 16 + lane / 4 + 8 * (i / 2);
                const int n = %s + nt * 8 + 2 * (lane %% 4) + (i & 1);
                if (m >= row1) continue;
`, nts, col)
	scale := ""
	if g.Weight == WeightNVFP4 {
		scale = " * gs"
	}
	fmt.Fprintf(&b, "                const float v = acc[mt][nt][i]%s;\n", scale)
	args := []string{"v"}
	if g.GLU {
		fmt.Fprintf(&b, "                const float u = acc[mt][nt + NT / 2][i]%s;\n", scale)
		args = append(args, "u")
	}
	for _, e := range g.Extras {
		args = append(args, e.load("m", "n", "NOUT"))
	}
	expr, err := g.Epilogue.CUDA(args...)
	if err != nil {
		return "", err
	}
	fmt.Fprintf(&b, "                const float y = %s;\n", expr)
	b.WriteString("                yv[i & 1] = y;\n")
	b.WriteString("                if (i & 1) eg_store<2>(out + (size_t)m * NOUT + n - 1, yv);\n")
	if g.SumSq {
		b.WriteString("                sq[mt][i / 2] += y * y;\n")
	}
	b.WriteString("            }\n        }\n")
	if g.SumSq {
		bno := "BN"
		if g.GLU {
			bno = "BN / 2"
		}
		fmt.Fprintf(&b, `    // Row sums of squares: lanes of a row, then the warps of each 32-column
    // group in column order.
    constexpr int E = (%s) / 32, WPE = 4 / E;
    __shared__ float red_ss[4][BM];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float s = sq[mt][h];
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            if (lane %% 4 == 0) red_ss[wn][wm * WM + mt * 16 + lane / 4 + 8 * h] = s;
        }
    __syncthreads();
    for (int c = tid; c < BM * E; c += 256) {
        const int r = c / E, j = c %% E;
        if (bm0 + r >= row1) continue;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < WPE; ++k) s += red_ss[j * WPE + k][r];
        ss[(size_t)(bm0 + r) * (NOUT / 32) + %s * E + j] = s;
    }
`, bno, bx)
	}
	if g.StreamK > 0 {
		b.WriteString("    }\n")
	}
	b.WriteString("}\n")
	return b.String(), nil
}

// tileSetup emits the CTA coordinate setup, or the persistent Stream-K outer
// loop and uniform shared-memory broadcast when streamK > 0.
func tileSetup(b *strings.Builder, groupM, streamK int) {
	if streamK <= 0 {
		tileCoord(b, groupM)
		b.WriteString("    if (bm0 >= row1) return;\n")
		return
	}
	fmt.Fprintf(b, `    constexpr int STREAM_K = %d, NUM_N = N / BN;
    const int num_m = (row1 - row0 + BM - 1) / BM;
    const int total_tiles = num_m * NUM_N;
    const int total_iters = total_tiles * KT;
    int* const counters = reinterpret_cast<int*>(ws + (size_t)(2 * STREAM_K) * BM * BN);
    __shared__ int sk_ctrl[4];
    for (int sk_base = (int)(((int64_t)blockIdx.x * total_iters) / STREAM_K),
             sk_end = (int)(((int64_t)(blockIdx.x + 1) * total_iters) / STREAM_K);
         ; ) {
        __syncthreads();
        if (tid == 0) {
            if (sk_base >= sk_end) {
                sk_ctrl[0] = -1;
                sk_ctrl[1] = 0;
                sk_ctrl[2] = 0;
                sk_ctrl[3] = 0;
            } else {
                const int tile_idx = sk_base / KT;
                const int k_begin = sk_base - tile_idx * KT;
                const int k_end = min(sk_end, (tile_idx + 1) * KT) - tile_idx * KT;
                sk_ctrl[0] = tile_idx;
                sk_ctrl[1] = k_begin;
                sk_ctrl[2] = k_end;
                sk_ctrl[3] = (k_begin > 0 || k_end < KT) ? 1 : 0;
            }
        }
        __syncthreads();
        const int tile_idx = sk_ctrl[0];
        if (tile_idx < 0) break;
        const int sk_k0 = sk_ctrl[1], sk_k1 = sk_ctrl[2], is_partial = sk_ctrl[3];
        sk_base = tile_idx * KT + sk_k1;
`, streamK)
	if groupM <= 1 {
		b.WriteString(`        const int tile_m = tile_idx / NUM_N, tile_n = tile_idx % NUM_N;
        const int bm0 = row0 + tile_m * BM, bn0 = tile_n * BN;
`)
	} else {
		fmt.Fprintf(b, `        constexpr int GROUP_M = %d;
        const int in_group = GROUP_M * NUM_N;
        const int first_m = (tile_idx / in_group) * GROUP_M;
        const int gsz = min(num_m - first_m, GROUP_M);
        const int tile_m = first_m + (tile_idx %% in_group) %% gsz;
        const int tile_n = (tile_idx %% in_group) / gsz;
        const int bm0 = row0 + tile_m * BM, bn0 = tile_n * BN;
`, groupM)
	}
}

// tileCoord emits the CTA (tile_m, tile_n) and (bm0, bn0) coordinate setup.
// When groupM <= 1, it emits the direct blockIdx.y / blockIdx.x mapping.
// When groupM > 1, it decomposes linear pid = blockIdx.y * gridDim.x + blockIdx.x
// into GroupM-row M-bands so concurrently scheduled blocks share A and B tiles in L2.
func tileCoord(b *strings.Builder, groupM int) {
	if groupM <= 1 {
		b.WriteString("    const int bm0 = row0 + blockIdx.y * BM, bn0 = blockIdx.x * BN;\n")
		return
	}
	fmt.Fprintf(b, `    constexpr int GROUP_M = %d, NUM_N = N / BN;
    const int num_m = (row1 - row0 + BM - 1) / BM;
    const int pid = blockIdx.y * gridDim.x + blockIdx.x;
    if (pid >= num_m * NUM_N) return;
    const int in_group = GROUP_M * NUM_N;
    const int first_m = (pid / in_group) * GROUP_M;
    const int gsz = min(num_m - first_m, GROUP_M);
    const int tile_m = first_m + (pid %% in_group) %% gsz;
    const int tile_n = (pid %% in_group) / gsz;
    const int bm0 = row0 + tile_m * BM, bn0 = tile_n * BN;
`, groupM)
}

// Launcher returns a host function
//
//	void Name_launch(int m, [int max_rows, int experts,] <params...>, cudaStream_t stream)
//
// where m is x's row count and, when Grouped, max_rows bounds any expert's
// rows (the grid is fixed by it, so the launch can be graph-captured).
func (g TileGEMM) Launcher() (string, error) {
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
	head := "int m"
	rows, experts := "m", "1"
	if g.Grouped {
		head, rows, experts = "int m, int max_rows, int experts", "max_rows", "experts"
	}
	fmt.Fprintf(&b, "inline void %s_launch(%s, %s, cudaStream_t stream) {\n", g.Name, head, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    constexpr int smem = %d;\n", g.SharedBytes())
	fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", g.Name)
	if g.StreamK > 0 {
		fmt.Fprintf(&b, "    const dim3 grid(%d, 1, 1), block(256);\n", g.StreamK)
		fmt.Fprintf(&b, "    if (m > 0) EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", g.Name, strings.Join(names, ", "))
	} else {
		_, block := g.Grid(0, 1)
		fmt.Fprintf(&b, "    const dim3 grid(%d, (%s + %d) / %d, %s), block(%d);\n", g.N/g.BN, rows, g.BM-1, g.BM, experts, block[0])
		fmt.Fprintf(&b, "    if (grid.y > 0 && grid.z > 0) EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", g.Name, strings.Join(names, ", "))
	}
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (g TileGEMM) Source() (string, error) {
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

// Reference computes the [m, N] output ([m, N/2] with GLU) in float64
// with the epilogue in FP32. w holds one weight set, or one per expert when Grouped; rows and
// offsets are as the kernel takes them (nil when unused).
func (g TileGEMM) Reference(m int, x []uint16, rows, offsets []int32, w []GEMMWeights, extras [][]float32) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if len(extras) != len(g.Extras) || g.Gather != (rows != nil) || g.Grouped != (offsets != nil) {
		return nil, fmt.Errorf("tile gemm %s reference: bad inputs", g.Name)
	}
	if !g.Grouped {
		offsets = []int32{0, int32(m)}
	}
	if len(w) != len(offsets)-1 || offsets[0] != 0 || int(offsets[len(offsets)-1]) != m {
		return nil, fmt.Errorf("tile gemm %s reference: %d weight sets for %d groups", g.Name, len(w), len(offsets)-1)
	}
	xRow := func(i int) []uint16 {
		if rows != nil {
			i = int(rows[i])
		}
		return x[i*g.K : (i+1)*g.K]
	}
	epi, ops, d := g.Epilogue, g.Extras, g.Out
	if g.GLU {
		epi, ops, d = Identity, nil, F32
	}
	out := make([]float32, m*g.N)
	for e := range w {
		var idx []int
		for i := offsets[e]; i < offsets[e+1]; i++ {
			idx = append(idx, int(i))
		}
		if err := gemmRows(g.N, g.K, g.Weight, w[e], xRow, idx, epi, ops, extras, d, out); err != nil {
			return nil, fmt.Errorf("tile gemm %s reference: %w", g.Name, err)
		}
	}
	if !g.GLU {
		return out, nil
	}
	half := g.N / 2
	glu := make([]float32, m*half)
	for i := range m {
		for c := range half {
			args := []float32{out[i*g.N+c], out[i*g.N+half+c]}
			for j, o := range g.Extras {
				switch o.Broadcast {
				case PerCol:
					args = append(args, extras[j][c])
				case PerRow:
					args = append(args, extras[j][i])
				default:
					args = append(args, extras[j][i*half+c])
				}
			}
			glu[i*half+c] = g.Out.Round(g.Epilogue.Eval(args...))
		}
	}
	return glu, nil
}
