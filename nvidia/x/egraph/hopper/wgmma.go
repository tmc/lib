package hopper

import (
	"fmt"
	"strings"
)

// nvfp4WGMMAHelpers is the header-free device helper block emitted ahead of a
// WGMMA NVFP4 kernel for Repacked (dequant9 / scale2 / mul.rn.bf16x2).
const nvfp4WGMMAHelpers = `EG_DEV uint32_t eg_rotl32(uint32_t x, int n) {
#if defined(EG_HOST_EMU)
    return (x << n) | (x >> (32 - n));
#else
    return __funnelshift_l(x, x, n);
#endif
}

// eg_dequant9 unpacks a dequant9-repacked 32-bit word into four bf16x2
// registers holding unscaled E2M1 subnormals (v * 2^-126 as bf16) in 9
// integer instructions.
EG_DEV void eg_dequant9(uint32_t q, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    constexpr uint32_t M = 0x81C081C0u, C1 = 0x01000100u, C2 = 0x80C080C0u;
    a0 = q & M;
    a1 = (q << 4) & M;
    a2 = eg_rotl32(q, 10) & M;
    a3 = ((q >> 2) & C1) | ((q << 6) & C2);
}

// eg_scale_e4m3_bf16x2 converts an E4M3 byte b to bf16x2 of (sf * 2^118, sf * 2^118):
// positive normal E4M3 bytes take the 1-instruction b * 16 + 0x7700 fast path;
// zero, subnormals, negatives and 0x7f/0xff NaN take the exact fallback.
EG_DEV uint32_t eg_scale_e4m3_bf16x2(uint32_t b) {
    const uint32_t byte = b & 0xffu;
    if ((byte & 0x80u) == 0u && ((byte >> 3) & 15u) != 0u && byte != 0x7fu) {
        const uint32_t x = byte | (byte << 16);
        return x * 16u + 0x77007700u;
    }
    if ((byte & 0x7fu) == 0x7fu) return 0x7fc07fc0u;
    const float v = eg_e4m3_to_f32(byte) * __uint_as_float(0x7a800000u); // 2^118
    const uint32_t u = eg_f32_to_bf16(v);
    return u | (u << 16);
}

EG_DEV uint32_t eg_mul_bf16x2(uint32_t a, uint32_t b) {
#if defined(EG_HOST_EMU)
    return eg_pack_bf16x2(eg_bf16_to_f32(a) * eg_bf16_to_f32(b), eg_bf16_hi_to_f32(a) * eg_bf16_hi_to_f32(b));
#else
    uint32_t r;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
#endif
}

EG_DEV uint32_t eg_wgmma_decode_nibble(uint32_t q, int j) {
    const uint32_t x = q >> (4 * j);
    return ((x << 6) & 0x01C001C0u) | ((x << 12) & 0x80008000u);
}

`

const wgmmaSwizzleHelpers = `// eg_wgmma_desc_128b returns a 128B-swizzled wgmma shared-memory matrix
// descriptor (layout mode 1, bits [63:62] = 1, LBO = 16) for the matrix at p
// with stride byte offset sbo.
EG_DEV uint64_t eg_wgmma_desc_128b(const void* p, uint32_t sbo) {
#if defined(EG_HOST_EMU)
    return eg_emu_wgmma_desc_128b(p, sbo);
#else
    const uint64_t a = (uint32_t)__cvta_generic_to_shared(p);
    return ((a & 0x3ffffu) >> 4) | ((uint64_t)1u << 16) | ((uint64_t)((sbo & 0x3ffffu) >> 4) << 32) | ((uint64_t)1u << 62);
#endif
}

EG_DEV uint32_t eg_swizzle_128b(uint32_t offset) {
    return offset ^ (((offset >> 7) & 7u) << 4);
}

`

// cudaWGMMA emits the WGMMA form of g (normalized, g.WGMMA set).
//
// Without Swizzle128B, x and W tiles sit in shared memory in the no-swizzle
// K-major layout of 8-row × 16-byte core matrices: element (r, k) of a
// 64-wide K slice is at byte ((r/8)·8 + ch)·128 + (r%8)·16 + (k_in_ch), so the
// descriptors have leading (K) byte offset 128 and stride (8-row) byte offset
// 1024. With Swizzle128B, each 8-row × 128-byte block is XOR-swizzled by
// offset ^ (((offset >> 7) & 7) << 4) and addressed with eg_wgmma_desc_128b.
func (g TileGEMM) cudaWGMMA() (string, error) {
	var b strings.Builder
	if g.Weight == WeightNVFP4 {
		b.WriteString(nvfp4WGMMAHelpers)
	}
	if g.Swizzle128B {
		b.WriteString(wgmmaSwizzleHelpers)
	}
	wg := g.Name + "_wgmma"
	if g.Weight == WeightE4M3 {
		b.WriteString(wgmmaE4M3Func(wg, g.BN))
	} else {
		b.WriteString(wgmmaFunc(wg, g.BN))
	}
	bx := "blockIdx.x"
	if g.GroupM > 1 {
		bx = "tile_n"
	}
	bk := 64
	if g.Weight == WeightE4M3 && g.Swizzle128B {
		bk = 128
	}
	fmt.Fprintf(&b, "// %s: out = %s over x[M, %d] · W[%d, %d]ᵀ (%v), %dx%d tiles, %d stages, wgmma.\n",
		g.Name, g.Epilogue, g.K, g.N, g.K, g.Weight, g.BM, g.BN, g.Stages)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n", g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = %d, STAGES = %d, KT = K / BK;\n", g.N, g.K, g.BM, g.BN, bk, g.Stages)
	if g.GLU {
		b.WriteString("    constexpr int NOUT = N / 2;\n")
	} else {
		b.WriteString("    constexpr int NOUT = N;\n")
	}
	b.WriteString("    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wgi = warp / 4;\n")
	if g.Grouped {
		b.WriteString("    const int e = blockIdx.z, row0 = offsets[e], row1 = offsets[e + 1];\n")
	} else {
		b.WriteString("    const int e = 0, row0 = 0, row1 = m_total;\n")
	}
	tileCoord(&b, g.GroupM)
	b.WriteString(`    if (bm0 >= row1) return;
    EG_DYNAMIC_SMEM(unsigned char, smem); // [STAGES][BM*64 + BN*64] BF16 in core-matrix order
`)
	switch {
	case g.Weight == WeightNVFP4 && g.Repacked:
		fmt.Fprintf(&b, `    constexpr int STAGE = (BM + BN) * BK * 2 + 256 * (16 + 8);
    const uint32_t* const we = w + ((size_t)e * (N / 128) + %s) * KT * 256 * 4;
    const uint8_t* const se = w_sf + ((size_t)e * (N / 128) + %s) * KT * 256 * 8;
    const float gs = w_gs[e] * 256.0f;
`, bx, bx)
	case g.Weight == WeightNVFP4:
		stageExpr := "(BM + BN) * BK * 2 + BN * (32 + 4)"
		if g.Swizzle128B {
			stageExpr = "((BM + BN) * BK * 2 + BN * (32 + 4) + 1023) / 1024 * 1024"
		}
		fmt.Fprintf(&b, `    constexpr int STAGE = %s;
    const uint8_t* const we = w + (size_t)e * N * (K / 2);
    const uint8_t* const se = w_sf + (size_t)e * N * (K / 16);
    const float gs = w_gs[e];
`, stageExpr)
	case g.Weight == WeightE4M3:
		b.WriteString(`    constexpr int STAGE = (BM + BN) * BK;
    const uint8_t* const we = w + (size_t)e * N * K;
`)
	default:
		b.WriteString(`    constexpr int STAGE = (BM + BN) * BK * 2;
    const uint16_t* const we = w + (size_t)e * N * K;
`)
	}
	if !g.Repacked {
		if g.GLU {
			fmt.Fprintf(&b, `    // Tile row j < BN/2 is gate column j, row BN/2 + j its up column.
    auto wrow = [&](int j) {
        const int c = %s * (BN / 2) + j %% (BN / 2);
        return j < BN / 2 ? c : NOUT + c;
    };
`, bx)
		} else {
			b.WriteString("    auto wrow = [&](int j) { return bn0 + j; };\n")
		}
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
		b.WriteString("    auto core = [](int r, int ch) { return (int)eg_swizzle_128b((uint32_t)(r * 128 + ch * 16)); };\n")
	} else if g.Weight == WeightE4M3 {
		b.WriteString("    auto core = [](int r, int ch) { return ((r / 8) * 4 + ch) * 128 + (r % 8) * 16; };\n")
	} else {
		b.WriteString("    auto core = [](int r, int ch) { return ((r / 8) * 8 + ch) * 128 + (r % 8) * 16; };\n")
	}
	switch {
	case g.Weight == WeightE4M3 && g.Swizzle128B:
		fmt.Fprintf(&b, `    auto load = [&](int st, int kt) {
        unsigned char* const sa = smem + st * STAGE;
        unsigned char* const sb = sa + BM * BK;
        for (int c = tid; c < BM * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(sa + core(r, ch), x + (size_t)%s * K + kt * BK + ch * 16);
        }
        for (int c = tid; c < BN * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(sb + core(r, ch), we + (size_t)wrow(r) * K + kt * BK + ch * 16);
        }
    };
`, xrow)
	case g.Weight == WeightE4M3:
		fmt.Fprintf(&b, `    auto load = [&](int st, int kt) {
        unsigned char* const sa = smem + st * STAGE;
        unsigned char* const sb = sa + BM * BK;
        for (int c = tid; c < BM * 4; c += 256) {
            const int r = c / 4, ch = c %% 4;
            eg_cp_async16(sa + core(r, ch), x + (size_t)%s * K + kt * BK + ch * 16);
        }
        for (int c = tid; c < BN * 4; c += 256) {
            const int r = c / 4, ch = c %% 4;
            eg_cp_async16(sb + core(r, ch), we + (size_t)wrow(r) * K + kt * BK + ch * 16);
        }
    };
`, xrow)
	default:
		fmt.Fprintf(&b, `    auto load = [&](int st, int kt) {
        unsigned char* const sa = smem + st * STAGE;
        unsigned char* const sb = sa + BM * BK * 2;
        for (int c = tid; c < BM * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(sa + core(r, ch), x + (size_t)%s * K + kt * BK + ch * 8);
        }
`, xrow)
		switch {
		case g.Weight == WeightNVFP4 && g.Repacked:
			b.WriteString(`        uint8_t* const sbq = sb + BN * BK * 2;
        uint8_t* const sbs = sbq + 256 * 16;
        eg_cp_async16(sbq + tid * 16, we + (size_t)(kt * 256 + tid) * 4);
        eg_cp_async4(sbs + tid * 8, se + (size_t)(kt * 256 + tid) * 8);
        eg_cp_async4(sbs + tid * 8 + 4, se + (size_t)(kt * 256 + tid) * 8 + 4);
    };
`)
		case g.Weight == WeightNVFP4:
			b.WriteString(`        uint8_t* const sbq = sb + BN * BK * 2;
        uint8_t* const sbs = sbq + BN * 32;
        for (int c = tid; c < BN * 2; c += 256) {
            const int r = c / 2, ch = c % 2;
            eg_cp_async16(sbq + r * 32 + ch * 16, we + (size_t)wrow(r) * (K / 2) + kt * 32 + ch * 16);
        }
        for (int r = tid; r < BN; r += 256) eg_cp_async4(sbs + r * 4, se + (size_t)wrow(r) * (K / 16) + kt * 4);
    };
`)
		default:
			b.WriteString(`        for (int c = tid; c < BN * 8; c += 256) {
            const int r = c / 8, ch = c % 8;
            eg_cp_async16(sb + core(r, ch), we + (size_t)wrow(r) * K + kt * BK + ch * 8);
        }
    };
`)
		}
	}
	b.WriteString(`    float d[BN / 2] = {};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        eg_cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        eg_cp_async_wait<STAGES - 2>();
`)
	switch {
	case g.Weight == WeightNVFP4 && g.Repacked:
		storeExpr := "sb + core(r, ch) + tq * 4"
		if g.Swizzle128B {
			storeExpr = "sb + eg_swizzle_128b((uint32_t)(r * 128 + ch * 16 + tq * 4))"
		}
		fmt.Fprintf(&b, `        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) %% STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        unsigned char* const sa = smem + (kt %% STAGES) * STAGE;
        unsigned char* const sb = sa + BM * BK * 2;
        const uint32_t* const sbq = reinterpret_cast<const uint32_t*>(sb + BN * BK * 2 + tid * 16);
        const uint8_t* const sbs = sb + BN * BK * 2 + 256 * 16 + tid * 8;
        const int g = lane / 4, tq = lane %% 4;
        const int rg = wgi * 64 + (warp %% 4) * 16 + g;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const uint32_t q = sbq[ks];
            const uint32_t s0 = eg_scale_e4m3_bf16x2(sbs[2 * ks]);
            const uint32_t s1 = eg_scale_e4m3_bf16x2(sbs[2 * ks + 1]);
            const uint32_t a[4] = {
                eg_mul_bf16x2(eg_wgmma_decode_nibble(q, 0), s0),
                eg_mul_bf16x2(eg_wgmma_decode_nibble(q, 1), s1),
                eg_mul_bf16x2(eg_wgmma_decode_nibble(q, 2), s0),
                eg_mul_bf16x2(eg_wgmma_decode_nibble(q, 3), s1),
            };
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = (j & 1) ? (rg + 8) : rg;
                const int ch = ks * 2 + (j >> 1);
                *reinterpret_cast<uint32_t*>(%s) = a[j];
            }
        }
        eg_fence_proxy_async();
        __syncthreads();
`, storeExpr)
	case g.Weight == WeightNVFP4:
		b.WriteString(`        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        unsigned char* const sa = smem + (kt % STAGES) * STAGE;
        unsigned char* const sb = sa + BM * BK * 2;
        const uint8_t* const sbq = sb + BN * BK * 2;
        const uint8_t* const sbs = sbq + BN * 32;
        for (int c = tid; c < BN * 8; c += 256) {
            const int r = c / 8, ch = c % 8, ks = ch / 2;
            const uint32_t q = *reinterpret_cast<const uint32_t*>(sbq + r * 32 + ch * 4);
            uint32_t h0[2], h1[2];
            eg_e2m1x4_to_bf16x4(q & 0xffffu, h0);
            eg_e2m1x4_to_bf16x4(q >> 16, h1);
            const float sf = eg_e4m3_to_f32(sbs[r * 4 + ks]);
            *reinterpret_cast<eg_u32x4*>(sb + core(r, ch)) = eg_u32x4{
                eg_pack_bf16x2(eg_bf16_to_f32(h0[0]) * sf, eg_bf16_hi_to_f32(h0[0]) * sf),
                eg_pack_bf16x2(eg_bf16_to_f32(h0[1]) * sf, eg_bf16_hi_to_f32(h0[1]) * sf),
                eg_pack_bf16x2(eg_bf16_to_f32(h1[0]) * sf, eg_bf16_hi_to_f32(h1[0]) * sf),
                eg_pack_bf16x2(eg_bf16_to_f32(h1[1]) * sf, eg_bf16_hi_to_f32(h1[1]) * sf),
            };
        }
        eg_fence_proxy_async();
        __syncthreads();
`)
	case g.Weight == WeightE4M3:
		b.WriteString(`        eg_fence_proxy_async();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        const unsigned char* const sa = smem + (kt % STAGES) * STAGE;
        const unsigned char* const sb = sa + BM * BK;
`)
	default:
		b.WriteString(`        eg_fence_proxy_async();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        const unsigned char* const sa = smem + (kt % STAGES) * STAGE;
        const unsigned char* const sb = sa + BM * BK * 2;
`)
	}
	switch {
	case g.Weight == WeightE4M3 && g.Swizzle128B:
		fmt.Fprintf(&b, `        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < BK / 32; ++ks)
            %s(d, eg_wgmma_desc_128b(sa + wgi * 8 * 1024 + ks * 32, 1024), eg_wgmma_desc_128b(sb + ks * 32, 1024));
        eg_wgmma_commit();
        eg_wgmma_wait<0>();
    }
    eg_cp_async_wait<0>();
`, wg)
	case g.Weight == WeightE4M3:
		fmt.Fprintf(&b, `        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < BK / 32; ++ks)
            %s(d, eg_wgmma_desc(sa + wgi * 8 * 512 + ks * 256, 128, 512), eg_wgmma_desc(sb + ks * 256, 128, 512));
        eg_wgmma_commit();
        eg_wgmma_wait<0>();
    }
    eg_cp_async_wait<0>();
`, wg)
	case g.Swizzle128B:
		fmt.Fprintf(&b, `        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks)
            %s(d, eg_wgmma_desc_128b(sa + wgi * 8 * 1024 + ks * 32, 1024), eg_wgmma_desc_128b(sb + ks * 32, 1024));
        eg_wgmma_commit();
        eg_wgmma_wait<0>();
    }
    eg_cp_async_wait<0>();
`, wg)
	default:
		fmt.Fprintf(&b, `        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks)
            %s(d, eg_wgmma_desc(sa + wgi * 8 * 1024 + ks * 256, 128, 1024), eg_wgmma_desc(sb + ks * 256, 128, 1024));
        eg_wgmma_commit();
        eg_wgmma_wait<0>();
    }
    eg_cp_async_wait<0>();
`, wg)
	}
	nts, col := "BN / 8", "bn0"
	if g.GLU {
		nts, col = "BN / 16", bx+" * (BN / 2)"
	}
	if g.SumSq {
		b.WriteString("    float sq[2] = {};\n")
	}
	fmt.Fprintf(&b, `    // Accumulator d[nt*4 + i]: row (warp %% 4)*16 + lane/4 + 8*(i/2), column nt*8 + 2*(lane%%4) + (i&1).
#pragma unroll
    for (int nt = 0; nt < %s; ++nt) {
        float yv[2]; // a row's two adjacent columns, stored together
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = bm0 + wgi * 64 + (warp %% 4) * 16 + lane / 4 + 8 * (i / 2);
            const int n = %s + nt * 8 + 2 * (lane %% 4) + (i & 1);
            if (m >= row1) continue;
`, nts, col)
	scale := ""
	if g.Weight == WeightNVFP4 {
		scale = " * gs"
	}
	if g.GLU && g.Repacked {
		// Under RepackNVFP4WGMMA with GLU, rows 16b..16b+7 are gate rows 8b..8b+7
		// and rows 16b+8..16b+15 are the matching up rows, so nt pairs (2*nt, 2*nt+1).
		fmt.Fprintf(&b, "            const float v = d[(2 * nt) * 4 + i]%s;\n", scale)
		fmt.Fprintf(&b, "            const float u = d[(2 * nt + 1) * 4 + i]%s;\n", scale)
	} else {
		fmt.Fprintf(&b, "            const float v = d[nt * 4 + i]%s;\n", scale)
		if g.GLU {
			fmt.Fprintf(&b, "            const float u = d[(nt + BN / 16) * 4 + i]%s;\n", scale)
		}
	}
	args := []string{"v"}
	if g.GLU {
		args = append(args, "u")
	}
	for _, e := range g.Extras {
		args = append(args, e.load("m", "n", "NOUT"))
	}
	expr, err := g.Epilogue.CUDA(args...)
	if err != nil {
		return "", err
	}
	if g.SumSq {
		fmt.Fprintf(&b, "            const float y = %s;\n", expr)
		b.WriteString("            yv[i & 1] = y;\n")
	} else {
		fmt.Fprintf(&b, "            yv[i & 1] = %s;\n", expr)
	}
	b.WriteString("            if (i & 1) eg_store<2>(out + (size_t)m * NOUT + n - 1, yv);\n")
	if g.SumSq {
		b.WriteString(`            sq[i / 2] += y * y;
            if ((nt % 4) == 3 && (i & 1) == 1) {
                float s = sq[i / 2];
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                if (lane % 4 == 0) ss[(size_t)m * (NOUT / 32) + n / 32] = s;
                sq[i / 2] = 0.0f;
            }
`)
	}
	b.WriteString("        }\n    }\n}\n")
	return b.String(), nil
}

// wgmmaFunc returns a device function name(float* d, uint64_t da,
// uint64_t db) issuing wgmma.mma_async m64n<n>k16 BF16 → FP32 with D += A·B,
// both operands K-major in shared memory.
func wgmmaFunc(name string, n int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "EG_DEV void %s(float* d, uint64_t da, uint64_t db) {\n", name)
	b.WriteString("#if defined(EG_HOST_EMU)\n")
	fmt.Fprintf(&b, "    eg_emu_wgmma_bf16(d, da, db, %d);\n", n)
	b.WriteString("#else\n")
	regs := make([]string, n/2)
	outs := make([]string, n/2)
	for i := range regs {
		regs[i] = fmt.Sprintf("%%%d", i)
		outs[i] = fmt.Sprintf("\"+f\"(d[%d])", i)
	}
	// Operand lines of 16 keep generated lines short.
	var regLines, outLines []string
	for i := 0; i < len(regs); i += 16 {
		regLines = append(regLines, strings.Join(regs[i:min(i+16, len(regs))], ", "))
		outLines = append(outLines, strings.Join(outs[i:min(i+16, len(outs))], ", "))
	}
	// scale-d is a predicate operand; setp makes it true (D += A·B).
	fmt.Fprintf(&b, "    asm volatile(\n        \"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n\"\n", n/2+2)
	fmt.Fprintf(&b, "        \"wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {\"\n", n)
	for i, l := range regLines {
		sep := ", "
		if i == len(regLines)-1 {
			sep = ""
		}
		fmt.Fprintf(&b, "        \"%s%s\"\n", l, sep)
	}
	fmt.Fprintf(&b, "        \"}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n\"\n", n/2, n/2+1)
	fmt.Fprintf(&b, "        : %s\n", strings.Join(outLines, ",\n          "))
	b.WriteString("        : \"l\"(da), \"l\"(db), \"r\"(1));\n#endif\n}\n\n")
	return b.String()
}

// wgmmaE4M3Func returns a device function name(float* d, uint64_t da,
// uint64_t db) issuing wgmma.mma_async m64n<n>k32 E4M3 → FP32 with D += A·B,
// both operands K-major in shared memory.
func wgmmaE4M3Func(name string, n int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "EG_DEV void %s(float* d, uint64_t da, uint64_t db) {\n", name)
	b.WriteString("#if defined(EG_HOST_EMU)\n")
	fmt.Fprintf(&b, "    eg_emu_wgmma_e4m3(d, da, db, %d);\n", n)
	b.WriteString("#else\n")
	regs := make([]string, n/2)
	outs := make([]string, n/2)
	for i := range regs {
		regs[i] = fmt.Sprintf("%%%d", i)
		outs[i] = fmt.Sprintf("\"+f\"(d[%d])", i)
	}
	var regLines, outLines []string
	for i := 0; i < len(regs); i += 16 {
		regLines = append(regLines, strings.Join(regs[i:min(i+16, len(regs))], ", "))
		outLines = append(outLines, strings.Join(outs[i:min(i+16, len(outs))], ", "))
	}
	fmt.Fprintf(&b, "    asm volatile(\n        \"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n\"\n", n/2+2)
	fmt.Fprintf(&b, "        \"wgmma.mma_async.sync.aligned.m64n%dk32.f32.e4m3.e4m3 {\"\n", n)
	for i, l := range regLines {
		sep := ", "
		if i == len(regLines)-1 {
			sep = ""
		}
		fmt.Fprintf(&b, "        \"%s%s\"\n", l, sep)
	}
	fmt.Fprintf(&b, "        \"}, %%%d, %%%d, p, 1, 1;\\n}\\n\"\n", n/2, n/2+1)
	fmt.Fprintf(&b, "        : %s\n", strings.Join(outLines, ",\n          "))
	b.WriteString("        : \"l\"(da), \"l\"(db), \"r\"(1));\n#endif\n}\n\n")
	return b.String()
}
