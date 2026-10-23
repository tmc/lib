package hopper

import (
	"fmt"
	"strings"
)

// cudaWGMMA emits the WGMMA form of g (normalized, g.WGMMA set).
//
// x and W tiles sit in shared memory in the no-swizzle K-major layout of
// 8-row × 16-byte core matrices: element (r, k) of a 64-wide K slice is at
// byte ((r/8)·8 + k/8)·128 + (r%8)·16 + (k%8)·2, so the descriptors have
// leading (K) byte offset 128 and stride (8-row) byte offset 1024.
func (g TileGEMM) cudaWGMMA() (string, error) {
	var b strings.Builder
	wg := g.Name + "_wgmma"
	b.WriteString(wgmmaFunc(wg, g.BN))
	fmt.Fprintf(&b, "// %s: out = %s over x[M, %d] · W[%d, %d]ᵀ (%v), %dx%d tiles, %d stages, wgmma.\n",
		g.Name, g.Epilogue, g.K, g.N, g.K, g.Weight, g.BM, g.BN, g.Stages)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n", g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = 64, STAGES = %d, KT = K / BK;\n", g.N, g.K, g.BM, g.BN, g.Stages)
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
	b.WriteString(`    const int bm0 = row0 + blockIdx.y * BM, bn0 = blockIdx.x * BN;
    if (bm0 >= row1) return;
    EG_DYNAMIC_SMEM(unsigned char, smem); // [STAGES][BM*64 + BN*64] BF16 in core-matrix order
    constexpr int STAGE = (BM + BN) * BK * 2;
    const uint16_t* const we = w + (size_t)e * N * K;
`)
	if g.GLU {
		b.WriteString(`    // Tile row j < BN/2 is gate column j, row BN/2 + j its up column.
    auto wrow = [&](int j) {
        const int c = blockIdx.x * (BN / 2) + j % (BN / 2);
        return j < BN / 2 ? c : NOUT + c;
    };
`)
	} else {
		b.WriteString("    auto wrow = [&](int j) { return bn0 + j; };\n")
	}
	xrow := "gm"
	if g.Gather {
		xrow = "rows[gm]"
	}
	fmt.Fprintf(&b, `    auto core = [](int r, int ch) { return ((r / 8) * 8 + ch) * 128 + (r %% 8) * 16; };
    auto load = [&](int st, int kt) {
        unsigned char* const sa = smem + st * STAGE;
        unsigned char* const sb = sa + BM * BK * 2;
        for (int c = tid; c < BM * 8; c += 256) {
            const int r = c / 8, ch = c %% 8, gm = min(bm0 + r, row1 - 1);
            eg_cp_async16(sa + core(r, ch), x + (size_t)%s * K + kt * BK + ch * 8);
        }
        for (int c = tid; c < BN * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(sb + core(r, ch), we + (size_t)wrow(r) * K + kt * BK + ch * 8);
        }
    };
    float d[BN / 2] = {};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load(s, s);
        eg_cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        eg_cp_async_wait<STAGES - 2>();
        eg_fence_proxy_async();
        __syncthreads();
        if (kt + STAGES - 1 < KT) load((kt + STAGES - 1) %% STAGES, kt + STAGES - 1);
        eg_cp_async_commit();
        const unsigned char* const sa = smem + (kt %% STAGES) * STAGE;
        const unsigned char* const sb = sa + BM * BK * 2;
        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks)
            %s(d, eg_wgmma_desc(sa + wgi * 8 * 1024 + ks * 256, 128, 1024), eg_wgmma_desc(sb + ks * 256, 128, 1024));
        eg_wgmma_commit();
        eg_wgmma_wait<0>();
    }
    eg_cp_async_wait<0>();
`, xrow, wg)
	if g.Weight == WeightNVFP4 {
		return "", fmt.Errorf("tile gemm %s: WGMMA with NVFP4 is not implemented", g.Name)
	}
	nts, col := "BN / 8", "bn0"
	if g.GLU {
		nts, col = "BN / 16", "blockIdx.x * (BN / 2)"
	}
	fmt.Fprintf(&b, `    // Accumulator d[nt*4 + i]: row (warp %% 4)*16 + lane/4 + 8*(i/2), column nt*8 + 2*(lane%%4) + (i&1).
#pragma unroll
    for (int nt = 0; nt < %s; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = bm0 + wgi * 64 + (warp %% 4) * 16 + lane / 4 + 8 * (i / 2);
            const int n = %s + nt * 8 + 2 * (lane %% 4) + (i & 1);
            if (m >= row1) continue;
            const float v = d[nt * 4 + i];
`, nts, col)
	args := []string{"v"}
	if g.GLU {
		b.WriteString("            const float u = d[(nt + BN / 16) * 4 + i];\n")
		args = append(args, "u")
	}
	for _, e := range g.Extras {
		args = append(args, e.load("m", "n", "NOUT"))
	}
	expr, err := g.Epilogue.CUDA(args...)
	if err != nil {
		return "", err
	}
	if g.Out == BF16 {
		fmt.Fprintf(&b, "            out[(size_t)m * NOUT + n] = eg_f32_to_bf16(%s);\n", expr)
	} else {
		fmt.Fprintf(&b, "            out[(size_t)m * NOUT + n] = %s;\n", expr)
	}
	b.WriteString("        }\n}\n")
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
