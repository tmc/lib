package hopper

import (
	"fmt"
	"strings"
)

// warpSpecHelpers are the mbarrier, bulk-copy and register-budget helpers of
// a warp-specialized kernel.
const warpSpecHelpers = `#if !defined(EG_HOST_EMU)
EG_DEV uint32_t eg_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#endif

// eg_mbar_init initializes the mbarrier at bar to complete each phase after count arrivals.
EG_DEV void eg_mbar_init(uint64_t* bar, uint32_t count) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_init(bar, (int)count);
#else
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(eg_smem_u32(bar)), "r"(count) : "memory");
#endif
}

// eg_fence_mbar_init makes initialized mbarriers visible to the async proxy.
EG_DEV void eg_fence_mbar_init() {
#if !defined(EG_HOST_EMU)
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

EG_DEV void eg_mbar_arrive(uint64_t* bar) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_arrive(bar);
#else
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(eg_smem_u32(bar)) : "memory");
#endif
}

// eg_mbar_arrive_expect_tx arrives on bar and adds bytes to the transaction
// bytes its phase waits for.
EG_DEV void eg_mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_arrive_expect_tx(bar, bytes);
#else
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(eg_smem_u32(bar)), "r"(bytes) : "memory");
#endif
}

// eg_mbar_wait returns once the phase of bar with parity parity has completed.
EG_DEV void eg_mbar_wait(uint64_t* bar, uint32_t parity) {
#if defined(EG_HOST_EMU)
    eg_emu_mbar_wait(bar, (int)parity);
#else
    asm volatile(
        "{\n.reg .pred p;\nEG_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra EG_MBAR_WAIT_%=;\n}\n" ::"r"(eg_smem_u32(bar)), "r"(parity) : "memory");
#endif
}

// eg_bulk_copy copies bytes, a multiple of 16, from global to shared memory
// with cp.async.bulk, completing them on bar's transaction bytes.
EG_DEV void eg_bulk_copy(void* smem, const void* gmem, uint32_t bytes, uint64_t* bar) {
#if defined(EG_HOST_EMU)
    eg_emu_cp_async_bulk(smem, gmem, bytes, bar);
#else
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(eg_smem_u32(smem)), "l"(gmem), "r"(bytes), "r"(eg_smem_u32(bar)) : "memory");
#endif
}

// eg_smem_align1024 rounds p up to a 1024-byte aligned shared address, as
// 128B-swizzled wgmma operands need.
EG_DEV unsigned char* eg_smem_align1024(unsigned char* p) {
#if defined(EG_HOST_EMU)
    return p + (1024 - (size_t)(p - eg_emu_dynamic_smem()) % 1024) % 1024;
#else
    return p + ((1024u - (eg_smem_u32(p) & 1023u)) & 1023u);
#endif
}

// eg_setmaxnreg_dec and eg_setmaxnreg_inc set the calling warpgroup's
// registers per thread to R.
template <int R> EG_DEV void eg_setmaxnreg_dec() {
#if !defined(EG_HOST_EMU)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R));
#endif
}

template <int R> EG_DEV void eg_setmaxnreg_inc() {
#if !defined(EG_HOST_EMU)
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R));
#endif
}

`

// cudaWarpSpecialized emits the warp-specialized form of g (normalized,
// g.WarpSpecialized set).
//
// The 384-thread block is one producer warpgroup and two consumer
// warpgroups sharing a Stages-deep ring of 128B-swizzled stages. For each
// k step, producer thread 0 posts the stage's byte count on full[stage] and
// every producer thread issues cp.async.bulk copies of 16-byte chunks into
// their swizzled places (and, for NVFP4, of the fragment words and scales),
// after waiting on empty[stage] for the consumers to release the stage's
// previous use. Each consumer warp waits on full[stage], issues its
// warpgroup's wgmma, and, once the next step's wgmma is issued and the
// step's group has retired, releases the stage with one arrival on
// empty[stage] (8 per phase).
func (g TileGEMM) cudaWarpSpecialized() (string, error) {
	nvfp4 := g.Weight == WeightNVFP4
	var b strings.Builder
	if nvfp4 {
		b.WriteString(nvfp4WGMMAHelpers)
	}
	b.WriteString(wgmmaSwizzleHelpers)
	b.WriteString(warpSpecHelpers)
	wg := g.Name + "_wgmma"
	switch g.Weight {
	case WeightNVFP4:
		b.WriteString(wgmmaRSFunc(wg, g.BM))
	case WeightE4M3:
		b.WriteString(wgmmaE4M3Func(wg, g.BN))
	default:
		b.WriteString(wgmmaFunc(wg, g.BN))
	}
	bx := "blockIdx.x"
	if g.GroupM > 1 {
		bx = "tile_n"
	}
	bk, chunk, kstep := 64, 8, 16
	if g.Weight == WeightE4M3 {
		bk, chunk, kstep = 128, 16, 32
	}
	fmt.Fprintf(&b, "// %s: out = %s over x[M, %d] · W[%d, %d]ᵀ (%v), %dx%d tiles, %d stages, warp-specialized wgmma.\n",
		g.Name, g.Epilogue, g.K, g.N, g.K, g.Weight, g.BM, g.BN, g.Stages)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(384, 1) %s(\n    %s) {\n", g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = %d, STAGES = %d, KT = K / BK;\n", g.N, g.K, g.BM, g.BN, bk, g.Stages)
	if g.GLU {
		b.WriteString("    constexpr int NOUT = N / 2;\n")
	} else {
		b.WriteString("    constexpr int NOUT = N;\n")
	}
	if nvfp4 {
		fmt.Fprintf(&b, `    // Stage: x [BM][64] BF16, then 256 threads' fragment words and scales.
    constexpr int XBYTES = BM * 128, WBYTES = 256 * 16, SBYTES = %s;
    constexpr int TX = XBYTES + WBYTES + SBYTES, STAGE = (TX + 1023) / 1024 * 1024;
`, scaleSize(g))
	} else {
		b.WriteString(`    // Stage: x [BM][BK], then W [BN][BK], in 128-byte swizzled rows.
    constexpr int TX = (BM + BN) * 128, STAGE = TX;
`)
	}
	b.WriteString("    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;\n")
	if g.Grouped {
		b.WriteString("    const int e = blockIdx.z, row0 = offsets[e], row1 = offsets[e + 1];\n")
	} else {
		b.WriteString("    const int e = 0, row0 = 0, row1 = m_total;\n")
	}
	tileCoord(&b, g.GroupM)
	b.WriteString(`    if (bm0 >= row1) return;
    EG_DYNAMIC_SMEM(unsigned char, smem_raw);
    unsigned char* const smem = eg_smem_align1024(smem_raw); // [STAGES][STAGE], then full[STAGES], empty[STAGES]
    uint64_t* const full = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE);
    uint64_t* const empty = full + STAGES;
`)
	switch g.Weight {
	case WeightNVFP4:
		fmt.Fprintf(&b, `    const uint32_t* const we = w + ((size_t)e * (N / 128) + %s) * KT * 256 * 4;
    const uint8_t* const se = w_sf + ((size_t)e * (N / 128) + %s) * KT * %s;
    const float gs = w_gs[e] * 256.0f;
`, bx, bx, scaleSize(g))
	case WeightE4M3:
		b.WriteString("    const uint8_t* const we = w + (size_t)e * N * K;\n")
	default:
		b.WriteString("    const uint16_t* const we = w + (size_t)e * N * K;\n")
	}
	if !nvfp4 {
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
		xrow = "srow[r]"
		b.WriteString(`    __shared__ int srow[BM];
    for (int r = tid; r < BM; r += 384) srow[r] = rows[min(bm0 + r, row1 - 1)];
`)
	}
	b.WriteString(`    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            eg_mbar_init(&full[s], 1);
            eg_mbar_init(&empty[s], 8);
        }
        eg_fence_mbar_init();
    }
    __syncthreads();

    if (warp < 4) {
        // Producer warpgroup.
        eg_setmaxnreg_dec<40>();
        for (int kt = 0; kt < KT; ++kt) {
            const int s = kt % STAGES;
            if (kt >= STAGES) eg_mbar_wait(&empty[s], (kt / STAGES - 1) & 1);
            unsigned char* const sa = smem + s * STAGE;
`)
	if g.VectorActivationLoads {
		fmt.Fprintf(&b, `            for (int c = tid; c < BM * 8; c += 128) {
                const int r = c / 8, ch = c %% 8;
                const eg_u32x4 value = *reinterpret_cast<const eg_u32x4*>(x + (size_t)%s * K + kt * BK + ch * 8);
                *reinterpret_cast<eg_u32x4*>(sa + eg_swizzle_128b(r * 128 + ch * 16)) = value;
            }
            eg_fence_proxy_async();
#if defined(EG_HOST_EMU)
            eg_emu_bar_sync(1, 128);
#else
            asm volatile("barrier.cta.sync 1, 128;" ::: "memory");
#endif
            if (tid == 0) eg_mbar_arrive_expect_tx(&full[s], WBYTES + SBYTES);
`, xrow)
	} else {
		b.WriteString("            if (tid == 0) eg_mbar_arrive_expect_tx(&full[s], TX);\n")
		fmt.Fprintf(&b, `            for (int c = tid; c < BM * 8; c += 128) {
                const int r = c / 8, ch = c %% 8;
                eg_bulk_copy(sa + eg_swizzle_128b(r * 128 + ch * 16), x + (size_t)%s * K + kt * BK + ch * %d, 16, &full[s]);
            }
`, xrow, chunk)
	}
	if nvfp4 {
		fmt.Fprintf(&b, `            if (tid == 0) {
                eg_bulk_copy(sa + XBYTES, we + (size_t)kt * 256 * 4, WBYTES, &full[s]);
                eg_bulk_copy(sa + XBYTES + WBYTES, se + (size_t)kt * %s, SBYTES, &full[s]);
            }
`, scaleSize(g))
	} else {
		fmt.Fprintf(&b, `            unsigned char* const sb = sa + BM * 128;
            for (int c = tid; c < BN * 8; c += 128) {
                const int r = c / 8, ch = c %% 8;
                eg_bulk_copy(sb + eg_swizzle_128b(r * 128 + ch * 16), we + (size_t)wrow(r) * K + kt * BK + ch * %d, 16, &full[s]);
            }
`, chunk)
	}
	b.WriteString(`        }
        return;
    }

    // Consumer warpgroups.
    eg_setmaxnreg_inc<232>();
    const int wgi = warp / 4 - 1;
`)
	if nvfp4 {
		fmt.Fprintf(&b, `    const int ct = tid - 128; // 0..255 over both consumer warpgroups, as RepackNVFP4WGMMA numbers threads
    float d[BM / 2] = {};
    uint32_t a[2][4][4]; // two steps' A fragments: a step's stay untouched while its wgmma runs
    // step decodes stage s's fragments into av and issues its wgmmas.
    auto step = [&](int kt, uint32_t (&av)[4][4]) {
        const int s = kt %% STAGES;
        eg_mbar_wait(&full[s], (kt / STAGES) & 1);
        const unsigned char* const sa = smem + s * STAGE;
        const eg_u32x4 q = *reinterpret_cast<const eg_u32x4*>(sa + XBYTES + ct * 16);
        const eg_u32x2 sc = *reinterpret_cast<const eg_u32x2*>(sa + XBYTES + WBYTES + %s * 8);
        const uint32_t qs[4] = {q.x, q.y, q.z, q.w};
`, scaleIndex(g))
		if g.DequantWords == 2 {
			b.WriteString(`#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint32_t qpair[2] = {qs[2 * p], qs[2 * p + 1]};
            uint32_t decoded[8];
            eg_dequant_2w(qpair, decoded);
#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int ks = 2 * p + sub;
                const uint32_t sw = (ks < 2 ? sc.x : sc.y) >> (16 * (ks & 1));
                const uint32_t s0 = eg_scale_e4m3_bf16x2(sw), s1 = eg_scale_e4m3_bf16x2(sw >> 8);
#pragma unroll
                for (int j = 0; j < 4; ++j) av[ks][j] = eg_mul_bf16x2(decoded[4 * sub + j], (j & 1) ? s1 : s0);
            }
        }
`)
		} else {
			b.WriteString(`#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // Scale byte 2*ks + h is row (lane/4) + 8h's scale for this 16 k.
            const uint32_t sw = (ks < 2 ? sc.x : sc.y) >> (16 * (ks & 1));
            const uint32_t s0 = eg_scale_e4m3_bf16x2(sw), s1 = eg_scale_e4m3_bf16x2(sw >> 8);
#pragma unroll
            for (int j = 0; j < 4; ++j) av[ks][j] = eg_mul_bf16x2(eg_wgmma_decode_nibble(qs[ks], j), (j & 1) ? s1 : s0);
        }
`)
		}
		fmt.Fprintf(&b, `        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) %s(d, av[ks], eg_wgmma_desc_128b(sa + ks * 32, 1024));
        eg_wgmma_commit();
        // The previous step's group has retired: release its stage.
        eg_wgmma_wait<1>();
        if (kt > 0) {
            __syncwarp();
            if (lane == 0) eg_mbar_arrive(&empty[(kt - 1) %% STAGES]);
        }
    };
    for (int kt = 0; kt < KT; kt += 2) {
        step(kt, a[0]);
        if (kt + 1 < KT) step(kt + 1, a[1]);
    }
`, wg)
	} else {
		fmt.Fprintf(&b, `    float d[BN / 2] = {};
    for (int kt = 0; kt < KT; ++kt) {
        const int s = kt %% STAGES;
        eg_mbar_wait(&full[s], (kt / STAGES) & 1);
        const unsigned char* const sa = smem + s * STAGE;
        const unsigned char* const sb = sa + BM * 128;
        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < BK / %d; ++ks)
            %s(d, eg_wgmma_desc_128b(sa + wgi * 8 * 1024 + ks * 32, 1024), eg_wgmma_desc_128b(sb + ks * 32, 1024));
        eg_wgmma_commit();
        // The previous step's group has retired: release its stage.
        eg_wgmma_wait<1>();
        if (kt > 0) {
            __syncwarp();
            if (lane == 0) eg_mbar_arrive(&empty[(kt - 1) %% STAGES]);
        }
    }
`, kstep, wg)
	}
	b.WriteString(`    eg_wgmma_wait<0>();
    __syncwarp();
    if (lane == 0) eg_mbar_arrive(&empty[(KT - 1) % STAGES]);
`)
	if nvfp4 {
		if err := g.swapEpilogue(&b, bx); err != nil {
			return "", err
		}
	} else if err := g.wgmmaEpilogue(&b, bx); err != nil {
		return "", err
	}
	b.WriteString("}\n")
	return b.String(), nil
}

// swapEpilogue emits the epilogue of the NVFP4 warp-specialized kernel,
// whose accumulator is the transposed tile: consumer warpgroup wgi holds
// fragment rows 64*wgi .. 64*wgi+63 of the RepackNVFP4WGMMA weight tile
// against BM tokens. A thread holds one output column per token, and the
// lane four away the next column, so each pair of lanes trades values to
// store two adjacent columns at once: the lane with even lane/4 those of
// its first token, its partner those of its second.
func (g TileGEMM) swapEpilogue(b *strings.Builder, bx string) error {
	b.WriteString(`    // Accumulator d[nt*4 + i]: fragment row 64*wgi + (warp % 4)*16 + lane/4 + 8*(i/2), token nt*8 + 2*(lane%4) + (i&1).
    const bool even = (lane / 4) % 2 == 0;
#pragma unroll
    for (int nt = 0; nt < BM / 8; ++nt) {
        const int m0 = bm0 + nt * 8 + 2 * (lane % 4);
`)
	args := []string{"v"}
	var col string
	if g.GLU {
		// Fragment rows 16b..16b+7 are gate columns 8b.., rows 16b+8.. the matching up columns.
		col = bx + " * 64 + (wgi * 4 + warp % 4) * 8 + lane / 4"
		b.WriteString("        float y[2];\n#pragma unroll\n        for (int i = 0; i < 2; ++i) {\n")
		b.WriteString("            const float v = d[nt * 4 + i] * gs, u = d[nt * 4 + 2 + i] * gs;\n")
		args = append(args, "u")
	} else {
		col = "bn0 + wgi * 64 + (warp % 4) * 16 + lane / 4 + 8 * (i / 2)"
		b.WriteString("        float y[4];\n#pragma unroll\n        for (int i = 0; i < 4; ++i) {\n")
		b.WriteString("            const float v = d[nt * 4 + i] * gs;\n")
	}
	fmt.Fprintf(b, "            const int m = m0 + (i & 1), n = %s;\n", col)
	for _, e := range g.Extras {
		args = append(args, e.load("m", "n", "NOUT"))
	}
	expr, err := g.Epilogue.CUDA(args...)
	if err != nil {
		return err
	}
	fmt.Fprintf(b, "            y[i] = m < row1 ? %s : 0.0f;\n        }\n", expr)
	half := func(lo, hi, n string) {
		fmt.Fprintf(b, `        {
            const float mine = even ? %s : %s;
            const float theirs = __shfl_xor_sync(0xffffffffu, even ? %s : %s, 4);
            const float yv[2] = {even ? mine : theirs, even ? theirs : mine};
            const int m = m0 + (even ? 0 : 1);
            if (m < row1) eg_store<2>(out + (size_t)m * NOUT + %s - (even ? 0 : 1), yv);
        }
`, lo, hi, hi, lo, n)
	}
	if g.GLU {
		half("y[0]", "y[1]", col)
	} else {
		half("y[0]", "y[1]", "bn0 + wgi * 64 + (warp % 4) * 16 + lane / 4")
		half("y[2]", "y[3]", "bn0 + wgi * 64 + (warp % 4) * 16 + lane / 4 + 8")
	}
	b.WriteString("    }\n")
	return nil
}

// wgmmaRSFunc returns a device function name(float* d, const uint32_t a[4],
// uint64_t db) issuing wgmma.mma_async m64n<n>k16 BF16 → FP32 with D += A·B,
// A from registers in the mma.sync fragment order and B K-major in shared
// memory.
func wgmmaRSFunc(name string, n int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "EG_DEV void %s(float* d, const uint32_t a[4], uint64_t db) {\n", name)
	b.WriteString("#if defined(EG_HOST_EMU)\n")
	fmt.Fprintf(&b, "    eg_emu_wgmma_bf16(d, a, db, %d);\n", n)
	b.WriteString("#else\n")
	var regLines, outLines []string
	for i := 0; i < n/2; i += 16 {
		var regs, outs []string
		for j := i; j < min(i+16, n/2); j++ {
			regs = append(regs, fmt.Sprintf("%%%d", j))
			outs = append(outs, fmt.Sprintf("\"+f\"(d[%d])", j))
		}
		regLines = append(regLines, strings.Join(regs, ", "))
		outLines = append(outLines, strings.Join(outs, ", "))
	}
	r := n / 2
	fmt.Fprintf(&b, "    asm volatile(\n        \"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n\"\n", r+5)
	fmt.Fprintf(&b, "        \"wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {\"\n", n)
	for i, l := range regLines {
		sep := ", "
		if i == len(regLines)-1 {
			sep = ""
		}
		fmt.Fprintf(&b, "        \"%s%s\"\n", l, sep)
	}
	fmt.Fprintf(&b, "        \"}, {%%%d, %%%d, %%%d, %%%d}, %%%d, p, 1, 1, 0;\\n}\\n\"\n", r, r+1, r+2, r+3, r+4)
	fmt.Fprintf(&b, "        : %s\n", strings.Join(outLines, ",\n          "))
	b.WriteString("        : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(db), \"r\"(1));\n#endif\n}\n\n")
	return b.String()
}

func scaleSize(g TileGEMM) string {
	if g.CompactScales {
		return "64 * 8"
	}
	return "256 * 8"
}

func scaleIndex(g TileGEMM) string {
	if g.CompactScales {
		return "(ct / 4)"
	}
	return "ct"
}
