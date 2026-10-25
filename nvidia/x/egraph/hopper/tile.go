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

	BM, BN int // 64 or 128 (BN 256 with WGMMA); 0 means 128
	Stages int // 2 to 4; 0 means 3
}

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
	if g.WGMMA && g.BM != 128 {
		return g, fmt.Errorf("tile gemm %s: WGMMA needs BM 128", g.Name)
	}
	if g.Repacked && (!g.WGMMA || g.Weight != WeightNVFP4 || g.BN != 128) {
		return g, fmt.Errorf("tile gemm %s: Repacked requires WGMMA, WeightNVFP4 and BN 128", g.Name)
	}
	if g.BM != 64 && g.BM != 128 || g.BN != 64 && g.BN != 128 && !(g.WGMMA && g.BN == 256) || g.Stages < 2 || g.Stages > 4 {
		return g, fmt.Errorf("tile gemm %s: unsupported tile %dx%d, %d stages", g.Name, g.BM, g.BN, g.Stages)
	}
	if g.N < 1 || g.N%g.BN != 0 || g.K < 64 || g.K%64 != 0 {
		return g, fmt.Errorf("tile gemm %s: N %d must be a multiple of %d and K %d of 64", g.Name, g.N, g.BN, g.K)
	}
	if g.Weight != WeightBF16 && g.Weight != WeightNVFP4 {
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
	seen := map[string]bool{"x": true, "rows": true, "w": true, "w_sf": true, "w_gs": true, "offsets": true, "out": true, "ss": true, "m_total": true}
	for _, e := range g.Extras {
		if !cIdent(e.Name) || seen[e.Name] || e.DType != F32 && e.DType != BF16 || e.Broadcast < PerCol || e.Broadcast > Full {
			return g, fmt.Errorf("tile gemm %s: bad extra %+v", g.Name, e)
		}
		seen[e.Name] = true
	}
	return g, nil
}

// SharedBytes returns the dynamic shared memory one block uses.
func (g TileGEMM) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	if g.WGMMA {
		if g.Weight == WeightNVFP4 {
			if g.Repacked {
				return g.Stages * (g.BM*64*2 + g.BN*64*2 + 256*(16+8))
			}
			return g.Stages * (g.BM*64*2 + g.BN*64*2 + g.BN*(32+4))
		}
		return g.Stages * (g.BM + g.BN) * 64 * 2
	}
	b := g.BN * 72 * 2
	if g.Weight == WeightNVFP4 {
		b = g.BN * (48 + 4)
	}
	return g.Stages * (g.BM*72*2 + b)
}

func (g TileGEMM) params() []string {
	p := []string{"int m_total", "const uint16_t* __restrict__ x"}
	if g.Gather {
		p = append(p, "const int32_t* __restrict__ rows")
	}
	if g.Weight == WeightNVFP4 {
		if g.WGMMA && g.Repacked {
			p = append(p, "const uint32_t* __restrict__ w", "const uint8_t* __restrict__ w_sf", "const float* __restrict__ w_gs")
		} else {
			p = append(p, "const uint8_t* __restrict__ w", "const uint8_t* __restrict__ w_sf", "const float* __restrict__ w_gs")
		}
	} else {
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
	return p
}

// Grid returns the grid and block for m rows (the most rows of any expert
// when Grouped) and experts experts (1 unless Grouped).
func (g TileGEMM) Grid(m, experts int) (grid, block [3]int) {
	g, err := g.norm()
	if err != nil {
		return
	}
	return [3]int{g.N / g.BN, (m + g.BM - 1) / g.BM, experts}, [3]int{256, 1, 1}
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
	if g.WGMMA {
		return g.cudaWGMMA()
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: out = %s over x[M, %d] · W[%d, %d]ᵀ (%v), %dx%d tiles, %d stages.\n",
		g.Name, g.Epilogue, g.K, g.N, g.K, g.Weight, g.BM, g.BN, g.Stages)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(256) %s(\n    %s) {\n", g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, BM = %d, BN = %d, BK = 64, STAGES = %d, LD = 72;\n", g.N, g.K, g.BM, g.BN, g.Stages)
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
	b.WriteString(`    const int bm0 = row0 + blockIdx.y * BM, bn0 = blockIdx.x * BN;
    if (bm0 >= row1) return;
    EG_DYNAMIC_SMEM(uint16_t, sa); // [STAGES][BM][LD], then the W ring
`)
	if g.GLU {
		b.WriteString(`    // Tile row j: warp column j / WN holds WN/2 gate rows, then the matching up rows.
    auto wrow = [&](int j) {
        const int o = j % WN, c = blockIdx.x * (BN / 2) + (j / WN) * (WN / 2) + o % (WN / 2);
        return o < WN / 2 ? c : NOUT + c;
    };
`)
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
	fmt.Fprintf(&b, `    auto load = [&](int st, int kt) {
        for (int c = tid; c < BM * 8; c += 256) {
            const int r = c / 8, ch = c %% 8;
            eg_cp_async16(&sa[(st * BM + r) * LD + ch * 8], x + (size_t)%s * K + kt * BK + ch * 8);
        }
`, xrow)
	if g.Weight == WeightNVFP4 {
		b.WriteString(`        for (int c = tid; c < BN * 2; c += 256) {
            const int r = c / 2, ch = c % 2;
            eg_cp_async16(&sbq[(st * BN + r) * 48 + ch * 16], we + (size_t)wrow(r) * (K / 2) + kt * 32 + ch * 16);
        }
        for (int r = tid; r < BN; r += 256) eg_cp_async4(&sbs[(st * BN + r) * 4], se + (size_t)wrow(r) * (K / 16) + kt * 4);
`)
	} else {
		b.WriteString(`        for (int c = tid; c < BN * 8; c += 256) {
            const int r = c / 8, ch = c % 8;
            eg_cp_async16(&sb[(st * BN + r) * LD + ch * 8], we + (size_t)wrow(r) * K + kt * BK + ch * 8);
        }
`)
	}
	b.WriteString(`    };
    float acc[MT][NT][4] = {};
#pragma unroll
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
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            uint32_t af[MT][4], bf[NT][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                eg_ldmatrix_x4(af[mt], &sa[(st * BM + wm * WM + mt * 16 + lane % 16) * LD + ks * 16 + (lane / 16) * 8]);
`)
	if g.Weight == WeightNVFP4 {
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
	nts, col := "NT", "bn0 + wn * WN"
	if g.GLU {
		nts, col = "NT / 2", "blockIdx.x * (BN / 2) + wn * (WN / 2)"
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
        ss[(size_t)(bm0 + r) * (NOUT / 32) + blockIdx.x * E + j] = s;
    }
`, bno)
	}
	b.WriteString("}\n")
	return b.String(), nil
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
	fmt.Fprintf(&b, "    const dim3 grid(%d, (%s + %d) / %d, %s), block(256);\n", g.N/g.BN, rows, g.BM-1, g.BM, experts)
	fmt.Fprintf(&b, "    if (grid.y > 0 && grid.z > 0) EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", g.Name, strings.Join(names, ", "))
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
