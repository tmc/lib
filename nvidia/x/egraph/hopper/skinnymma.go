package hopper

import (
	"fmt"
	"strings"
)

// SkinnyMMA computes out[m, n] = Epilogue(Σ_k x[m, k] · W[n, k], extras...)
// for up to 32 BF16 activation rows with mma.sync m16n8k16 tensor-core
// tiles and FP32 accumulation: the projections of a short prefill chunk,
// where, as in decode, streaming the BF16 weights W [N, K] is the whole
// cost.
//
// Weights go from global memory straight into mma B fragments in
// registers. Each lane loads 16 bytes of a W row at a time, the access
// pattern of [SkinnyGEMM]: lane (g, q) of a warp reads row g of an 8-row
// tile at columns 8·(4c + q) of each 32·Chunks-column step. The k order the
// mma sees is a permutation of memory order applied to x and W alike, so
// no shuffle or shared memory stands between a weight load and its mma.
// x is staged in shared memory KChunk columns at a time, its rows padded
// with zeros to 16 or 32.
//
// A block of Warps warps owns Warps·Tiles·8 rows of W, each warp Tiles
// 8-row tiles. With GLU (as in [TileGEMM]) half of a warp's tiles are gate
// rows c and half the matching up rows c + N/2. With SplitK > 1, blocks
// along gridDim.y split K: each writes FP32 partials to a workspace, and
// the last block of a column group to finish (counted with an atomic that
// it then resets, so the kernel stays graph-replayable) sums them in a
// fixed order and applies the epilogue.
type SkinnyMMA struct {
	Name     string
	N, K     int
	MaxM     int    // 1 to 32
	Out      DType  // BF16 or F32
	Epilogue Lambda // $0 is the accumulator ($0, $1 with GLU); zero value means $0
	Extras   []Operand
	GLU      bool

	Warps  int // per block; 0 means 4
	Tiles  int // 8-row tiles per warp, even with GLU; 0 means 1, or 2 with GLU
	Chunks int // 16-byte loads per lane per tile and step; 0 means 2
	Steps  int // steps of weight loads in flight per warp; 0 means 2
	KChunk int // columns of x staged at once; 0 means at most 512
	// SplitK is the number of blocks along K; 0 chooses the fewest that
	// give at least three blocks per multiprocessor.
	SplitK int
	SMs    int // multiprocessors to fill; 0 means 132
}

// norm validates g and fills defaults.
func (g SkinnyMMA) norm() (SkinnyMMA, error) {
	if !cIdent(g.Name) {
		return g, fmt.Errorf("skinny mma: bad kernel name %q", g.Name)
	}
	if g.Out != BF16 && g.Out != F32 {
		return g, fmt.Errorf("skinny mma %s: output must be bf16 or f32, not %v", g.Name, g.Out)
	}
	if g.MaxM < 1 || g.MaxM > 32 {
		return g, fmt.Errorf("skinny mma %s: MaxM %d outside 1..32", g.Name, g.MaxM)
	}
	// GB10 streams Llama-3-8B's projections fastest (about 220 GB/s,
	// against 175 for a 64-row tile gemm) with small blocks: 4 warps of
	// one tile, 4 weight loads in flight per lane.
	if g.Warps == 0 {
		g.Warps = 4
	}
	if g.Tiles == 0 {
		g.Tiles = 1
		if g.GLU {
			g.Tiles = 2
		}
	}
	if g.Chunks == 0 {
		g.Chunks = 2
	}
	if g.Steps == 0 {
		g.Steps = 2
	}
	if g.SMs == 0 {
		g.SMs = 132
	}
	if g.Warps < 1 || g.Warps > 32 || g.Tiles < 1 || g.Chunks < 1 || g.Steps < 1 || g.SplitK < 0 || g.KChunk < 0 || g.SMs < 1 ||
		g.GLU && g.Tiles%2 != 0 {
		return g, fmt.Errorf("skinny mma %s: bad schedule", g.Name)
	}
	if g.Epilogue.root == nil {
		g.Epilogue = Identity
	}
	acc := 1
	if g.GLU {
		acc = 2
	}
	if g.Epilogue.Arity() > acc+len(g.Extras) {
		return g, fmt.Errorf("skinny mma %s: epilogue uses $%d, have %d accumulators and %d extras", g.Name, g.Epilogue.Arity()-1, acc, len(g.Extras))
	}
	seen := map[string]bool{"x": true, "w": true, "out": true, "ws": true, "m_total": true}
	for _, e := range g.Extras {
		if !cIdent(e.Name) || seen[e.Name] || e.DType != F32 && e.DType != BF16 || e.Broadcast < PerCol || e.Broadcast > Full {
			return g, fmt.Errorf("skinny mma %s: bad extra %+v", g.Name, e)
		}
		seen[e.Name] = true
	}
	rows := 8 * g.Warps * g.Tiles
	if g.N < 1 || g.N%rows != 0 {
		return g, fmt.Errorf("skinny mma %s: N %d not a multiple of %d rows per block", g.Name, g.N, rows)
	}
	unit := 32 * g.Chunks * g.Steps
	if g.K < unit || g.K%unit != 0 {
		return g, fmt.Errorf("skinny mma %s: K %d not a multiple of %d", g.Name, g.K, unit)
	}
	if g.SplitK == 0 {
		// The fewest splits of K into whole units that give three blocks
		// per multiprocessor, or one unit per block. On GB10 (48 SMs),
		// 128 blocks of 4096 rows stream about 4% faster split two ways;
		// 192 blocks gain nothing from a split.
		units, blocks := g.K/unit, g.N/rows
		g.SplitK = units
		for s := 1; s <= units; s++ {
			if units%s == 0 && blocks*s >= 3*g.SMs {
				g.SplitK = s
				break
			}
		}
	}
	if g.K%(g.SplitK*unit) != 0 {
		return g, fmt.Errorf("skinny mma %s: K %d does not split %d ways into multiples of %d", g.Name, g.K, g.SplitK, unit)
	}
	split := g.K / g.SplitK
	if g.KChunk == 0 {
		// The widest multiple of unit up to 512 that divides the split.
		g.KChunk = unit
		for c := unit; c <= max(512, unit) && c <= split; c += unit {
			if split%c == 0 {
				g.KChunk = c
			}
		}
	}
	if g.KChunk%unit != 0 || split%g.KChunk != 0 {
		return g, fmt.Errorf("skinny mma %s: KChunk %d must be a multiple of %d dividing %d", g.Name, g.KChunk, unit, split)
	}
	return g, nil
}

// paddedRows is MaxM rounded up to a whole number of 16-row mma tiles.
func (g SkinnyMMA) paddedRows() int { return (g.MaxM + 15) / 16 * 16 }

// Grid returns the launch grid and block size.
func (g SkinnyMMA) Grid() (grid, block [3]int, err error) {
	g, err = g.norm()
	if err != nil {
		return grid, block, err
	}
	return [3]int{g.N / (8 * g.Warps * g.Tiles), g.SplitK, 1}, [3]int{32 * g.Warps, 1, 1}, nil
}

// SharedBytes returns the dynamic shared memory one block uses: KChunk
// columns of the padded x rows, each row padded by 64 bytes so that the
// fragment loads of a quarter warp hit distinct banks.
func (g SkinnyMMA) SharedBytes() int {
	g, err := g.norm()
	if err != nil {
		return 0
	}
	return g.paddedRows() * (g.KChunk + 32) * 2
}

// WorkspaceBytes reports the bytes of zero-initialized workspace the kernel
// needs (FP32 partials [SplitK, MaxM, N], then an int32 counter per block
// column), or 0 without split-K. The counters must be zero before the
// first launch; the kernel leaves them zero.
func (g SkinnyMMA) WorkspaceBytes() int {
	g, err := g.norm()
	if err != nil || g.SplitK == 1 {
		return 0
	}
	return 4*g.SplitK*g.MaxM*g.N + 4*(g.N/(8*g.Warps*g.Tiles))
}

// params lists the kernel's parameters as C declarations.
func (g SkinnyMMA) params() []string {
	p := []string{"int m_total", "const uint16_t* __restrict__ x", "const uint16_t* __restrict__ w"}
	for _, e := range g.Extras {
		p = append(p, "const "+e.DType.cType()+"* __restrict__ "+e.Name)
	}
	p = append(p, g.Out.cType()+"* __restrict__ out")
	if g.SplitK > 1 {
		p = append(p, "float* __restrict__ ws")
	}
	return p
}

// CUDA returns the kernel source (without [Prelude]):
//
//	__global__ void Name(int m_total, x, w, extras..., out[, ws])
//
// launched over m_total ≤ MaxM rows with [SkinnyMMA.SharedBytes] of
// dynamic shared memory.
func (g SkinnyMMA) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: out = %s over x[M, %d] · W[%d, %d]ᵀ (bf16), M <= %d; %d warps of %d 8-row tiles, %d-way split K.\n",
		g.Name, g.Epilogue, g.K, g.N, g.K, g.MaxM, g.Warps, g.Tiles, g.SplitK)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(%d) %s(\n    %s) {\n", 32*g.Warps, g.Name, strings.Join(g.params(), ",\n    "))
	nout := "N"
	if g.GLU {
		nout = "N / 2"
	}
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, NOUT = %s, MP = %d, MT = MP / 16;\n", g.N, g.K, nout, g.paddedRows())
	fmt.Fprintf(&b, "    constexpr int WARPS = %d, NT = %d, C = %d, U = %d, SPLIT = %d, KS = K / SPLIT, KC = %d;\n",
		g.Warps, g.Tiles, g.Chunks, g.Steps, g.SplitK, g.KChunk)
	b.WriteString(`    constexpr int SS = 32 * C;    // columns per step: lane q holds 8 at 8·(4c + q), c < C
    constexpr int LDX = KC + 32;  // x row stride in shared memory
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
    EG_DYNAMIC_SMEM(uint16_t, xs); // [MP][LDX]
    // Tile nt of this warp is W rows wrow[nt] .. wrow[nt] + 7.
    int wrow[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
`)
	if g.GLU {
		b.WriteString(`        // Tiles nt and nt + NT/2 are the gate and up rows of the same 8 columns.
        const int c = ((blockIdx.x * WARPS + warp) * (NT / 2) + nt % (NT / 2)) * 8;
        wrow[nt] = nt < NT / 2 ? c : NOUT + c;
`)
	} else {
		b.WriteString("        wrow[nt] = ((blockIdx.x * WARPS + warp) * NT + nt) * 8;\n")
	}
	b.WriteString(`    }
    const int k0 = blockIdx.y * KS;
    float acc[MT][NT][4] = {};
    for (int kc = k0; kc < k0 + KS; kc += KC) {
        for (int s = 0; s < KC; s += U * SS) {
            eg_u32x4 wv[U][NT][C];
#pragma unroll
            for (int u = 0; u < U; ++u)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int c = 0; c < C; ++c)
                        wv[u][nt][c] = eg_ldg_stream(w + (size_t)(wrow[nt] + g) * K + kc + s + u * SS + (4 * c + q) * 8);
            if (s == 0) {
                // Stage x's next KC columns while the weight loads are in flight.
                __syncthreads();
                for (int i = tid; i < MP * (KC / 8); i += WARPS * 32) {
                    const int r = i / (KC / 8), ch = i % (KC / 8);
                    eg_u32x4 v = {0, 0, 0, 0};
                    if (r < m_total) v = eg_ldg16(x + (size_t)r * K + kc + ch * 8);
                    *reinterpret_cast<eg_u32x4*>(&xs[r * LDX + ch * 8]) = v;
                }
                __syncthreads();
            }
#pragma unroll
            for (int u = 0; u < U; ++u)
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const int col = s + u * SS + (4 * c + q) * 8;
                    eg_u32x4 xa[MT][2]; // rows 16mt + g and 16mt + g + 8
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                        for (int h = 0; h < 2; ++h)
                            xa[mt][h] = *reinterpret_cast<const eg_u32x4*>(&xs[(mt * 16 + g + 8 * h) * LDX + col]);
                    // A lane's 8 columns are two k16 steps of 4: mma k 2q, 2q+1
                    // take the first pair, k 2q+8, 2q+9 the second.
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const uint32_t a0[4] = {xa[mt][0].x, xa[mt][1].x, xa[mt][0].y, xa[mt][1].y};
                        const uint32_t a1[4] = {xa[mt][0].z, xa[mt][1].z, xa[mt][0].w, xa[mt][1].w};
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            const uint32_t b0[2] = {wv[u][nt][c].x, wv[u][nt][c].y};
                            const uint32_t b1[2] = {wv[u][nt][c].z, wv[u][nt][c].w};
                            eg_mma_bf16(acc[mt][nt], a0, b0, acc[mt][nt]);
                            eg_mma_bf16(acc[mt][nt], a1, b1, acc[mt][nt]);
                        }
                    }
                }
        }
    }
`)
	if g.SplitK > 1 {
		fmt.Fprintf(&b, "    constexpr int MAXM = %d;\n", g.MaxM)
		b.WriteString(`    // Split K: publish this block's partial; the last block of the column group reduces.
    int* const counters = reinterpret_cast<int*>(ws + (size_t)SPLIT * MAXM * N);
    __shared__ int is_last;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = mt * 16 + g + 8 * h;
                if (m < m_total)
                    *reinterpret_cast<float2*>(&ws[((size_t)blockIdx.y * MAXM + m) * N + wrow[nt] + 2 * q]) =
                        float2{acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]};
            }
    __threadfence();
    __syncthreads();
    if (tid == 0) is_last = atomicAdd(&counters[blockIdx.x], 1) == SPLIT - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = mt * 16 + g + 8 * h;
                if (m >= m_total) continue;
                float2 v = {0.0f, 0.0f};
                for (int sp = 0; sp < SPLIT; ++sp) {
                    const float2 p = eg_ld_cg2(&ws[((size_t)sp * MAXM + m) * N + wrow[nt] + 2 * q]);
                    v.x += p.x;
                    v.y += p.y;
                }
                acc[mt][nt][2 * h] = v.x;
                acc[mt][nt][2 * h + 1] = v.y;
            }
    if (tid == 0) counters[blockIdx.x] = 0;
`)
	}
	nts := "NT"
	if g.GLU {
		nts = "NT / 2"
	}
	fmt.Fprintf(&b, `#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < %s; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int m = mt * 16 + g + 8 * h;
                if (m >= m_total) continue;
                float yv[2]; // a row's two adjacent columns, stored together
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float v = acc[mt][nt][2 * h + j];
`, nts)
	args := []string{"v"}
	if g.GLU {
		b.WriteString("                    const float u = acc[mt][nt + NT / 2][2 * h + j];\n")
		args = append(args, "u")
	}
	for _, e := range g.Extras {
		args = append(args, e.load("m", "(wrow[nt] + 2 * q + j)", "NOUT"))
	}
	expr, err := g.Epilogue.CUDA(args...)
	if err != nil {
		return "", err
	}
	fmt.Fprintf(&b, "                    yv[j] = %s;\n", expr)
	b.WriteString(`                }
                eg_store<2>(out + (size_t)m * NOUT + wrow[nt] + 2 * q, yv);
            }
}
`)
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(int m, <params...>, cudaStream_t stream)
//
// where m ≤ MaxM is x's row count.
func (g SkinnyMMA) Launcher() (string, error) {
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
	grid, block, _ := g.Grid()
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int m, %s, cudaStream_t stream) {\n", g.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    constexpr int smem = %d;\n", g.SharedBytes())
	fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", g.Name)
	fmt.Fprintf(&b, "    const dim3 grid(%d, %d), block(%d);\n", grid[0], grid[1], block[0])
	fmt.Fprintf(&b, "    if (m > 0) EG_LAUNCH(%s, grid, block, smem, stream, %s);\n}\n", g.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (g SkinnyMMA) Source() (string, error) {
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

// Reference computes the [m, N] output ([m, N/2] with GLU) in float64,
// with the epilogue in FP32.
func (g SkinnyMMA) Reference(m int, x []uint16, w []uint16, extras [][]float32) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if m < 1 || m > g.MaxM || len(x) != m*g.K || len(w) != g.N*g.K || len(extras) != len(g.Extras) {
		return nil, fmt.Errorf("skinny mma %s reference: bad inputs", g.Name)
	}
	rows := make([]int, m)
	for i := range rows {
		rows[i] = i
	}
	xRow := func(i int) []uint16 { return x[i*g.K : (i+1)*g.K] }
	out := make([]float32, m*g.N)
	if !g.GLU {
		if err := gemmRows(g.N, g.K, WeightBF16, GEMMWeights{BF16: w}, xRow, rows, g.Epilogue, g.Extras, extras, g.Out, out); err != nil {
			return nil, fmt.Errorf("skinny mma %s reference: %w", g.Name, err)
		}
		return out, nil
	}
	if err := gemmRows(g.N, g.K, WeightBF16, GEMMWeights{BF16: w}, xRow, rows, Identity, nil, nil, F32, out); err != nil {
		return nil, fmt.Errorf("skinny mma %s reference: %w", g.Name, err)
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

// UseSkinnyMMA replaces each tile gemm of p over at most 32 rows that a
// [SkinnyMMA] computes (BF16 weights, not Grouped, Gather, SumSq or Stream-K)
// with one scheduled for sms multiprocessors. Its arguments are unchanged;
// a split-K kernel also takes a workspace of [SkinnyMMA.WorkspaceBytes]
// after them.
func (p *Plan) UseSkinnyMMA(sms int) {
	for i := range p.Kernels {
		k := &p.Kernels[i]
		t := k.Tile
		if t == nil || k.M > 32 || t.Weight != WeightBF16 || t.Grouped || t.Gather || t.SumSq || t.StreamK > 0 {
			continue
		}
		g := SkinnyMMA{Name: t.Name, N: t.N, K: t.K, MaxM: k.M, Out: t.Out, Epilogue: t.Epilogue, Extras: t.Extras, GLU: t.GLU, SMs: sms}
		if g, err := g.norm(); err == nil {
			k.MMA, k.Tile = &g, nil
		}
	}
}
