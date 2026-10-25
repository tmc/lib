package hopper

import (
	"fmt"
	"strings"
)

// A WeightFormat is how a GEMM's weight matrix W [N, K] is stored.
type WeightFormat uint8

const (
	// WeightBF16 is w: BF16 [N, K].
	WeightBF16 WeightFormat = iota + 1
	// WeightBF16HiLo is w_hi, w_lo: BF16 [N, K] each, with W = hi + lo as
	// split by [SplitBF16]. It carries FP32 router weights at BF16 speed
	// (moe8_bf16_proj/reference.py:28-29).
	WeightBF16HiLo
	// WeightNVFP4 is w: E2M1 [N, K/2] (element 2i in the low nibble), w_sf:
	// E4M3 [N, K/16] block scales, and w_gs: one FP32 global scale, with
	// W = e2m1 · sf · gs (moe8_moe_w4a16/nvfp4.py:90-97).
	WeightNVFP4
)

func (f WeightFormat) String() string {
	switch f {
	case WeightBF16:
		return "bf16"
	case WeightBF16HiLo:
		return "bf16hilo"
	case WeightNVFP4:
		return "nvfp4"
	}
	return fmt.Sprintf("WeightFormat(%d)", uint8(f))
}

// chunk is the number of K values one lane loads in one 16-byte weight load.
func (f WeightFormat) chunk() int {
	if f == WeightNVFP4 {
		return 32
	}
	return 8
}

// A Broadcast says how an epilogue operand indexes the [M, N] output.
type Broadcast uint8

const (
	PerCol Broadcast = iota + 1 // operand [N], indexed by n
	PerRow                      // operand [M], indexed by m
	Full                        // operand [M, N]
)

// An Operand is an extra epilogue input, available to the epilogue lambda as
// $1, $2, ... in order.
type Operand struct {
	Name      string
	DType     DType // F32 or BF16
	Broadcast Broadcast
}

func (o Operand) index(m, n, cols string) string {
	switch o.Broadcast {
	case PerCol:
		return n
	case PerRow:
		return m
	}
	return m + " * " + cols + " + " + n
}

func (o Operand) load(m, n, cols string) string {
	e := o.Name + "[" + o.index(m, n, cols) + "]"
	if o.DType == BF16 {
		return "eg_bf16_to_f32(" + e + ")"
	}
	return e
}

// SkinnyGEMM computes out[m, n] = Epilogue(Σ_k x[m, k] · W[n, k], extras...)
// for M ≤ MaxM BF16 activation rows with FP32 accumulation: the decode-time
// projections, where streaming W from HBM is the whole cost.
//
// Each block owns RowsPerBlock rows of W; its warps split K, each lane
// streams 16 bytes of every owned row per step with an L1-bypassing load, and
// the warps reduce through shared memory. With SplitK > 1, blocks along
// gridDim.y split K further: each writes FP32 partials to a workspace, and
// the last block to finish a column group (counted with an atomic that it
// then resets, so the kernel stays graph-replayable) sums them in a fixed
// order and applies the epilogue.
type SkinnyGEMM struct {
	Name     string
	N, K     int
	MaxM     int // kernels are instantiated for M = 1..MaxM
	Weight   WeightFormat
	Out      DType  // BF16 or F32
	Epilogue Lambda // $0 is the accumulator; zero value means $0
	Extras   []Operand

	RowsPerBlock int // 0 chooses
	Warps        int // per block; 0 means 8
	SplitK       int // 0 or 1 means no split
	Unroll       int // weight loads in flight per lane per row; 0 means 4

	// Prefetch, when at least 1, unrolls the K loop in batches of Prefetch
	// steps per warp (or the whole split when it has at most Prefetch steps per
	// warp) and issues every weight load of the batch into registers before the
	// activation loads and FMA accumulation.
	Prefetch int
}

// norm validates g and fills defaults.
func (g SkinnyGEMM) norm() (SkinnyGEMM, error) {
	if !cIdent(g.Name) {
		return g, fmt.Errorf("skinny gemm: bad kernel name %q", g.Name)
	}
	if g.Weight < WeightBF16 || g.Weight > WeightNVFP4 {
		return g, fmt.Errorf("skinny gemm %s: bad weight format", g.Name)
	}
	if g.Out != BF16 && g.Out != F32 {
		return g, fmt.Errorf("skinny gemm %s: output must be bf16 or f32, not %v", g.Name, g.Out)
	}
	if g.MaxM < 1 || g.MaxM > 16 {
		return g, fmt.Errorf("skinny gemm %s: MaxM %d outside 1..16", g.Name, g.MaxM)
	}
	if g.Warps == 0 {
		g.Warps = 8
	}
	if g.SplitK == 0 {
		g.SplitK = 1
	}
	if g.Unroll == 0 {
		g.Unroll = 4
	}
	if g.Epilogue.root == nil {
		g.Epilogue = Identity
	}
	if g.Epilogue.Arity() > 1+len(g.Extras) {
		return g, fmt.Errorf("skinny gemm %s: epilogue %s uses %d operands, have %d", g.Name, g.Epilogue, g.Epilogue.Arity(), 1+len(g.Extras))
	}
	for _, e := range g.Extras {
		if !cIdent(e.Name) || e.DType != F32 && e.DType != BF16 || e.Broadcast < PerCol || e.Broadcast > Full {
			return g, fmt.Errorf("skinny gemm %s: bad operand %+v", g.Name, e)
		}
	}
	if g.RowsPerBlock == 0 {
		g.RowsPerBlock = 1
		for _, r := range []int{4, 2} {
			if g.Weight == WeightNVFP4 && r > 2 {
				continue
			}
			if g.N%r == 0 && g.N/r >= 2*132 {
				g.RowsPerBlock = r
				break
			}
		}
	}
	if g.N <= 0 || g.N%g.RowsPerBlock != 0 {
		return g, fmt.Errorf("skinny gemm %s: N %d not a multiple of %d rows per block", g.Name, g.N, g.RowsPerBlock)
	}
	if per := 32 * g.Weight.chunk() * g.SplitK; g.K <= 0 || g.K%per != 0 {
		return g, fmt.Errorf("skinny gemm %s: K %d not a multiple of %d", g.Name, g.K, per)
	}
	if g.Warps < 1 || g.Warps > 32 || g.SplitK < 1 || g.Unroll < 1 || g.Prefetch < 0 || 32*g.Warps < g.RowsPerBlock*g.MaxM {
		return g, fmt.Errorf("skinny gemm %s: bad schedule", g.Name)
	}
	return g, nil
}

// Grid returns the launch grid and block size.
func (g SkinnyGEMM) Grid() (grid, block [3]int, err error) {
	g, err = g.norm()
	if err != nil {
		return grid, block, err
	}
	return [3]int{g.N / g.RowsPerBlock, g.SplitK, 1}, [3]int{32 * g.Warps, 1, 1}, nil
}

// WorkspaceBytes reports the bytes of zero-initialized workspace the kernel
// needs (FP32 partials then int32 counters), or 0 without split-K. The
// counters must be zero before the first launch; the kernel leaves them zero.
func (g SkinnyGEMM) WorkspaceBytes() int {
	g, err := g.norm()
	if err != nil || g.SplitK == 1 {
		return 0
	}
	return 4*g.SplitK*g.MaxM*g.N + 4*(g.N/g.RowsPerBlock)
}

// params lists the kernel's parameters as C declarations.
func (g SkinnyGEMM) params() []string {
	p := []string{"const uint16_t* __restrict__ x"}
	switch g.Weight {
	case WeightBF16:
		p = append(p, "const uint16_t* __restrict__ w")
	case WeightBF16HiLo:
		p = append(p, "const uint16_t* __restrict__ w_hi", "const uint16_t* __restrict__ w_lo")
	case WeightNVFP4:
		p = append(p, "const uint8_t* __restrict__ w", "const uint8_t* __restrict__ w_sf", "const float* __restrict__ w_gs")
	}
	for _, e := range g.Extras {
		p = append(p, "const "+e.DType.cType()+"* __restrict__ "+e.Name)
	}
	p = append(p, g.Out.cType()+"* __restrict__ out")
	if g.SplitK > 1 {
		p = append(p, "float* __restrict__ ws")
	}
	return p
}

// CUDA returns the kernel source (without [Prelude]). The kernel is
// template <int M> __global__ void Name(params...).
func (g SkinnyGEMM) CUDA() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	R, W, E := g.RowsPerBlock, g.Warps, g.Weight.chunk()
	steps := g.K / (32 * E)
	u := g.Unroll
	fullPrefetch := false
	if g.Prefetch > 0 {
		u = g.Prefetch
		if maxSteps := (steps/g.SplitK + W - 1) / W; maxSteps <= g.Prefetch {
			u = maxSteps
			fullPrefetch = true
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: out[m,n] = %s over x[M,%d] bf16 · W[%d,%d] %s, M <= %d.\n", g.Name, g.Epilogue, g.K, g.N, g.K, g.Weight, g.MaxM)
	fmt.Fprintf(&b, "template <int M>\n__global__ void __launch_bounds__(%d) %s(\n    %s) {\n", 32*W, g.Name, strings.Join(g.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, K = %d, R = %d, WARPS = %d, U = %d, SPLIT = %d, MAXM = %d;\n", g.N, g.K, R, W, u, g.SplitK, g.MaxM)
	b.WriteString("    static_assert(M >= 1 && M <= MAXM, \"M out of range\");\n")
	fmt.Fprintf(&b, "    constexpr int STEPS = %d;  // 32-lane steps of %d values along K\n", steps, E)
	b.WriteString(`    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n0 = blockIdx.x * R;
    const int first = blockIdx.y * (STEPS / SPLIT), last = first + STEPS / SPLIT;
`)
	if g.Weight == WeightNVFP4 {
		b.WriteString("    const float gs = w_gs[0]; // read before the reduction's barriers\n")
	}
	b.WriteString(`    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
`)
	if fullPrefetch {
		b.WriteString("    {\n        const int base = first + warp;\n")
	} else {
		b.WriteString("    for (int base = first + warp; base < last; base += WARPS * U) {\n")
	}
	switch g.Weight {
	case WeightBF16, WeightBF16HiLo:
		planes := []string{"w"}
		if g.Weight == WeightBF16HiLo {
			planes = []string{"w_hi", "w_lo"}
		}
		fmt.Fprintf(&b, "        eg_u32x4 wv[U][R][%d];\n", len(planes))
		b.WriteString(`#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = base + u * WARPS;
#pragma unroll
            for (int r = 0; r < R; ++r) {
`)
		for i, p := range planes {
			fmt.Fprintf(&b, "                wv[u][r][%d] = s < last ? eg_ldg_stream(%s + (size_t)(n0 + r) * K + s * 256 + lane * 8) : eg_u32x4{0, 0, 0, 0};\n", i, p)
		}
		b.WriteString(`            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = base + u * WARPS;
            if (s >= last) break;
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const eg_u32x4 xv = eg_ldg16(x + (size_t)m * K + s * 256 + lane * 8);
                const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float a = acc[r][m];
`)
		for i := range planes {
			fmt.Fprintf(&b, "                    {\n                        const uint32_t ww[4] = {wv[u][r][%d].x, wv[u][r][%d].y, wv[u][r][%d].z, wv[u][r][%d].w};\n", i, i, i, i)
			b.WriteString(`#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a += eg_bf16_to_f32(xw[j]) * eg_bf16_to_f32(ww[j]);
                            a += eg_bf16_hi_to_f32(xw[j]) * eg_bf16_hi_to_f32(ww[j]);
                        }
                    }
`)
		}
		b.WriteString(`                    acc[r][m] = a;
                }
            }
        }
    }
`)
	case WeightNVFP4:
		b.WriteString(`        eg_u32x4 wv[U][R];
        uint32_t sv[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = base + u * WARPS;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                // 16 bytes hold 32 codes, k = s*1024 + lane*32 + [0, 32); two block scales.
                wv[u][r] = s < last ? eg_ldg_stream(w + (size_t)(n0 + r) * (K / 2) + s * 512 + lane * 16) : eg_u32x4{0, 0, 0, 0};
                sv[u][r] = s < last ? *reinterpret_cast<const uint16_t*>(w_sf + (size_t)(n0 + r) * (K / 16) + s * 64 + lane * 2) : 0u;
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int s = base + u * WARPS;
            if (s >= last) break;
            float wf[R][32];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t q[4] = {wv[u][r].x, wv[u][r].y, wv[u][r].z, wv[u][r].w};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    uint32_t h[2];
                    eg_e2m1x4_to_bf16x4(q[j / 2] >> (16 * (j % 2)), h);
                    wf[r][4 * j] = eg_bf16_to_f32(h[0]);
                    wf[r][4 * j + 1] = eg_bf16_hi_to_f32(h[0]);
                    wf[r][4 * j + 2] = eg_bf16_to_f32(h[1]);
                    wf[r][4 * j + 3] = eg_bf16_hi_to_f32(h[1]);
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                float xf[32];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const eg_u32x4 xv = eg_ldg16(x + (size_t)m * K + s * 1024 + lane * 32 + c * 8);
                    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        xf[8 * c + 2 * j] = eg_bf16_to_f32(xw[j]);
                        xf[8 * c + 2 * j + 1] = eg_bf16_hi_to_f32(xw[j]);
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float lo = 0.0f, hi = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        lo += xf[i] * wf[r][i];
                        hi += xf[16 + i] * wf[r][16 + i];
                    }
                    acc[r][m] += lo * eg_e4m3_to_f32(sv[u][r] & 0xffu) + hi * eg_e4m3_to_f32(sv[u][r] >> 8);
                }
            }
        }
    }
`)
	}
	// Cross-lane, then cross-warp reduction.
	b.WriteString(`#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) acc[r][m] = eg_warp_sum(acc[r][m]);
    __shared__ float red[WARPS][R][M];
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < M; ++m) red[warp][r][m] = acc[r][m];
    }
    __syncthreads();
    const int t = threadIdx.x;
    float v = 0.0f;
    if (t < R * M) {
        for (int i = 0; i < WARPS; ++i) v += red[i][t / M][t % M];
    }
`)
	if g.SplitK > 1 {
		b.WriteString(`    // Split-K: publish this block's partial; the last block of the column group reduces.
    int* counters = reinterpret_cast<int*>(ws + (size_t)SPLIT * MAXM * N);
    __shared__ int is_last;
    if (t < R * M) ws[((size_t)blockIdx.y * M + t % M) * N + n0 + t / M] = v;
    __threadfence();
    __syncthreads();
    if (t == 0) is_last = atomicAdd(&counters[blockIdx.x], 1) == SPLIT - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    if (t < R * M) {
        v = 0.0f;
        for (int s = 0; s < SPLIT; ++s) v += eg_ld_cg(&ws[((size_t)s * M + t % M) * N + n0 + t / M]);
    }
    if (t == 0) counters[blockIdx.x] = 0;
`)
	}
	b.WriteString("    if (t < R * M) {\n        const int n = n0 + t / M, m = t % M;\n")
	if g.Weight == WeightNVFP4 {
		b.WriteString("        v *= gs;\n")
	}
	args := []string{"v"}
	for _, e := range g.Extras {
		args = append(args, e.load("m", "n", "N"))
	}
	expr, err := g.Epilogue.CUDA(args...)
	if err != nil {
		return "", err
	}
	if g.Out == BF16 {
		fmt.Fprintf(&b, "        out[(size_t)m * N + n] = eg_f32_to_bf16(%s);\n", expr)
	} else {
		fmt.Fprintf(&b, "        out[(size_t)m * N + n] = %s;\n", expr)
	}
	b.WriteString("    }\n}\n")
	return b.String(), nil
}

// Launcher returns a host function
//
//	void Name_launch(int m, <params...>, cudaStream_t stream)
//
// that launches the M = m instantiation, for a torch load_inline wrapper.
func (g SkinnyGEMM) Launcher() (string, error) {
	g, err := g.norm()
	if err != nil {
		return "", err
	}
	grid, block, _ := g.Grid()
	params := g.params()
	names := make([]string, len(params))
	for i, p := range params {
		f := strings.Fields(p)
		names[i] = f[len(f)-1]
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int m, %s, cudaStream_t stream) {\n", g.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    const dim3 grid(%d, %d), block(%d);\n    switch (m) {\n", grid[0], grid[1], block[0])
	for m := 1; m <= g.MaxM; m++ {
		fmt.Fprintf(&b, "    case %d: EG_LAUNCH(%s<%d>, grid, block, 0, stream, %s); break;\n", m, g.Name, m, strings.Join(names, ", "))
	}
	b.WriteString("    }\n}\n")
	return b.String(), nil
}

// GEMMWeights holds W in one of the formats of [WeightFormat].
type GEMMWeights struct {
	BF16       []uint16 // WeightBF16 [N*K]; WeightBF16HiLo hi
	Lo         []uint16 // WeightBF16HiLo lo
	Packed     []uint8  // WeightNVFP4 [N*K/2]
	Scales     []uint8  // WeightNVFP4 [N*K/16]
	GlobalScal float32  // WeightNVFP4
}

// Reference computes g's output for m rows of x (BF16 bits, [m, K]) in
// float64, as moe8_bf16_proj/reference.py:72-76 does, then applies the
// epilogue in FP32 and rounds to g.Out. extras holds each operand's values
// as FP32.
func (g SkinnyGEMM) Reference(m int, x []uint16, w GEMMWeights, extras [][]float32) ([]float32, error) {
	g, err := g.norm()
	if err != nil {
		return nil, err
	}
	if m < 1 || m > g.MaxM || len(x) != m*g.K || len(extras) != len(g.Extras) {
		return nil, fmt.Errorf("skinny gemm %s reference: bad inputs", g.Name)
	}
	out := make([]float32, m*g.N)
	rows := make([]int, m)
	for i := range rows {
		rows[i] = i
	}
	xRow := func(i int) []uint16 { return x[i*g.K : (i+1)*g.K] }
	if err := gemmRows(g.N, g.K, g.Weight, w, xRow, rows, g.Epilogue, g.Extras, extras, g.Out, out); err != nil {
		return nil, fmt.Errorf("skinny gemm %s reference: %w", g.Name, err)
	}
	return out, nil
}

// gemmRows computes out[i, n] = epi(Σ_k xRow(i)[k] · W[n, k], extras at
// (i, n)) for each output row i of rows in float64, applies the epilogue in
// FP32 and rounds to d. out and Full extras are [*, n] row-major.
func gemmRows(n, k int, f WeightFormat, w GEMMWeights, xRow func(int) []uint16, rows []int, epi Lambda, ops []Operand, extras [][]float32, d DType, out []float32) error {
	row := make([]float64, k)
	vals := make([]float32, k)
	for c := 0; c < n; c++ {
		switch f {
		case WeightBF16:
			for j := range row {
				row[j] = float64(FromBF16Bits(w.BF16[c*k+j]))
			}
		case WeightBF16HiLo:
			for j := range row {
				row[j] = float64(FromBF16Bits(w.BF16[c*k+j])) + float64(FromBF16Bits(w.Lo[c*k+j]))
			}
		case WeightNVFP4:
			if err := DequantNVFP4(vals, w.Packed[c*k/2:(c+1)*k/2], w.Scales[c*k/16:(c+1)*k/16], w.GlobalScal); err != nil {
				return err
			}
			for j := range row {
				row[j] = float64(vals[j])
			}
		}
		for _, i := range rows {
			var s float64
			for j, xv := range xRow(i) {
				s += float64(FromBF16Bits(xv)) * row[j]
			}
			args := []float32{float32(s)}
			for e, o := range ops {
				switch o.Broadcast {
				case PerCol:
					args = append(args, extras[e][c])
				case PerRow:
					args = append(args, extras[e][i])
				default:
					args = append(args, extras[e][i*n+c])
				}
			}
			out[i*n+c] = d.Round(epi.Eval(args...))
		}
	}
	return nil
}

// cIdent reports whether s is a C identifier that cannot collide with the
// prelude's eg_ names.
func cIdent(s string) bool {
	if s == "" || len(s) > 64 || strings.HasPrefix(s, "eg_") || strings.HasPrefix(s, "__") {
		return false
	}
	for i, r := range s {
		if !(r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || i > 0 && r >= '0' && r <= '9') {
			return false
		}
	}
	return true
}

// Source returns a complete CUDA translation unit: [Prelude], the kernel and
// its launcher, ready for torch.utils.cpp_extension.load_inline's
// cuda_sources (build with -std=c++20 -arch=sm_90a).
func (g SkinnyGEMM) Source() (string, error) {
	k, err := g.CUDA()
	if err != nil {
		return "", err
	}
	l, err := g.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + k + "\n" + l, nil
}
