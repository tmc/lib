package hopper

import (
	"fmt"
	"math"
	"strings"
)

// HeadDim is the attention head width PagedAttention supports.
const HeadDim = 64

// PagedAttention is grouped-query attention of a block of QLen new tokens
// per request over a paged FP8 E4M3 KV cache, causal inside the block and
// optionally windowed, as in moe8_attention_decode/reference.py:216-236.
//
// Request b's cache holds seqlens[b] tokens, the block's own included; query
// j sits at position p = seqlens[b] - QLen + j and attends keys
// max(0, p - Window + 1) .. p (0 .. p when Window is 0). Token t of request
// b is at cache row page_table[b][t / Page]·Page + t % Page, and a row holds
// Hkv heads of 64 E4M3 values each. Keys and values are e4m3·k_scale and
// e4m3·v_scale. Query values are q·QScale; logits are scaled by SMScale.
//
// The kernel splits each request's keys into Chunk-key blocks (split-KV).
// One block of 4 warps owns one (request, KV head, chunk) and all its
// query rows (the KV head's Hq/Hkv query heads times QLen tokens), computes
// Q·Kᵀ and P·V with mma.sync m16n8k16 BF16 tensor-core tiles (FP8 and
// FP32 inputs convert to BF16 exactly; P rounds to BF16) and an online
// softmax in FP32. When a request needs more than one chunk, each block
// stores its (max, sum, output) partial in a workspace. The chunks form
// groups of 16; the last block to finish in a group merges the group's
// partials, and with more than one group the last group to finish merges the
// group results. Every merge runs in chunk order, so results do not depend on
// scheduling, and at most ceil(chunks/16) blocks per (request, KV head) read
// 16 partials each, plus one that reads the group results.
type PagedAttention struct {
	Name    string
	Hq, Hkv int
	QLen    int     // 1 to 8; Hq/Hkv·QLen ≤ 64
	Window  int     // 0 for full attention
	Page    int     // tokens per page
	Q       DType   // E4M3 or BF16 [tokens, Hq, 64], tokens = batch·QLen
	Out     DType   // F32 or BF16 [tokens, Hq, 64]
	QScale  float32 // 0 means 1
	SMScale float32 // 0 means 1/8
	Chunk   int     // keys per block, a multiple of 64; 0 means 256

	// Depth is the number of 16-key tiles each warp loads per step: 1 (the
	// default for 0) or 2. A step's K and V loads are issued a step ahead,
	// so Depth 2 doubles the bytes each thread has in flight (8 16-byte
	// loads instead of 4) and the K/V shared memory.
	Depth int

	// QKFP8 computes Q·Kᵀ with mma.sync m16n8k32 E4M3 tiles on the stored
	// bytes (Q must be E4M3), skipping the widening of K to BF16. Products
	// are exact; the tensor core's FP32 accumulation order differs.
	QKFP8 bool

	// With Gate, out = head_rms(attention, GateEps) · sigmoid(gate), the
	// rms_norm_gate epilogue (moe8_attention_decode/reference.py:336), with
	// gate BF16 read at gate[t·GateStride + GateOffset + head·64 + d] (in
	// place from the fused qkv rows: stride (2Hq+2Hkv)·64, offset
	// (Hq+2Hkv)·64).
	Gate       bool
	GateStride int
	GateOffset int
	GateEps    float32 // 0 means 1e-5
}

type attnGeom struct {
	G, rows, mrows, mt, kw int
}

// attnGroup is the number of chunk partials one block merges.
const attnGroup = 16

func (a PagedAttention) norm() (PagedAttention, attnGeom, error) {
	var g attnGeom
	if !cIdent(a.Name) {
		return a, g, fmt.Errorf("paged attention: bad name %q", a.Name)
	}
	if a.Hq < 1 || a.Hkv < 1 || a.Hq%a.Hkv != 0 {
		return a, g, fmt.Errorf("paged attention %s: bad heads %d/%d", a.Name, a.Hq, a.Hkv)
	}
	g.G = a.Hq / a.Hkv
	g.rows = g.G * a.QLen
	if a.QLen < 1 || a.QLen > 8 || g.rows > 64 {
		return a, g, fmt.Errorf("paged attention %s: %d query rows per KV head (QLen %d), want at most 64", a.Name, g.rows, a.QLen)
	}
	if a.Window < 0 || a.Page < 1 {
		return a, g, fmt.Errorf("paged attention %s: bad window %d or page %d", a.Name, a.Window, a.Page)
	}
	if a.QKFP8 && a.Q != E4M3 {
		return a, g, fmt.Errorf("paged attention %s: QKFP8 needs E4M3 queries", a.Name)
	}
	if a.Q != E4M3 && a.Q != BF16 || a.Out != F32 && a.Out != BF16 {
		return a, g, fmt.Errorf("paged attention %s: q must be E4M3 or BF16, out F32 or BF16", a.Name)
	}
	if a.QScale == 0 {
		a.QScale = 1
	}
	if a.SMScale == 0 {
		a.SMScale = 1 / float32(math.Sqrt(HeadDim))
	}
	if a.Chunk == 0 {
		a.Chunk = 256
	}
	if a.GateEps == 0 {
		a.GateEps = 1e-5
	}
	if a.Gate && (a.GateStride < a.Hq*HeadDim || a.GateOffset < 0 || a.GateOffset%8 != 0 || a.GateStride%8 != 0) {
		return a, g, fmt.Errorf("paged attention %s: bad gate stride %d or offset %d", a.Name, a.GateStride, a.GateOffset)
	}
	if a.Chunk%64 != 0 {
		return a, g, fmt.Errorf("paged attention %s: chunk %d is not a multiple of 64", a.Name, a.Chunk)
	}
	if a.Depth == 0 {
		a.Depth = 1
	}
	if a.Depth != 1 && a.Depth != 2 {
		return a, g, fmt.Errorf("paged attention %s: depth %d, want 1 or 2", a.Name, a.Depth)
	}
	switch {
	case g.rows <= 16:
		g.mt = 1
	case g.rows <= 32:
		g.mt = 2
	default:
		g.mt = 4
	}
	g.mrows, g.kw = 16*g.mt, 4/g.mt
	if n := a.SharedBytes(); n > 48<<10 {
		return a, g, fmt.Errorf("paged attention %s: %d bytes of static shared memory, want at most 48 KiB", a.Name, n)
	}
	return a, g, nil
}

// SharedBytes returns the kernel's static shared memory in bytes, or 0 if a
// is invalid.
func (a PagedAttention) SharedBytes() int {
	if a.Hkv < 1 || a.Hq%a.Hkv != 0 || a.QLen < 1 {
		return 0
	}
	rows := a.Hq / a.Hkv * a.QLen
	mt := 4
	switch {
	case rows <= 16:
		mt = 1
	case rows <= 32:
		mt = 2
	}
	mrows, kw, depth := 16*mt, 4/mt, max(a.Depth, 1)
	ld := 72 // BF16 row stride
	if a.QKFP8 {
		ld = 40 // E4M3 bytes as uint16
	}
	return 2*mrows*ld + 2*kw*16*depth*ld + 2*kw*16*depth*72 + 8*kw*mrows + 4*kw*mrows*HeadDim + 4
}

// Splits returns the number of key chunks for a request attending maxKeys
// keys: the grid's x dimension. maxKeys is at most Window+QLen-1 for a
// windowed layer, else the longest cache length.
func (a PagedAttention) Splits(maxKeys int) int {
	a, _, err := a.norm()
	if err != nil || maxKeys < 1 {
		return 0
	}
	return (maxKeys + a.Chunk - 1) / a.Chunk
}

// WorkspaceBytes returns the workspace size for batch requests of at most
// maxKeys keys: for each (request, KV head) 1 + ceil(splits/16) int32
// counters, then the chunk and group partials. The counters must be zero
// before the first launch; the kernel leaves them zero.
func (a PagedAttention) WorkspaceBytes(batch, maxKeys int) int {
	_, g, err := a.norm()
	if err != nil {
		return 0
	}
	s := a.Splits(maxKeys)
	ng := (s + attnGroup - 1) / attnGroup
	return 4*batch*a.Hkv*(1+ng) + 4*batch*a.Hkv*(s+ng)*g.rows*(2+HeadDim)
}

func (a PagedAttention) params() []string {
	p := []string{
		"const " + a.Q.cType() + "* __restrict__ q",
		"const uint8_t* __restrict__ k_cache",
		"const uint8_t* __restrict__ v_cache",
		"const int32_t* __restrict__ page_table",
		"int max_pages",
		"const int32_t* __restrict__ seqlens",
		"const float* __restrict__ k_scale",
		"const float* __restrict__ v_scale",
	}
	if a.Gate {
		p = append(p, "const uint16_t* __restrict__ gate")
	}
	return append(p, a.Out.cType()+"* __restrict__ out", "unsigned char* __restrict__ ws")
}

// Grid returns the grid and block for batch requests of at most maxKeys keys.
func (a PagedAttention) Grid(batch, maxKeys int) (grid, block [3]int) {
	return [3]int{a.Splits(maxKeys), a.Hkv, batch}, [3]int{128, 1, 1}
}

// CUDA returns the kernel source (without [Prelude]):
//
//	__global__ void Name(q, k_cache, v_cache, page_table, max_pages,
//	    seqlens, k_scale, v_scale, out, ws)
func (a PagedAttention) CUDA() (string, error) {
	a, g, err := a.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: paged FP8 GQA attention, %d/%d heads, %d query tokens, window %d, page %d, %d-key splits, depth %d.\n",
		a.Name, a.Hq, a.Hkv, a.QLen, a.Window, a.Page, a.Chunk, a.Depth)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n", a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, L = %d, ROWS = %d, MROWS = %d, MT = %d, KW = %d;\n",
		a.Hq, a.Hkv, g.G, a.QLen, g.rows, g.mrows, g.mt, g.kw)
	fmt.Fprintf(&b, "    constexpr int WINDOW = %d, PAGE = %d, CHUNK = %d, DEPTH = %d, STEP = 16 * KW * DEPTH, LD = 72;\n",
		a.Window, a.Page, a.Chunk, a.Depth)
	b.WriteString("    constexpr int PIECES = STEP * 4, NP = (PIECES + 127) / 128, GROUP = " + fmt.Sprint(attnGroup) + ";\n")
	fmt.Fprintf(&b, "    const float scale = %s * k_scale[0] * 1.4426950408889634f; // log2 domain\n",
		floatLiteral(a.QScale*a.SMScale))
	body := attnBody
	if a.QKFP8 {
		for _, r := range attnFP8 {
			if !strings.Contains(body, r[0]) {
				return "", fmt.Errorf("paged attention %s: internal: fp8 rewrite %q does not apply", a.Name, r[0])
			}
			body = strings.ReplaceAll(body, r[0], r[1])
		}
	}
	b.WriteString(body)
	if a.Out == BF16 {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = eg_f32_to_bf16(v); };\n")
	} else {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = v; };\n")
	}
	b.WriteString(attnCombine)
	if a.Gate {
		fmt.Fprintf(&b, "    constexpr int GATE_STRIDE = %d, GATE_OFFSET = %d;\n    constexpr float GATE_EPS = %s;\n",
			a.GateStride, a.GateOffset, floatLiteral(a.GateEps))
		b.WriteString(attnGateStore)
	} else {
		b.WriteString(attnStore)
	}
	return b.String(), nil
}

const attnBody = `    const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int mt = warp % MT, kw = warp / MT;
    const int seqlen = seqlens[b], p0 = seqlen - L;
    const int kbegin = WINDOW ? max(0, p0 - WINDOW + 1) : 0;
    const int nsplit = (seqlen - kbegin + CHUNK - 1) / CHUNK;
    if (split >= nsplit) return;
    const int c0 = kbegin + split * CHUNK, c1 = min(seqlen, c0 + CHUNK);
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    alignas(16) __shared__ uint16_t sq[MROWS * LD];
    alignas(16) __shared__ uint16_t sk[STEP * LD];
    alignas(16) __shared__ uint16_t sv[STEP * LD];
    __shared__ float red_m[KW * MROWS], red_l[KW * MROWS];
    __shared__ float red_o[KW * MROWS * 64];
    __shared__ int last;

    // Query rows r = j*G + g (token j, head kvh*G + g) as BF16, padded with zeros.
    for (int i = tid; i < MROWS * 8; i += 128) {
        const int r = i / 8, d = (i % 8) * 8;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) {
            float f[8];
            eg_load<8>(q + ((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, f);
            w = eg_u32x4{eg_pack_bf16x2(f[0], f[1]), eg_pack_bf16x2(f[2], f[3]),
                         eg_pack_bf16x2(f[4], f[5]), eg_pack_bf16x2(f[6], f[7])};
        }
        *reinterpret_cast<eg_u32x4*>(&sq[r * LD + d]) = w;
    }

    // Each thread moves NP 16-byte pieces of K and of V (4 per key).
    eg_u32x4 kreg[NP], vreg[NP];
    auto fetch = [&](int kb) {
#pragma unroll
        for (int n = 0; n < NP; ++n) {
            const int piece = tid + 128 * n, kp = kb + piece / 4;
            kreg[n] = vreg[n] = eg_u32x4{0, 0, 0, 0};
            if (piece < PIECES && kp < c1) {
                const size_t row = (size_t)page_table[(size_t)b * max_pages + kp / PAGE] * PAGE + kp % PAGE;
                const size_t at = (row * HKV + kvh) * 64 + (piece % 4) * 16;
                kreg[n] = eg_ldg16(k_cache + at);
                vreg[n] = eg_ldg16(v_cache + at);
            }
        }
    };
    auto widen = [&](uint16_t* dst, eg_u32x4 u) {
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
        uint32_t o[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            o[2 * i] = eg_pack_bf16x2(eg_e4m3_to_f32(w[i] & 0xffu), eg_e4m3_to_f32((w[i] >> 8) & 0xffu));
            o[2 * i + 1] = eg_pack_bf16x2(eg_e4m3_to_f32((w[i] >> 16) & 0xffu), eg_e4m3_to_f32(w[i] >> 24));
        }
        reinterpret_cast<eg_u32x4*>(dst)[0] = eg_u32x4{o[0], o[1], o[2], o[3]};
        reinterpret_cast<eg_u32x4*>(dst)[1] = eg_u32x4{o[4], o[5], o[6], o[7]};
    };
    fetch(c0);
    __syncthreads();

    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LD + ks * 16 + (lane / 16) * 8]);

    // This lane's two rows (g and g+8 of its tile): positions and first keys.
    int pos[2], lo[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + lane / 4 + 8 * h;
        pos[h] = p0 + min(r / G, L - 1);
        lo[h] = WINDOW ? max(0, pos[h] - WINDOW + 1) : 0;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f}, o[8][4] = {};

    for (int kb = c0; kb < c1; kb += STEP) {
        __syncthreads();
#pragma unroll
        for (int n = 0; n < NP; ++n) {
            const int piece = tid + 128 * n;
            if (piece < PIECES) {
                widen(&sk[(piece / 4) * LD + (piece % 4) * 16], kreg[n]);
                widen(&sv[(piece / 4) * LD + (piece % 4) * 16], vreg[n]);
            }
        }
        __syncthreads();
        if (kb + STEP < c1) fetch(kb + STEP);
#pragma unroll
        for (int sub = 0; sub < DEPTH; ++sub) {
        const int kt0 = (kw * DEPTH + sub) * 16, key0 = kb + kt0;
        if (key0 >= c1) break;
        float s[2][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kt0 + lane % 8 + 8 * (lane / 16)) * LD + ks * 16 + 8 * ((lane / 8) % 2)]);
            eg_mma_bf16(s[0], qa[ks], &kf[0], s[0]);
            eg_mma_bf16(s[1], qa[ks], &kf[2], s[1]);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float mx = NEG_INF;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int key = key0 + nt * 8 + 2 * (lane % 4) + e;
                    float& x = s[nt][2 * h + e];
                    x = key < c1 && key >= lo[h] && key <= pos[h] ? x * scale : NEG_INF;
                    mx = fmaxf(mx, x);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mnew = fmaxf(m[h], mx), base = mnew == NEG_INF ? 0.0f : mnew;
            const float corr = exp2f(m[h] - base);
            m[h] = mnew;
            l[h] *= corr;
#pragma unroll
            for (int nd = 0; nd < 8; ++nd) { o[nd][2 * h] *= corr; o[nd][2 * h + 1] *= corr; }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float& x = s[nt][2 * h + e];
                    x = exp2f(x - base);
                    l[h] += x;
                }
        }
        const uint32_t pa[4] = {eg_pack_bf16x2(s[0][0], s[0][1]), eg_pack_bf16x2(s[0][2], s[0][3]),
                                eg_pack_bf16x2(s[1][0], s[1][1]), eg_pack_bf16x2(s[1][2], s[1][3])};
#pragma unroll
        for (int nd = 0; nd < 4; ++nd) {
            uint32_t vf[4];
            eg_ldmatrix_x4_trans(vf, &sv[(kt0 + lane % 8 + 8 * ((lane / 8) % 2)) * LD + nd * 16 + 8 * (lane / 16)]);
            eg_mma_bf16(o[2 * nd], pa, &vf[0], o[2 * nd]);
            eg_mma_bf16(o[2 * nd + 1], pa, &vf[2], o[2 * nd + 1]);
        }
        }
    }

    // Combine the KW key groups of each row tile in shared memory.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
        const int r = mt * 16 + lane / 4 + 8 * h;
        if (lane % 4 == 0) { red_m[kw * MROWS + r] = m[h]; red_l[kw * MROWS + r] = l[h]; }
#pragma unroll
        for (int nd = 0; nd < 8; ++nd)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                red_o[(kw * MROWS + r) * 64 + nd * 8 + 2 * (lane % 4) + e] = o[nd][2 * h + e];
    }
    __syncthreads();
    const float vs = v_scale[0];
    // Per (request, KV head): 1 + NG counters, then gridDim.x chunk partials
    // and NG group partials of ROWS x (max, sum, 64 outputs).
    const int NG = (gridDim.x + GROUP - 1) / GROUP;
    int* const cnt = reinterpret_cast<int*>(ws) + ((size_t)b * HKV + kvh) * (1 + NG);
    float* const all = reinterpret_cast<float*>(ws + 4 * (size_t)gridDim.z * HKV * (1 + NG)) +
                       ((size_t)b * HKV + kvh) * (gridDim.x + NG) * ROWS * 66;
    float* const mine = all + (size_t)split * ROWS * 66;
    for (int i = tid; i < ROWS * 64 && nsplit > 1; i += 128) {
        const int r = i / 64, d = i % 64;
        float mx = NEG_INF;
#pragma unroll
        for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_m[k * MROWS + r]);
        const float base = mx == NEG_INF ? 0.0f : mx;
        float sum = 0.0f, acc = 0.0f;
#pragma unroll
        for (int k = 0; k < KW; ++k) {
            const float w = exp2f(red_m[k * MROWS + r] - base);
            sum += w * red_l[k * MROWS + r];
            acc += w * red_o[(k * MROWS + r) * 64 + d];
        }
        mine[r * 66 + 2 + d] = acc;
        if (d == 0) { mine[r * 66] = mx; mine[r * 66 + 1] = sum; }
    }
`

// attnCombine finishes: direct store for one split, else the two-level
// merge of the chunk partials.
// attnFP8 rewrites attnBody to keep Q and K as E4M3 bytes (rows of 64
// bytes padded to 80) and multiply them with m16n8k32 tiles. In the b16
// view the k32 fragments have the k16 BF16 layout, so the ldmatrix
// addressing is unchanged apart from the row stride and step count.
var attnFP8 = [][2]string{
	{"alignas(16) __shared__ uint16_t sq[MROWS * LD];", "constexpr int LD8 = 40;\n    alignas(16) __shared__ uint16_t sq[MROWS * LD8];"},
	{"alignas(16) __shared__ uint16_t sk[STEP * LD];", "alignas(16) __shared__ uint16_t sk[STEP * LD8];"},
	{`    // Query rows r = j*G + g (token j, head kvh*G + g) as BF16, padded with zeros.
    for (int i = tid; i < MROWS * 8; i += 128) {
        const int r = i / 8, d = (i % 8) * 8;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) {
            float f[8];
            eg_load<8>(q + ((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, f);
            w = eg_u32x4{eg_pack_bf16x2(f[0], f[1]), eg_pack_bf16x2(f[2], f[3]),
                         eg_pack_bf16x2(f[4], f[5]), eg_pack_bf16x2(f[6], f[7])};
        }
        *reinterpret_cast<eg_u32x4*>(&sq[r * LD + d]) = w;
    }`, `    // Query rows r = j*G + g (token j, head kvh*G + g) as E4M3 bytes, padded with zeros.
    for (int i = tid; i < MROWS * 4; i += 128) {
        const int r = i / 4, piece = i % 4;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) w = eg_ldg16(q + ((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + piece * 16);
        *reinterpret_cast<eg_u32x4*>(&sq[r * LD8 + piece * 8]) = w;
    }`},
	{"    uint32_t qa[4][4];\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LD + ks * 16 + (lane / 16) * 8]);",
		"    uint32_t qa[2][4];\n#pragma unroll\n    for (int ks = 0; ks < 2; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LD8 + ks * 16 + (lane / 16) * 8]);"},
	{"widen(&sk[(piece / 4) * LD + (piece % 4) * 16], kreg[n]);",
		"*reinterpret_cast<eg_u32x4*>(&sk[(piece / 4) * LD8 + (piece % 4) * 8]) = kreg[n];"},
	{`        for (int ks = 0; ks < 4; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kt0 + lane % 8 + 8 * (lane / 16)) * LD + ks * 16 + 8 * ((lane / 8) % 2)]);
            eg_mma_bf16(s[0], qa[ks], &kf[0], s[0]);
            eg_mma_bf16(s[1], qa[ks], &kf[2], s[1]);`, `        for (int ks = 0; ks < 2; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kt0 + lane % 8 + 8 * (lane / 16)) * LD8 + ks * 16 + 8 * ((lane / 8) % 2)]);
            eg_mma_e4m3(s[0], qa[ks], &kf[0], s[0]);
            eg_mma_e4m3(s[1], qa[ks], &kf[2], s[1]);`},
}

const attnCombine = `    // The finished rows go to fin (the first KW slot of red_o, which each
    // thread reads only at the index it then writes).
    float* const fin = red_o;
    if (nsplit == 1) {
        for (int i = tid; i < ROWS * 64; i += 128) {
            const int r = i / 64, d = i % 64;
            float mx = NEG_INF;
#pragma unroll
            for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_m[k * MROWS + r]);
            float sum = 0.0f, acc = 0.0f;
#pragma unroll
            for (int k = 0; k < KW; ++k) {
                const float w = exp2f(red_m[k * MROWS + r] - mx);
                sum += w * red_l[k * MROWS + r];
                acc += w * red_o[(k * MROWS + r) * 64 + d];
            }
            fin[r * 64 + d] = acc / sum * vs;
        }
    } else {
        // merge combines n partials at src in chunk order: into fin (scaled
        // by vs) when final, else into the partial at dst. One thread per row
        // finds the row's max and weight sum in one online pass (red_m,
        // red_l); then each thread accumulates its OUTS outputs, with the
        // loads of several partials in flight.
        constexpr int OUTS = (ROWS * 64 + 127) / 128;
        auto merge = [&](const float* src, int n, float* dst, bool final) {
            for (int r = tid; r < ROWS; r += 128) {
                float mx = NEG_INF, sum = 0.0f;
#pragma unroll 8
                for (int s = 0; s < n; ++s) {
                    const float* p = src + (size_t)s * ROWS * 66 + r * 66;
                    const float m = eg_ld_cg(p), l = eg_ld_cg(p + 1), mnew = fmaxf(mx, m);
                    if (mnew != NEG_INF) sum = sum * exp2f(mx - mnew) + l * exp2f(m - mnew);
                    mx = mnew;
                }
                red_m[r] = mx;
                red_l[r] = sum;
            }
            __syncthreads();
            float base[OUTS], acc[OUTS];
#pragma unroll
            for (int k = 0; k < OUTS; ++k) {
                const int i = min(tid + 128 * k, ROWS * 64 - 1);
                base[k] = red_m[i / 64] == NEG_INF ? 0.0f : red_m[i / 64];
                acc[k] = 0.0f;
            }
#pragma unroll 4
            for (int s = 0; s < n; ++s) {
#pragma unroll
                for (int k = 0; k < OUTS; ++k) {
                    const int i = min(tid + 128 * k, ROWS * 64 - 1);
                    const float* p = src + (size_t)s * ROWS * 66 + (i / 64) * 66;
                    acc[k] += exp2f(eg_ld_cg(p) - base[k]) * eg_ld_cg(p + 2 + i % 64);
                }
            }
#pragma unroll
            for (int k = 0; k < OUTS; ++k) {
                const int i = tid + 128 * k;
                if (i >= ROWS * 64) break;
                if (final) fin[i] = acc[k] / red_l[i / 64] * vs;
                else dst[(i / 64) * 66 + 2 + i % 64] = acc[k];
            }
            for (int r = tid; r < ROWS && !final; r += 128) { dst[r * 66] = red_m[r]; dst[r * 66 + 1] = red_l[r]; }
        };
        const int ng = (nsplit + GROUP - 1) / GROUP, grp = split / GROUP;
        const int gn = min(GROUP, nsplit - grp * GROUP);
        __threadfence();
        __syncthreads();
        if (tid == 0) last = atomicAdd(cnt + 1 + grp, 1) == gn - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        if (tid == 0) cnt[1 + grp] = 0;
        float* const groups = all + (size_t)gridDim.x * ROWS * 66;
        if (ng == 1) {
            merge(all, nsplit, nullptr, true);
        } else {
            merge(all + (size_t)grp * GROUP * ROWS * 66, gn, groups + (size_t)grp * ROWS * 66, false);
            __threadfence();
            __syncthreads();
            if (tid == 0) last = atomicAdd(cnt, 1) == ng - 1;
            __syncthreads();
            if (!last) return;
            __threadfence();
            if (tid == 0) cnt[0] = 0;
            merge(groups, ng, nullptr, true);
        }
    }
    __syncthreads();
`

// attnStore writes fin to out.
const attnStore = `    for (int i = tid; i < ROWS * 64; i += 128) {
        const int r = i / 64, d = i % 64;
        put(((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, fin[i]);
    }
}
`

// attnGateStore writes head_rms(fin) · sigmoid(gate) to out, one warp per
// query row (head).
const attnGateStore = `    for (int r = warp; r < ROWS; r += 4) {
        const float v0 = fin[r * 64 + lane], v1 = fin[r * 64 + lane + 32];
        const float inv = rsqrtf(eg_warp_sum(v0 * v0 + v1 * v1) * (1.0f / 64) + GATE_EPS);
        const size_t t = (size_t)b * L + r / G;
        const int h = kvh * G + r % G;
        const uint16_t* const g = gate + t * GATE_STRIDE + GATE_OFFSET + h * 64;
        const size_t o = (t * HQ + h) * 64;
        put(o + lane, v0 * inv * eg_sigmoid(eg_bf16_to_f32(g[lane])));
        put(o + lane + 32, v1 * inv * eg_sigmoid(eg_bf16_to_f32(g[lane + 32])));
    }
}
`

// Launcher returns
//
//	void Name_launch(int batch, int max_keys, <kernel params...>, cudaStream_t stream)
//
// whose grid is fixed by batch and max_keys, so it can be captured in a CUDA
// graph and replayed for any lengths up to max_keys keys per request.
func (a PagedAttention) Launcher() (string, error) {
	a, _, err := a.norm()
	if err != nil {
		return "", err
	}
	params := a.params()
	names := make([]string, len(params))
	for i, p := range params {
		f := strings.Fields(p)
		names[i] = f[len(f)-1]
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int batch, int max_keys, %s, cudaStream_t stream) {\n", a.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    const dim3 grid((max_keys + %d) / %d, %d, batch), block(128);\n", a.Chunk-1, a.Chunk, a.Hkv)
	fmt.Fprintf(&b, "    if (grid.x > 0 && batch > 0) EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", a.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns [Prelude], the kernel and its launcher.
func (a PagedAttention) Source() (string, error) {
	src, err := a.CUDA()
	if err != nil {
		return "", err
	}
	l, err := a.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// PagedCache is the input of [PagedAttention.Reference].
type PagedCache struct {
	K, V      []uint8 // E4M3 [rows, Hkv, 64]
	PageTable []int32 // [batch, MaxPages]
	MaxPages  int
	SeqLens   []int32 // [batch]
	KScale    float32
	VScale    float32
	Gate      []float32 // with PagedAttention.Gate: [tokens, GateStride] values
}

// Reference computes the attention output [batch·QLen, Hq, 64] in float64
// from query values q (as stored, before QScale), following
// moe8_attention_decode/reference.py:216-236, then the gate epilogue (:236)
// when a.Gate is set.
func (a PagedAttention) Reference(q []float32, c PagedCache) ([]float64, error) {
	a, g, err := a.norm()
	if err != nil {
		return nil, err
	}
	batch := len(c.SeqLens)
	if len(q) != batch*a.QLen*a.Hq*HeadDim || len(c.PageTable) != batch*c.MaxPages ||
		a.Gate && len(c.Gate) != batch*a.QLen*a.GateStride {
		return nil, fmt.Errorf("paged attention %s reference: bad input sizes", a.Name)
	}
	out := make([]float64, len(q))
	row := func(b, t int) (int, error) {
		if t/a.Page >= c.MaxPages {
			return 0, fmt.Errorf("token %d past the page table", t)
		}
		r := int(c.PageTable[b*c.MaxPages+t/a.Page])*a.Page + t%a.Page
		if r < 0 || (r+1)*a.Hkv*HeadDim > len(c.K) {
			return 0, fmt.Errorf("cache row %d out of range", r)
		}
		return r, nil
	}
	for b := range batch {
		seq := int(c.SeqLens[b])
		for j := range a.QLen {
			p := seq - a.QLen + j
			start := 0
			if a.Window > 0 {
				start = max(0, p-a.Window+1)
			}
			for h := range a.Hq {
				kvh := h / g.G
				qv := q[((b*a.QLen+j)*a.Hq+h)*HeadDim:][:HeadDim]
				logits := make([]float64, p+1-start)
				mx := math.Inf(-1)
				for t := start; t <= p; t++ {
					r, err := row(b, t)
					if err != nil {
						return nil, fmt.Errorf("paged attention %s reference: %w", a.Name, err)
					}
					var dot float64
					for d, kq := range c.K[(r*a.Hkv+kvh)*HeadDim:][:HeadDim] {
						dot += float64(qv[d]) * float64(a.QScale) * float64(FromE4M3Bits(kq)) * float64(c.KScale)
					}
					logits[t-start] = dot * float64(a.SMScale)
					mx = math.Max(mx, logits[t-start])
				}
				var sum float64
				for i := range logits {
					logits[i] = math.Exp(logits[i] - mx)
					sum += logits[i]
				}
				o := out[((b*a.QLen+j)*a.Hq+h)*HeadDim:][:HeadDim]
				for t := start; t <= p; t++ {
					r, _ := row(b, t)
					w := logits[t-start] / sum
					for d, vq := range c.V[(r*a.Hkv+kvh)*HeadDim:][:HeadDim] {
						o[d] += w * float64(FromE4M3Bits(vq)) * float64(c.VScale)
					}
				}
				if a.Gate {
					var ss float64
					for _, v := range o {
						ss += v * v
					}
					inv := 1 / math.Sqrt(ss/HeadDim+float64(a.GateEps))
					g := c.Gate[(b*a.QLen+j)*a.GateStride+a.GateOffset+h*HeadDim:][:HeadDim]
					for d := range o {
						o[d] *= inv / (1 + math.Exp(-float64(g[d])))
					}
				}
			}
		}
	}
	return out, nil
}
