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
// Q·Kᵀ and P·V with mma.sync m16n8k16 F16 tensor-core tiles and an online
// softmax in FP32. K and V widen from E4M3 to F16 exactly
// (cvt.rn.f16x2.e4m3x2), as do E4M3 queries; BF16 queries and P in [0, 1]
// round to F16. K and V bytes stream through a cp.async ring in shared
// memory, Stages steps deep, with one barrier per step; V widens in
// registers. When a request needs more than one chunk, each block
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

	// Depth is the number of 16-key tiles each warp computes per step: 1
	// (the default for 0) or 2.
	Depth int

	// Stages is the depth of the ring that stages K and V bytes in shared
	// memory, in steps: 2 to 8. The block copies a step's keys with
	// cp.async Stages-1 steps before it uses them. 0 means the most, up to
	// 4, that fit 40 KiB.
	Stages int

	// QKFP8 computes Q·Kᵀ with mma.sync m16n8k32 E4M3 tiles on the stored
	// bytes (Q must be E4M3), skipping the widening of K to F16. Products
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
	if a.Stages == 0 {
		a.Stages = max(2, min(4, 40<<10/attnStageBytes(16*g.kw*a.Depth)))
	}
	if a.Stages < 2 || a.Stages > 8 {
		return a, g, fmt.Errorf("paged attention %s: %d stages, want 2 to 8", a.Name, a.Stages)
	}
	if n := a.SharedBytes(); n > 48<<10 {
		return a, g, fmt.Errorf("paged attention %s: %d bytes of static shared memory, want at most 48 KiB", a.Name, n)
	}
	return a, g, nil
}

// attnRowBytes is the shared memory stride of a staged K or V row: 64
// bytes padded so ldmatrix rows fall in distinct banks.
const attnRowBytes = 80

// attnStageBytes returns the bytes of one ring stage: step K and V rows.
func attnStageBytes(step int) int { return 2 * step * attnRowBytes }

// attnQStride returns the shared memory stride of a query row in uint16s.
func (a PagedAttention) attnQStride() int {
	if a.QKFP8 {
		return attnRowBytes / 2 // E4M3 bytes
	}
	return 72 // F16
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
	stage := attnStageBytes(16 * kw * depth)
	stages := a.Stages
	if stages == 0 {
		stages = max(2, min(4, 40<<10/stage))
	}
	ring := max(stages*stage, 4*kw*mrows*HeadDim) // red_o reuses the ring
	return 2*mrows*a.attnQStride() + 8*kw*mrows + 4 + ring
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
	fmt.Fprintf(&b, "// %s: paged FP8 GQA attention, %d/%d heads, %d query tokens, window %d, page %d, %d-key splits, depth %d, %d stages.\n",
		a.Name, a.Hq, a.Hkv, a.QLen, a.Window, a.Page, a.Chunk, a.Depth, a.Stages)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n", a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, L = %d, ROWS = %d, MROWS = %d, MT = %d, KW = %d;\n",
		a.Hq, a.Hkv, g.G, a.QLen, g.rows, g.mrows, g.mt, g.kw)
	fmt.Fprintf(&b, "    constexpr int WINDOW = %d, PAGE = %d, CHUNK = %d, DEPTH = %d, STEP = 16 * KW * DEPTH, STAGES = %d;\n",
		a.Window, a.Page, a.Chunk, a.Depth, a.Stages)
	fmt.Fprintf(&b, "    constexpr int PIECES = STEP * 4, NP = (PIECES + 127) / 128, GROUP = %d, LDQ = %d, RB = %d;\n",
		attnGroup, a.attnQStride(), attnRowBytes)
	b.WriteString("    constexpr int STAGE_BYTES = 2 * STEP * RB, RING_BYTES = STAGES * STAGE_BYTES > KW * MROWS * 256 ? STAGES * STAGE_BYTES : KW * MROWS * 256;\n")
	fmt.Fprintf(&b, "    const float scale = %s * k_scale[0] * 1.4426950408889634f; // log2 domain\n",
		floatLiteral(a.QScale*a.SMScale))
	b.WriteString(attnHead)
	qk := attnQKF16
	if a.QKFP8 {
		b.WriteString(attnQFP8)
		qk = attnQKFP8
	} else {
		b.WriteString(attnQF16)
	}
	b.WriteString(strings.NewReplacer("QK_LOAD\n", qk[0], "QK_TILE\n", qk[1]).Replace(attnLoop))
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

// attnHead declares the block's coordinates and stages its query rows
// (r = j*G + g: token j, head kvh*G + g) in sq, padded with zeros: F16
// values, or with QKFP8 the E4M3 bytes.
const attnHead = `    const int split = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane / 4, c = lane % 4;
    const int mt = warp % MT, kw = warp / MT;
    const int seqlen = seqlens[b], p0 = seqlen - L;
    const int kbegin = WINDOW ? max(0, p0 - WINDOW + 1) : 0;
    const int nsplit = (seqlen - kbegin + CHUNK - 1) / CHUNK;
    if (split >= nsplit) return;
    const int c0 = kbegin + split * CHUNK, c1 = min(seqlen, c0 + CHUNK);
    const float NEG_INF = -__uint_as_float(0x7f800000u);

    alignas(16) __shared__ uint16_t sq[MROWS * LDQ];
    __shared__ float red_m[KW * MROWS], red_l[KW * MROWS];
    __shared__ int last;
    // The cp.async ring: STAGES stages of STEP K rows then STEP V rows of RB
    // bytes; after the main loop, red_o.
    alignas(16) __shared__ unsigned char ring[RING_BYTES];
    float* const red_o = reinterpret_cast<float*>(ring);
`

const attnQF16 = `    for (int i = tid; i < MROWS * 8; i += 128) {
        const int r = i / 8, d = (i % 8) * 8;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) {
            float f[8];
            eg_load<8>(q + ((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, f);
            w = eg_u32x4{eg_pack_f16x2(f[0], f[1]), eg_pack_f16x2(f[2], f[3]),
                         eg_pack_f16x2(f[4], f[5]), eg_pack_f16x2(f[6], f[7])};
        }
        *reinterpret_cast<eg_u32x4*>(&sq[r * LDQ + d]) = w;
    }
`

const attnQFP8 = `    for (int i = tid; i < MROWS * 4; i += 128) {
        const int r = i / 4, piece = i % 4;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) w = eg_ldg16(q + ((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + piece * 16);
        *reinterpret_cast<eg_u32x4*>(&sq[r * LDQ + piece * 8]) = w;
    }
`

// attnLoop is the main loop. Thread tid copies NP 16-byte pieces of K and
// of V per step (pieces NP·tid .. NP·tid + NP - 1, all of key fk) with
// cp.async into a STAGES-deep ring, reading the key's page-table entry a
// step before it issues the copy. One barrier per step frees the stage
// the step refills. Warp kw takes DEPTH 16-key tiles of each step; K is
// read raw by the QK tile (QK_TILE) and V widened to F16 in registers as
// the P·V B fragments: lane (g, c) loads bytes 8g..8g+7 of keys 2c, 2c+1,
// 2c+8 and 2c+9, so n-tile t's column g is dimension 8g + t and o[t][i]
// holds dimension 8(2c + i%2) + t.
const attnLoop = `    const int fk = tid * NP / 4;
    auto rowof = [&](int kb) -> long long {
        const int kp = kb + fk;
        if (tid * NP >= PIECES || kp >= c1) return -1;
        return (long long)page_table[(size_t)b * max_pages + kp / PAGE] * PAGE + kp % PAGE;
    };
    auto issue = [&](int st, long long row) {
        unsigned char* const kd = ring + st * STAGE_BYTES;
#pragma unroll
        for (int n = 0; n < NP; ++n) {
            const int piece = tid * NP + n;
            if (piece >= PIECES) break;
            unsigned char* const dk = kd + (piece / 4) * RB + (piece % 4) * 16;
            if (row >= 0) {
                const size_t at = ((size_t)row * HKV + kvh) * 64 + (piece % 4) * 16;
                eg_cp_async16(dk, k_cache + at);
                eg_cp_async16(dk + STEP * RB, v_cache + at);
            } else {
                // Past the chunk's end P is 0, and 0·NaN is NaN: zero V.
                // K needs nothing; its logits are masked.
                *reinterpret_cast<eg_u32x4*>(dk + STEP * RB) = eg_u32x4{0, 0, 0, 0};
            }
        }
    };
    const int nsteps = (c1 - c0 + STEP - 1) / STEP;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nsteps) issue(s, rowof(c0 + s * STEP));
        eg_cp_async_commit();
    }
    long long next = rowof(c0 + (STAGES - 1) * STEP);
    __syncthreads();

QK_LOAD
    // This lane's two rows (g and g+8 of its tile): positions and first keys.
    int pos[2], lo[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + g + 8 * h;
        pos[h] = p0 + min(r / G, L - 1);
        lo[h] = WINDOW ? max(0, pos[h] - WINDOW + 1) : 0;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f}, o[8][4] = {};

    for (int it = 0; it < nsteps; ++it) {
        eg_cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (it + STAGES - 1 < nsteps) {
            issue((it + STAGES - 1) % STAGES, next);
            next = rowof(c0 + (it + STAGES) * STEP);
        }
        eg_cp_async_commit();
        const int kb = c0 + it * STEP;
        const unsigned char* const kst = ring + (it % STAGES) * STAGE_BYTES;
        const unsigned char* const vst = kst + STEP * RB;
#pragma unroll
        for (int sub = 0; sub < DEPTH; ++sub) {
        const int kt0 = (kw * DEPTH + sub) * 16, key0 = kb + kt0;
        if (key0 >= c1) break;
        float s[2][4] = {};
QK_TILE
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float mx = NEG_INF;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int key = key0 + nt * 8 + 2 * c + e;
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
            for (int t = 0; t < 8; ++t) { o[t][2 * h] *= corr; o[t][2 * h + 1] *= corr; }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float& x = s[nt][2 * h + e];
                    x = exp2f(x - base);
                    l[h] += x;
                }
        }
        const uint32_t pa[4] = {eg_pack_f16x2(s[0][0], s[0][1]), eg_pack_f16x2(s[0][2], s[0][3]),
                                eg_pack_f16x2(s[1][0], s[1][1]), eg_pack_f16x2(s[1][2], s[1][3])};
        const unsigned char* const vr = vst + (kt0 + 2 * c) * RB + 8 * g;
        const eg_u32x2 va = *reinterpret_cast<const eg_u32x2*>(vr), vb = *reinterpret_cast<const eg_u32x2*>(vr + RB);
        const eg_u32x2 vc = *reinterpret_cast<const eg_u32x2*>(vr + 8 * RB), vd = *reinterpret_cast<const eg_u32x2*>(vr + 9 * RB);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t sel = (t % 4) | (4 + t % 4) << 4;
            const uint32_t vf[2] = {eg_e4m3x2_to_f16x2(t < 4 ? eg_prmt(va.x, vb.x, sel) : eg_prmt(va.y, vb.y, sel)),
                                    eg_e4m3x2_to_f16x2(t < 4 ? eg_prmt(vc.x, vd.x, sel) : eg_prmt(vc.y, vd.y, sel))};
            eg_mma_f16(o[t], pa, vf, o[t]);
        }
        }
    }
    eg_cp_async_wait<0>();
    __syncthreads();

    // Combine the KW key groups of each row tile in shared memory.
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
        const int r = mt * 16 + g + 8 * h;
        if (c == 0) { red_m[kw * MROWS + r] = m[h]; red_l[kw * MROWS + r] = l[h]; }
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                red_o[(kw * MROWS + r) * 64 + 8 * (2 * c + e) + t] = o[t][2 * h + e];
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

// attnQKF16 loads the F16 query fragments and computes a 16-key tile's
// logits with m16n8k16 F16 tiles, K widened from the raw bytes: b0 of
// n-tile nt holds dimensions 16ks + 2c, +1 of key kt0 + 8nt + g.
var attnQKF16 = [2]string{`    uint32_t qa[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LDQ + ks * 16 + (lane / 16) * 8]);
`, `#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned char* const kr = kst + (kt0 + nt * 8 + g) * RB + ks * 16 + 2 * c;
                const uint32_t kf[2] = {eg_e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(kr)),
                                        eg_e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(kr + 8))};
                eg_mma_f16(s[nt], qa[ks], kf, s[nt]);
            }
`}

// attnQKFP8 multiplies the E4M3 bytes of Q and K with m16n8k32 tiles. In
// the b16 view the k32 fragments have the k16 layout, so ldmatrix
// addresses them as F16 rows of 32 elements.
var attnQKFP8 = [2]string{`    uint32_t qa[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LDQ + ks * 16 + (lane / 16) * 8]);
`, `        const uint16_t* const sk = reinterpret_cast<const uint16_t*>(kst);
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kt0 + lane % 8 + 8 * (lane / 16)) * (RB / 2) + ks * 16 + 8 * ((lane / 8) % 2)]);
            eg_mma_e4m3(s[0], qa[ks], &kf[0], s[0]);
            eg_mma_e4m3(s[1], qa[ks], &kf[2], s[1]);
        }
`}

// attnCombine finishes: direct store for one split, else the two-level
// merge of the chunk partials.
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
        constexpr int OUTS = (ROWS * 64 + 127) / 128, CH = OUTS < 4 ? OUTS : 4;
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
            // Outputs in chunks of CH, so a 64-row tile does not hold all of
            // its OUTS accumulators and their loads at once.
#pragma unroll 1
            for (int k0 = 0; k0 < OUTS; k0 += CH) {
                float base[CH], acc[CH];
#pragma unroll
                for (int k = 0; k < CH; ++k) {
                    const int i = min(tid + 128 * (k0 + k), ROWS * 64 - 1);
                    base[k] = red_m[i / 64] == NEG_INF ? 0.0f : red_m[i / 64];
                    acc[k] = 0.0f;
                }
#pragma unroll 4
                for (int s = 0; s < n; ++s) {
#pragma unroll
                    for (int k = 0; k < CH; ++k) {
                        const int i = min(tid + 128 * (k0 + k), ROWS * 64 - 1);
                        const float* p = src + (size_t)s * ROWS * 66 + (i / 64) * 66;
                        acc[k] += exp2f(eg_ld_cg(p) - base[k]) * eg_ld_cg(p + 2 + i % 64);
                    }
                }
#pragma unroll
                for (int k = 0; k < CH; ++k) {
                    const int i = tid + 128 * (k0 + k);
                    if (k0 + k >= OUTS || i >= ROWS * 64) break;
                    if (final) fin[i] = acc[k] / red_l[i / 64] * vs;
                    else dst[(i / 64) * 66 + 2 + i % 64] = acc[k];
                }
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
