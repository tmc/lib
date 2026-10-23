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
// stores its (max, sum, output) partial in a workspace and the last block
// to finish combines them in chunk order, so results do not depend on
// scheduling.
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
}

type attnGeom struct {
	G, rows, mrows, mt, kw int
}

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
	if a.Chunk%64 != 0 {
		return a, g, fmt.Errorf("paged attention %s: chunk %d is not a multiple of 64", a.Name, a.Chunk)
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
	return a, g, nil
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
// maxKeys keys: Hkv·batch int32 counters, then the partials. The counters
// must be zero before the first launch; the kernel leaves them zero.
func (a PagedAttention) WorkspaceBytes(batch, maxKeys int) int {
	_, g, err := a.norm()
	if err != nil {
		return 0
	}
	return 4*batch*a.Hkv + 4*batch*a.Hkv*a.Splits(maxKeys)*g.rows*(2+HeadDim)
}

func (a PagedAttention) params() []string {
	return []string{
		"const " + a.Q.cType() + "* __restrict__ q",
		"const uint8_t* __restrict__ k_cache",
		"const uint8_t* __restrict__ v_cache",
		"const int32_t* __restrict__ page_table",
		"int max_pages",
		"const int32_t* __restrict__ seqlens",
		"const float* __restrict__ k_scale",
		"const float* __restrict__ v_scale",
		a.Out.cType() + "* __restrict__ out",
		"unsigned char* __restrict__ ws",
	}
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
	fmt.Fprintf(&b, "// %s: paged FP8 GQA attention, %d/%d heads, %d query tokens, window %d, page %d, %d-key splits.\n",
		a.Name, a.Hq, a.Hkv, a.QLen, a.Window, a.Page, a.Chunk)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n", a.Name, strings.Join(a.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, L = %d, ROWS = %d, MROWS = %d, MT = %d, KW = %d;\n",
		a.Hq, a.Hkv, g.G, a.QLen, g.rows, g.mrows, g.mt, g.kw)
	fmt.Fprintf(&b, "    constexpr int WINDOW = %d, PAGE = %d, CHUNK = %d, LD = 72, PIECES = KW * 64, NP = (PIECES + 127) / 128;\n",
		a.Window, a.Page, a.Chunk)
	fmt.Fprintf(&b, "    const float scale = %s * k_scale[0] * 1.4426950408889634f; // log2 domain\n",
		floatLiteral(a.QScale*a.SMScale))
	b.WriteString(attnBody)
	if a.Out == BF16 {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = eg_f32_to_bf16(v); };\n")
	} else {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = v; };\n")
	}
	b.WriteString(attnCombine)
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
    alignas(16) __shared__ uint16_t sk[KW * 16 * LD];
    alignas(16) __shared__ uint16_t sv[KW * 16 * LD];
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

    // Each thread moves NP 16-byte pieces (16 keys x 4 pieces per KW group).
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

    for (int kb = c0; kb < c1; kb += 16 * KW) {
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
        if (kb + 16 * KW < c1) fetch(kb + 16 * KW);
        const int key0 = kb + kw * 16;
        if (key0 >= c1) continue;
        float s[2][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kw * 16 + lane % 8 + 8 * (lane / 16)) * LD + ks * 16 + 8 * ((lane / 8) % 2)]);
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
            eg_ldmatrix_x4_trans(vf, &sv[(kw * 16 + lane % 8 + 8 * ((lane / 8) % 2)) * LD + nd * 16 + 8 * (lane / 16)]);
            eg_mma_bf16(o[2 * nd], pa, &vf[0], o[2 * nd]);
            eg_mma_bf16(o[2 * nd + 1], pa, &vf[2], o[2 * nd + 1]);
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
    unsigned char* const counters = ws;
    float* const parts = reinterpret_cast<float*>(ws + 4 * (size_t)gridDim.z * HKV);
    float* const mine = parts + (((size_t)b * HKV + kvh) * gridDim.x + split) * ROWS * 66;
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

// attnCombine finishes: direct store for one split, else the last block
// combines every split's partial in split order.
const attnCombine = `    if (nsplit == 1) {
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
            put(((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, acc / sum * vs);
        }
        return;
    }
    __threadfence();
    __syncthreads();
    int* const counter = reinterpret_cast<int*>(counters) + b * HKV + kvh;
    if (tid == 0) last = atomicAdd(counter, 1) == nsplit - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float* const all = parts + ((size_t)b * HKV + kvh) * gridDim.x * ROWS * 66;
    for (int i = tid; i < ROWS * 64; i += 128) {
        const int r = i / 64, d = i % 64;
        float mx = NEG_INF;
        for (int s = 0; s < nsplit; ++s) mx = fmaxf(mx, eg_ld_cg(all + (size_t)s * ROWS * 66 + r * 66));
        float sum = 0.0f, acc = 0.0f;
        for (int s = 0; s < nsplit; ++s) {
            const float* p = all + (size_t)s * ROWS * 66 + r * 66;
            const float w = exp2f(eg_ld_cg(p) - mx);
            sum += w * eg_ld_cg(p + 1);
            acc += w * eg_ld_cg(p + 2 + d);
        }
        put(((size_t)(b * L + r / G) * HQ + kvh * G + r % G) * 64 + d, acc / sum * vs);
    }
    if (tid == 0) *counter = 0;
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
	fmt.Fprintf(&b, "    if (grid.x > 0 && batch > 0) %s<<<grid, block, 0, stream>>>(%s);\n}\n", a.Name, strings.Join(names, ", "))
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
}

// Reference computes the attention output [batch·QLen, Hq, 64] in float64
// from query values q (as stored, before QScale), following
// moe8_attention_decode/reference.py:216-236.
func (a PagedAttention) Reference(q []float32, c PagedCache) ([]float64, error) {
	a, g, err := a.norm()
	if err != nil {
		return nil, err
	}
	batch := len(c.SeqLens)
	if len(q) != batch*a.QLen*a.Hq*HeadDim || len(c.PageTable) != batch*c.MaxPages {
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
			}
		}
	}
	return out, nil
}
