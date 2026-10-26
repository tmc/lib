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
	Chunk   int     // keys per block, a multiple of 64; 0 means 256 (128 with DeviceSplits)

	// KVHeadsPerBlock is 1 (the default for 0) or 2. With 2, one block
	// serves two adjacent KV heads: their query rows (at most 8 per head)
	// fill one 16-row tile, each key is one 128-byte row of K and of V, Q·Kᵀ
	// runs over both heads' 128 dimensions with the other head's query
	// dimensions zero, and P·V keeps only each row's own head. It halves
	// the blocks, page-table reads and partials.
	KVHeadsPerBlock int

	// DeviceSplits sizes the splits at launch: the launcher gives each
	// (request, KV head) enough splits to fill the device's resident
	// blocks once ([PagedAttention.PlanSplits]), with at least Chunk keys
	// per split. The kernel then takes keys per split as a parameter.
	DeviceSplits bool

	// Depth is the number of 16-key tiles each warp computes per step: 1
	// (the default for 0) or 2.
	Depth int

	// Stages is the depth of the ring that stages K and V bytes in shared
	// memory, in steps: 2 to 8. The block copies a step's keys with
	// cp.async Stages-1 steps before it uses them. 0 means the most, up to
	// 4, that fit 40 KiB.
	Stages int

	// TableAhead is the depth of the register ring that prefetches
	// page-table entries ahead of the cp.async copies that consume them:
	// 1 to Stages. 0 or 1 prefetches one step ahead; TableAhead >= 2 keeps
	// TableAhead steps of page-table entries in registers so the cp.async
	// of step s + (Stages - 1) never waits on a dependent page-table load.
	TableAhead int

	// QKFP8 computes Q·Kᵀ with mma.sync m16n8k32 E4M3 tiles on the stored
	// bytes (Q must be E4M3), skipping the widening of K to F16. Products
	// are exact; the tensor core's FP32 accumulation order differs.
	QKFP8 bool

	// WGMMA computes Q·Kᵀ over 64 keys per step with sm_90a warpgroup MMA
	// (wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3) across the
	// 128-thread warpgroup (Q must be E4M3 and KVHeadsPerBlock 1), mapping
	// the block's query rows (up to 64) across the 4 warps (16 rows per
	// warp) and computing online softmax and P·V over all 64 keys.
	WGMMA bool

	// Swizzle stages K and V rows in unpadded 64·KVHeadsPerBlock-byte shared
	// memory rows with 16-byte chunks XOR-swizzled by row & (KP - 1).
	Swizzle bool

	// ClusterCombine launches splits in 8-CTA clusters (__cluster_dims__(8, 1, 1))
	// and folds each cluster's 8 split partials in distributed shared memory
	// before writing one partial per cluster to the global workspace.
	ClusterCombine bool

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

// attnGeom is a kernel's layout: G query heads per KV head, rows query
// rows per block (partials and outputs), a 16·mt-row tile (mrows) per
// warp group, kw warps splitting each step's keys, hpb KV heads per block
// and rb shared memory bytes per staged K or V row.
type attnGeom struct {
	G, rows, mrows, mt, kw, hpb, rb int
}

// geom returns a's layout; a's fields must be valid.
func (a PagedAttention) geom() attnGeom {
	g := attnGeom{G: a.Hq / a.Hkv, hpb: max(a.KVHeadsPerBlock, 1)}
	g.rows = g.hpb * g.G * a.QLen
	g.mt = 4
	switch {
	case a.WGMMA:
		g.mrows, g.kw, g.rb = 64, 4, 64
		return g
	case g.hpb == 2 || g.rows <= 16:
		g.mt = 1
	case g.rows <= 32:
		g.mt = 2
	}
	g.mrows, g.kw = 16*g.mt, 4/g.mt
	// 64·hpb bytes padded so ldmatrix rows fall in distinct banks, or
	// unpadded when 16-byte chunks are XOR-swizzled.
	g.rb = 64*g.hpb + 16
	if a.Swizzle {
		g.rb = 64 * g.hpb
	}
	return g
}

// stageBytes returns the bytes of one ring stage: a step's K and V rows.
func (g attnGeom) stageBytes(depth int) int { return 2 * 16 * g.kw * depth * g.rb }

// qStride returns the shared memory stride of a query row in uint16s.
func (g attnGeom) qStride(qkfp8 bool) int {
	if qkfp8 {
		return 32*g.hpb + 8 // E4M3 bytes
	}
	return 64*g.hpb + 8 // F16
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
	if rows := a.Hq / a.Hkv * a.QLen; a.QLen < 1 || a.QLen > 8 || rows > 64 {
		return a, g, fmt.Errorf("paged attention %s: %d query rows per KV head (QLen %d), want at most 64", a.Name, rows, a.QLen)
	}
	switch a.KVHeadsPerBlock {
	case 0, 1:
	case 2:
		if a.Hkv%2 != 0 || a.Hq/a.Hkv*a.QLen > 8 {
			return a, g, fmt.Errorf("paged attention %s: 2 KV heads per block need an even Hkv and at most 8 query rows per KV head", a.Name)
		}
	default:
		return a, g, fmt.Errorf("paged attention %s: %d KV heads per block, want 1 or 2", a.Name, a.KVHeadsPerBlock)
	}
	if a.Window < 0 || a.Page < 1 {
		return a, g, fmt.Errorf("paged attention %s: bad window %d or page %d", a.Name, a.Window, a.Page)
	}
	if a.WGMMA && (a.Q != E4M3 || a.KVHeadsPerBlock > 1) {
		return a, g, fmt.Errorf("paged attention %s: WGMMA needs E4M3 queries and 1 KV head per block", a.Name)
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
		if a.DeviceSplits {
			a.Chunk = 128
		}
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
	g = a.geom()
	if a.Stages == 0 {
		a.Stages = max(2, min(4, 40<<10/g.stageBytes(a.Depth)))
	}
	if a.Stages < 2 || a.Stages > 8 {
		return a, g, fmt.Errorf("paged attention %s: %d stages, want 2 to 8", a.Name, a.Stages)
	}
	if a.TableAhead == 0 {
		a.TableAhead = 1
	}
	if a.TableAhead < 1 || a.TableAhead > a.Stages {
		return a, g, fmt.Errorf("paged attention %s: table ahead %d, want 1 to %d", a.Name, a.TableAhead, a.Stages)
	}
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
	g := a.geom()
	stages := a.Stages
	if stages == 0 {
		stages = max(2, min(4, 40<<10/g.stageBytes(max(a.Depth, 1))))
	}
	kw := g.kw
	if a.WGMMA {
		kw = 1
	}
	ring := max(stages*g.stageBytes(max(a.Depth, 1)), 4*kw*g.mrows*HeadDim) // red_o reuses the ring
	if a.WGMMA {
		sqBytes := 64 * 64
		if a.Swizzle {
			sqBytes = 64 * 128
		}
		return sqBytes + 8*g.mrows + 4 + ring
	}
	return 2*g.mrows*g.qStride(a.QKFP8) + 8*g.kw*g.mrows + 4 + ring
}

// Splits returns the number of key chunks for a request attending maxKeys
// keys: the grid's x dimension, or with DeviceSplits its largest value.
// maxKeys is at most Window+QLen-1 for a windowed layer, else the longest
// cache length.
func (a PagedAttention) Splits(maxKeys int) int {
	a, _, err := a.norm()
	if err != nil || maxKeys < 1 {
		return 0
	}
	s := (maxKeys + a.Chunk - 1) / a.Chunk
	if a.ClusterCombine {
		s = (s + ClusterSize - 1) / ClusterSize * ClusterSize
	}
	return s
}

// PlanSplits returns the keys per split and the splits (grid x) that a
// DeviceSplits launcher picks for batch requests of at most maxKeys keys on
// a device with slots resident blocks (SMs times blocks per SM): the
// fewest keys per split, a multiple of 64 and at least Chunk, that keep
// batch·Hkv·splits within slots.
func (a PagedAttention) PlanSplits(batch, maxKeys, slots int) (chunk, splits int) {
	a, _, err := a.norm()
	if err != nil || batch < 1 || maxKeys < 1 {
		return 0, 0
	}
	per := max(1, slots/(batch*a.Hkv/a.geom().hpb))
	chunk = max(a.Chunk, ((maxKeys+per-1)/per+63)/64*64)
	splits = (maxKeys + chunk - 1) / chunk
	if a.ClusterCombine {
		splits = (splits + ClusterSize - 1) / ClusterSize * ClusterSize
	}
	return chunk, splits
}

// WorkspaceBytes returns the workspace size for batch requests of at most
// maxKeys keys: for each (request, block of KV heads) 1 + ceil(splits/16) int32
// counters, then the chunk and group partials. The counters must be zero
// before a launch; the kernel leaves them zero. Their offsets depend on
// the grid, so clear them again when batch or maxKeys changes.
func (a PagedAttention) WorkspaceBytes(batch, maxKeys int) int {
	_, g, err := a.norm()
	if err != nil {
		return 0
	}
	s := a.Splits(maxKeys)
	if a.ClusterCombine {
		s /= ClusterSize
	}
	ng := (s + attnGroup - 1) / attnGroup
	units := batch * a.Hkv / g.hpb
	return 4*units*(1+ng) + 4*units*(s+ng)*g.rows*(2+HeadDim)
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
	if a.DeviceSplits {
		p = append(p[:5], append([]string{"int chunk"}, p[5:]...)...)
	}
	if a.Gate {
		p = append(p, "const uint16_t* __restrict__ gate")
	}
	return append(p, a.Out.cType()+"* __restrict__ out", "unsigned char* __restrict__ ws")
}

// launchParams returns the launcher's parameters after batch and max_keys:
// the kernel's, less the keys per split the launcher computes.
func (a PagedAttention) launchParams() []string {
	var p []string
	for _, s := range a.params() {
		if s != "int chunk" {
			p = append(p, s)
		}
	}
	return p
}

// Grid returns the grid and block for batch requests of at most maxKeys
// keys; with DeviceSplits, the largest grid the launcher picks.
func (a PagedAttention) Grid(batch, maxKeys int) (grid, block [3]int) {
	return [3]int{a.Splits(maxKeys), a.Hkv / a.geom().hpb, batch}, [3]int{128, 1, 1}
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
	if a.WGMMA {
		return a.cudaWGMMA(g), nil
	}
	var b strings.Builder
	if a.ClusterCombine {
		b.WriteString(ClusterPrelude)
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "// %s: paged FP8 GQA attention, %d/%d heads, %d query tokens, window %d, page %d, %d-key splits, depth %d, %d stages.\n",
		a.Name, a.Hq, a.Hkv, a.QLen, a.Window, a.Page, a.Chunk, a.Depth, a.Stages)
	if a.ClusterCombine {
		fmt.Fprintf(&b, "__global__ void __cluster_dims__(%d, 1, 1) __launch_bounds__(128, 1) %s(\n    %s) {\n",
			ClusterSize, a.Name, strings.Join(a.params(), ",\n    "))
	} else {
		fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n", a.Name, strings.Join(a.params(), ",\n    "))
	}
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, L = %d, ROWS = %d, MROWS = %d, MT = %d, KW = %d;\n",
		a.Hq, a.Hkv, g.G, a.QLen, g.rows, g.mrows, g.mt, g.kw)
	fmt.Fprintf(&b, "    constexpr int HPB = %d, HKVB = HKV / HPB, ROWS1 = G * L, D = 64 * HPB, KP = D / 16; // KV heads per block, blocks per request, rows per KV head, K/V row bytes and pieces\n", g.hpb)
	fmt.Fprintf(&b, "    constexpr int WINDOW = %d, PAGE = %d, DEPTH = %d, STEP = 16 * KW * DEPTH, STAGES = %d;\n",
		a.Window, a.Page, a.Depth, a.Stages)
	if a.DeviceSplits {
		b.WriteString("    const int CHUNK = chunk;\n")
	} else {
		fmt.Fprintf(&b, "    constexpr int CHUNK = %d;\n", a.Chunk)
	}
	fmt.Fprintf(&b, "    constexpr int PIECES = STEP * KP, NP = (PIECES + 127) / 128, GROUP = %d, LDQ = %d, RB = %d;\n",
		attnGroup, g.qStride(a.QKFP8), g.rb)
	b.WriteString("    constexpr int STAGE_BYTES = 2 * STEP * RB, RING_BYTES = STAGES * STAGE_BYTES > KW * MROWS * 256 ? STAGES * STAGE_BYTES : KW * MROWS * 256;\n")
	if a.TableAhead >= 2 {
		fmt.Fprintf(&b, "    constexpr int TABLE_AHEAD = %d;\n", a.TableAhead)
	}
	fmt.Fprintf(&b, "    const float scale = %s * k_scale[0] * 1.4426950408889634f; // log2 domain\n",
		floatLiteral(a.QScale*a.SMScale))
	if a.ClusterCombine {
		b.WriteString(attnHeadCluster)
	} else {
		b.WriteString(attnHead)
	}
	qk := attnQKF16
	if a.Swizzle {
		qk = attnQKF16Swizzle
	}
	if a.QKFP8 {
		b.WriteString(attnQFP8)
		qk = attnQKFP8
		if a.Swizzle {
			qk = attnQKFP8Swizzle
		}
	} else {
		b.WriteString(attnQF16)
	}
	table := attnTableAhead1
	if a.TableAhead >= 2 {
		table = attnTableAheadN
	}
	issue := attnIssue
	vload := attnVLoad
	if a.Swizzle {
		issue = attnIssueSwizzle
		vload = attnVLoadSwizzle
	}
	nsteps := "    const int nsteps = (c1 - c0 + STEP - 1) / STEP;\n"
	tail := attnTail
	if a.ClusterCombine {
		nsteps = "    const int nsteps = c1 > c0 ? (c1 - c0 + STEP - 1) / STEP : 0;\n"
		tail = attnClusterTail
	}
	b.WriteString(strings.NewReplacer(
		"ISSUE_STAGE\n", issue,
		"NSTEPS_DECL\n", nsteps,
		"QK_LOAD\n", qk[0],
		"QK_TILE\n", qk[1],
		"V_LOAD\n", vload,
		"TABLE_PREFETCH\n", table[0],
		"TABLE_STEP\n", table[1],
		"LOOP_TAIL\n", tail,
	).Replace(attnLoop))
	if a.Out == BF16 {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = eg_f32_to_bf16(v); };\n")
	} else {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = v; };\n")
	}
	if a.ClusterCombine {
		b.WriteString(attnClusterCombine)
	} else {
		b.WriteString(attnCombine)
	}
	if a.Gate {
		fmt.Fprintf(&b, "    constexpr int GATE_STRIDE = %d, GATE_OFFSET = %d;\n    constexpr float GATE_EPS = %s;\n",
			a.GateStride, a.GateOffset, floatLiteral(a.GateEps))
		b.WriteString(attnGateStore)
	} else {
		b.WriteString(attnStore)
	}
	if a.ClusterCombine {
		fmt.Fprintf(&b, "EG_REGISTER_CLUSTER(%s)\n", a.Name)
	}
	return b.String(), nil
}

// attnHead declares the block's coordinates. Partial row pr is query row
// j = pr % ROWS1 (token j / G, head (kvh0 + pr / ROWS1)·G + j % G); its tile
// row is pr, or with two KV heads per block j + 8·(pr / ROWS1).
const attnHead = `    const int split = blockIdx.x, kvb = blockIdx.y, kvh0 = kvb * HPB, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane / 4, c = lane % 4;
    const int mt = warp % MT, kw = warp / MT;
    const int seqlen = seqlens[b], p0 = seqlen - L;
    const int kbegin = WINDOW ? max(0, p0 - WINDOW + 1) : 0;
    const int nsplit = (seqlen - kbegin + CHUNK - 1) / CHUNK;
    if (split >= nsplit) return;
    const int c0 = kbegin + split * CHUNK, c1 = min(seqlen, c0 + CHUNK);
    const float NEG_INF = -__uint_as_float(0x7f800000u);
    auto trow = [](int pr) { return HPB == 1 ? pr : pr / ROWS1 * 8 + pr % ROWS1; };
    auto qrow = [&](int pr) { // [token, head] index of partial row pr
        const int j = pr % ROWS1;
        return (size_t)(b * L + j / G) * HQ + (kvh0 + pr / ROWS1) * G + j % G;
    };

    alignas(16) __shared__ uint16_t sq[MROWS * LDQ];
    __shared__ float red_m[KW * MROWS], red_l[KW * MROWS];
    __shared__ int last;
    // The cp.async ring: STAGES stages of STEP K rows then STEP V rows of RB
    // bytes; after the main loop, red_o.
    alignas(16) __shared__ unsigned char ring[RING_BYTES];
    float* const red_o = reinterpret_cast<float*>(ring);
`

const attnHeadCluster = `    const int split = blockIdx.x, kvb = blockIdx.y, kvh0 = kvb * HPB, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane / 4, c = lane % 4;
    const int mt = warp % MT, kw = warp / MT;
    const int seqlen = seqlens[b], p0 = seqlen - L;
    const int kbegin = WINDOW ? max(0, p0 - WINDOW + 1) : 0;
    const int nsplit = (seqlen - kbegin + CHUNK - 1) / CHUNK;
    const int rank = eg_cluster_rank(), cl = split / 8, ncl = (nsplit + 7) / 8;
    if (cl >= ncl) return;
    const int c0 = kbegin + split * CHUNK, c1 = min(seqlen, c0 + CHUNK);
    const float NEG_INF = -__uint_as_float(0x7f800000u);
    auto trow = [](int pr) { return HPB == 1 ? pr : pr / ROWS1 * 8 + pr % ROWS1; };
    auto qrow = [&](int pr) { // [token, head] index of partial row pr
        const int j = pr % ROWS1;
        return (size_t)(b * L + j / G) * HQ + (kvh0 + pr / ROWS1) * G + j % G;
    };

    EG_SHARED(uint16_t, sq, [MROWS * LDQ]);
    EG_SHARED(float, red_m, [KW * MROWS]);
    EG_SHARED(float, red_l, [KW * MROWS]);
    EG_SHARED(int, last, [1]);
    // The cp.async ring: STAGES stages of STEP K rows then STEP V rows of RB
    // bytes; after the main loop, red_o.
    EG_SHARED(unsigned char, ring, [RING_BYTES]);
    float* const red_o = reinterpret_cast<float*>(ring);
`

// attnQF16 and attnQFP8 stage the query rows in sq as F16 values or E4M3
// bytes: tile row r holds its head's 64 dimensions at 64·hp and zeros
// elsewhere, as do rows that hold no query.
const attnQF16 = `    for (int i = tid; i < MROWS * (D / 8); i += 128) {
        const int r = i / (D / 8), d = (i % (D / 8)) * 8, hp = HPB == 1 ? 0 : r / 8, j = HPB == 1 ? r : r % 8;
        eg_u32x4 w = {0, 0, 0, 0};
        if (j < ROWS1 && d / 64 == hp) {
            float f[8];
            eg_load<8>(q + qrow(hp * ROWS1 + j) * 64 + d % 64, f);
            w = eg_u32x4{eg_pack_f16x2(f[0], f[1]), eg_pack_f16x2(f[2], f[3]),
                         eg_pack_f16x2(f[4], f[5]), eg_pack_f16x2(f[6], f[7])};
        }
        *reinterpret_cast<eg_u32x4*>(&sq[r * LDQ + d]) = w;
    }
`

const attnQFP8 = `    for (int i = tid; i < MROWS * KP; i += 128) {
        const int r = i / KP, piece = i % KP, hp = HPB == 1 ? 0 : r / 8, j = HPB == 1 ? r : r % 8;
        eg_u32x4 w = {0, 0, 0, 0};
        if (j < ROWS1 && piece / 4 == hp) w = eg_ldg16(q + qrow(hp * ROWS1 + j) * 64 + piece % 4 * 16);
        *reinterpret_cast<eg_u32x4*>(&sq[r * LDQ + piece * 8]) = w;
    }
`

const attnIssue = `            unsigned char* const dk = kd + (piece / KP) * RB + (piece % KP) * 16;
`

const attnIssueSwizzle = `            unsigned char* const dk = kd + (piece / KP) * RB + ((piece % KP) ^ ((piece / KP) & (KP - 1))) * 16;
`

const attnVLoad = `        const unsigned char* const vr = vst + (kt0 + 2 * c) * RB + 64 * hb + 8 * g;
        const eg_u32x2 va = *reinterpret_cast<const eg_u32x2*>(vr), vb = *reinterpret_cast<const eg_u32x2*>(vr + RB);
        const eg_u32x2 vc = *reinterpret_cast<const eg_u32x2*>(vr + 8 * RB), vd = *reinterpret_cast<const eg_u32x2*>(vr + 9 * RB);
`

const attnVLoadSwizzle = `        const unsigned char* const vr0 = vst + (kt0 + 2 * c) * RB + ((4 * hb + (g >> 1)) ^ ((2 * c) & (KP - 1))) * 16 + (g & 1) * 8;
        const unsigned char* const vr1 = vst + (kt0 + 2 * c + 1) * RB + ((4 * hb + (g >> 1)) ^ ((2 * c + 1) & (KP - 1))) * 16 + (g & 1) * 8;
        const eg_u32x2 va = *reinterpret_cast<const eg_u32x2*>(vr0), vb = *reinterpret_cast<const eg_u32x2*>(vr1);
        const eg_u32x2 vc = *reinterpret_cast<const eg_u32x2*>(vr0 + 8 * RB), vd = *reinterpret_cast<const eg_u32x2*>(vr1 + 8 * RB);
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
const attnLoop = `    const int fk = tid * NP / KP;
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
ISSUE_STAGE
            if (row >= 0) {
                const size_t at = ((size_t)row * HKV + kvh0) * 64 + (piece % KP) * 16;
                eg_cp_async16(dk, k_cache + at);
                eg_cp_async16(dk + STEP * RB, v_cache + at);
            } else {
                // Past the chunk's end P is 0, and 0·NaN is NaN: zero V.
                // K needs nothing; its logits are masked.
                *reinterpret_cast<eg_u32x4*>(dk + STEP * RB) = eg_u32x4{0, 0, 0, 0};
            }
        }
    };
NSTEPS_DECL
TABLE_PREFETCH
    __syncthreads();

QK_LOAD
    // This lane's two rows (g and g+8 of its tile): positions and first keys.
    int pos[2], lo[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = mt * 16 + g + 8 * h, j = HPB == 1 ? r : r % 8;
        pos[h] = p0 + min(j / G, L - 1);
        lo[h] = WINDOW ? max(0, pos[h] - WINDOW + 1) : 0;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f}, o[8][4] = {};

    for (int it = 0; it < nsteps; ++it) {
        eg_cp_async_wait<STAGES - 2>();
        __syncthreads();
TABLE_STEP
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
        // With two KV heads, rows g (head kvh0) take V's first 64 bytes
        // and rows g + 8 the second: each mma keeps one row half and
        // discards the other.
#pragma unroll
        for (int hb = 0; hb < HPB; ++hb) {
V_LOAD
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t sel = (t % 4) | (4 + t % 4) << 4;
            const uint32_t vf[2] = {eg_e4m3x2_to_f16x2(t < 4 ? eg_prmt(va.x, vb.x, sel) : eg_prmt(va.y, vb.y, sel)),
                                    eg_e4m3x2_to_f16x2(t < 4 ? eg_prmt(vc.x, vd.x, sel) : eg_prmt(vc.y, vd.y, sel))};
            if (HPB == 1) {
                eg_mma_f16(o[t], pa, vf, o[t]);
            } else {
                float d[4];
                eg_mma_f16(d, pa, vf, o[t]);
                if (hb == 0) { o[t][0] = d[0]; o[t][1] = d[1]; } else { o[t][2] = d[2]; o[t][3] = d[3]; }
            }
        }
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
LOOP_TAIL
`

const attnTail = `    // Per (request, block of KV heads): 1 + NG counters, then gridDim.x
    // chunk partials and NG group partials of ROWS x (max, sum, 64 outputs).
    const int NG = (gridDim.x + GROUP - 1) / GROUP;
    int* const cnt = reinterpret_cast<int*>(ws) + ((size_t)b * HKVB + kvb) * (1 + NG);
    float* const all = reinterpret_cast<float*>(ws + 4 * (size_t)gridDim.z * HKVB * (1 + NG)) +
                       ((size_t)b * HKVB + kvb) * (gridDim.x + NG) * ROWS * 66;
    float* const mine = all + (size_t)split * ROWS * 66;
    for (int i = tid; i < ROWS * 64 && nsplit > 1; i += 128) {
        const int r = trow(i / 64), d = i % 64;
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
        mine[i / 64 * 66 + 2 + d] = acc;
        if (d == 0) { mine[i / 64 * 66] = mx; mine[i / 64 * 66 + 1] = sum; }
    }
`

const attnClusterTail = `    for (int i = tid; i < ROWS * 64; i += 128) {
        const int r = trow(i / 64), d = i % 64;
        float mx = NEG_INF, acc = 0.0f;
        if (split < nsplit) {
#pragma unroll
            for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_m[k * MROWS + r]);
            const float base = mx == NEG_INF ? 0.0f : mx;
#pragma unroll
            for (int k = 0; k < KW; ++k)
                acc += exp2f(red_m[k * MROWS + r] - base) * red_o[(k * MROWS + r) * 64 + d];
        }
        red_o[r * 64 + d] = acc;
    }
    __syncthreads();
    if (tid < ROWS) {
        const int r = trow(tid);
        float mx = NEG_INF, sum = 0.0f;
        if (split < nsplit) {
#pragma unroll
            for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_m[k * MROWS + r]);
            const float base = mx == NEG_INF ? 0.0f : mx;
#pragma unroll
            for (int k = 0; k < KW; ++k)
                sum += exp2f(red_m[k * MROWS + r] - base) * red_l[k * MROWS + r];
        }
        red_m[r] = mx;
        red_l[r] = sum;
    }
    eg_cluster_sync();
`

// attnTableAhead1 and attnTableAheadN issue the prologue and per-step
// page-table lookups and cp.async copies for TableAhead == 1 and
// TableAhead >= 2. With TableAhead >= 2, pg[TABLE_AHEAD] holds the next
// TABLE_AHEAD steps' cache rows in registers.
var attnTableAhead1 = [2]string{`#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nsteps) issue(s, rowof(c0 + s * STEP));
        eg_cp_async_commit();
    }
    long long next = rowof(c0 + (STAGES - 1) * STEP);
`, `        if (it + STAGES - 1 < nsteps) {
            issue((it + STAGES - 1) % STAGES, next);
            next = rowof(c0 + (it + STAGES) * STEP);
        }
`}

var attnTableAheadN = [2]string{`    long long pg[TABLE_AHEAD];
#pragma unroll
    for (int s = 0; s < STAGES - 1 + TABLE_AHEAD; ++s) {
        const long long r = s < nsteps ? rowof(c0 + s * STEP) : -1;
        if (s < STAGES - 1) {
            if (s < nsteps) issue(s, r);
            eg_cp_async_commit();
        } else {
            pg[(s - (STAGES - 1)) % TABLE_AHEAD] = r;
        }
    }
`, `        if (it + STAGES - 1 < nsteps) {
            const int slot = it % TABLE_AHEAD;
            issue((it + STAGES - 1) % STAGES, pg[slot]);
            pg[slot] = it + STAGES - 1 + TABLE_AHEAD < nsteps ? rowof(c0 + (it + STAGES - 1 + TABLE_AHEAD) * STEP) : -1;
        }
`}

// attnQKF16 loads the F16 query fragments and computes a 16-key tile's
// logits over the D staged dimensions with m16n8k16 F16 tiles, K widened
// from the raw bytes: b0 of n-tile nt holds dimensions 16ks + 2c, +1 of
// key kt0 + 8nt + g.
var attnQKF16 = [2]string{`    uint32_t qa[KP][4];
#pragma unroll
    for (int ks = 0; ks < KP; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LDQ + ks * 16 + (lane / 16) * 8]);
`, `#pragma unroll
        for (int ks = 0; ks < KP; ++ks)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned char* const kr = kst + (kt0 + nt * 8 + g) * RB + ks * 16 + 2 * c;
                const uint32_t kf[2] = {eg_e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(kr)),
                                        eg_e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(kr + 8))};
                eg_mma_f16(s[nt], qa[ks], kf, s[nt]);
            }
`}

var attnQKF16Swizzle = [2]string{attnQKF16[0], `#pragma unroll
        for (int ks = 0; ks < KP; ++ks)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const unsigned char* const kr = kst + (kt0 + nt * 8 + g) * RB + (ks ^ (g & (KP - 1))) * 16 + 2 * c;
                const uint32_t kf[2] = {eg_e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(kr)),
                                        eg_e4m3x2_to_f16x2(*reinterpret_cast<const uint16_t*>(kr + 8))};
                eg_mma_f16(s[nt], qa[ks], kf, s[nt]);
            }
`}

// attnQKFP8 multiplies the E4M3 bytes of Q and K with m16n8k32 tiles. In
// the b16 view the k32 fragments have the k16 layout, so ldmatrix
// addresses them as F16 rows of 32 elements.
var attnQKFP8 = [2]string{`    uint32_t qa[KP / 2][4];
#pragma unroll
    for (int ks = 0; ks < KP / 2; ++ks) eg_ldmatrix_x4(qa[ks], &sq[(mt * 16 + lane % 16) * LDQ + ks * 16 + (lane / 16) * 8]);
`, `        const uint16_t* const sk = reinterpret_cast<const uint16_t*>(kst);
#pragma unroll
        for (int ks = 0; ks < KP / 2; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kt0 + lane % 8 + 8 * (lane / 16)) * (RB / 2) + ks * 16 + 8 * ((lane / 8) % 2)]);
            eg_mma_e4m3(s[0], qa[ks], &kf[0], s[0]);
            eg_mma_e4m3(s[1], qa[ks], &kf[2], s[1]);
        }
`}

var attnQKFP8Swizzle = [2]string{attnQKFP8[0], `        const uint16_t* const sk = reinterpret_cast<const uint16_t*>(kst);
#pragma unroll
        for (int ks = 0; ks < KP / 2; ++ks) {
            uint32_t kf[4];
            eg_ldmatrix_x4(kf, &sk[(kt0 + lane % 8 + 8 * (lane / 16)) * (RB / 2) + ((2 * ks + (lane / 8) % 2) ^ (lane & (KP - 1))) * 8]);
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
            const int r = trow(i / 64), d = i % 64;
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
            fin[i] = acc / sum * vs;
        }
    } else {
        // merge combines n partials at src in chunk order: into fin (scaled
        // by vs) when final, else into the partial at dst. Each thread owns
        // output pairs; for a batch of NB partials it issues every load of
        // (max, sum) and of the pair at once, then folds the batch into a
        // running max, sum and pair.
        constexpr int PAIRS = ROWS * 32, NB = 8;
        auto merge = [&](const float* src, int n, float* dst, bool final) {
#pragma unroll 1
            for (int i = tid; i < PAIRS; i += 128) {
                const int r = i / 32, d = 2 * (i % 32);
                float mx = NEG_INF, sum = 0.0f;
                float2 acc = {0.0f, 0.0f};
#pragma unroll 1
                for (int s0 = 0; s0 < n; s0 += NB) {
                    float2 ml[NB], v[NB];
#pragma unroll
                    for (int s = 0; s < NB; ++s) {
                        const float* p = src + (size_t)min(s0 + s, n - 1) * ROWS * 66 + r * 66;
                        ml[s] = eg_ld_cg2(p);
                        v[s] = eg_ld_cg2(p + 2 + d);
                    }
                    float bm = mx;
#pragma unroll
                    for (int s = 0; s < NB; ++s) bm = fmaxf(bm, ml[s].x);
                    const float base = bm == NEG_INF ? 0.0f : bm, c = exp2f(mx - base);
                    sum *= c;
                    acc.x *= c;
                    acc.y *= c;
#pragma unroll
                    for (int s = 0; s < NB; ++s) {
                        // Clamped loads past n repeat partial n-1: weigh them 0.
                        const float w = s0 + s < n ? exp2f(ml[s].x - base) : 0.0f;
                        sum += w * ml[s].y;
                        acc.x += w * v[s].x;
                        acc.y += w * v[s].y;
                    }
                    mx = bm;
                }
                if (final) {
                    fin[r * 64 + d] = acc.x / sum * vs;
                    fin[r * 64 + d + 1] = acc.y / sum * vs;
                } else {
                    dst[r * 66 + 2 + d] = acc.x;
                    dst[r * 66 + 3 + d] = acc.y;
                    if (d == 0) { dst[r * 66] = mx; dst[r * 66 + 1] = sum; }
                }
            }
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

// attnClusterCombine folds the 8 peer partials in cluster DSMEM on rank 0
// into fin (when a single cluster covers the request) or into one cluster
// partial in global workspace, then runs the two-level global merge across
// cluster partials.
const attnClusterCombine = `    const int NPART = gridDim.x / 8, NG = (NPART + GROUP - 1) / GROUP;
    int* const cnt = reinterpret_cast<int*>(ws) + ((size_t)b * HKVB + kvb) * (1 + NG);
    float* const all = reinterpret_cast<float*>(ws + 4 * (size_t)gridDim.z * HKVB * (1 + NG)) +
                       ((size_t)b * HKVB + kvb) * (NPART + NG) * ROWS * 66;
    float* const mine = all + (size_t)cl * ROWS * 66;
    float* const fin = red_o;
    if (rank == 0) {
        for (int i = tid; i < ROWS * 64; i += 128) {
            const int r = trow(i / 64), d = i % 64;
            float mx = NEG_INF, ml[8], v[8];
#pragma unroll
            for (int rk = 0; rk < 8; ++rk) {
                const float rm = eg_map_shared_rank(red_m, rk)[r];
                ml[rk] = rm;
                mx = fmaxf(mx, rm);
                v[rk] = eg_map_shared_rank(red_o, rk)[r * 64 + d];
            }
            const float base = mx == NEG_INF ? 0.0f : mx;
            float sum = 0.0f, acc = 0.0f;
#pragma unroll
            for (int rk = 0; rk < 8; ++rk) {
                const float w = exp2f(ml[rk] - base);
                sum += w * eg_map_shared_rank(red_l, rk)[r];
                acc += w * v[rk];
            }
            if (ncl == 1) {
                fin[i] = acc / sum * vs;
            } else {
                mine[i / 64 * 66 + 2 + d] = acc;
                if (d == 0) { mine[i / 64 * 66] = mx; mine[i / 64 * 66 + 1] = sum; }
            }
        }
    }
    eg_cluster_sync();
    if (rank != 0) return;
    if (ncl > 1) {
        constexpr int PAIRS = ROWS * 32, NB = 8;
        auto merge = [&](const float* src, int n, float* dst, bool final) {
#pragma unroll 1
            for (int i = tid; i < PAIRS; i += 128) {
                const int r = i / 32, d = 2 * (i % 32);
                float mx = NEG_INF, sum = 0.0f;
                float2 acc = {0.0f, 0.0f};
#pragma unroll 1
                for (int s0 = 0; s0 < n; s0 += NB) {
                    float2 ml[NB], v[NB];
#pragma unroll
                    for (int s = 0; s < NB; ++s) {
                        const float* p = src + (size_t)min(s0 + s, n - 1) * ROWS * 66 + r * 66;
                        ml[s] = eg_ld_cg2(p);
                        v[s] = eg_ld_cg2(p + 2 + d);
                    }
                    float bm = mx;
#pragma unroll
                    for (int s = 0; s < NB; ++s) bm = fmaxf(bm, ml[s].x);
                    const float base = bm == NEG_INF ? 0.0f : bm, c = exp2f(mx - base);
                    sum *= c;
                    acc.x *= c;
                    acc.y *= c;
#pragma unroll
                    for (int s = 0; s < NB; ++s) {
                        const float w = s0 + s < n ? exp2f(ml[s].x - base) : 0.0f;
                        sum += w * ml[s].y;
                        acc.x += w * v[s].x;
                        acc.y += w * v[s].y;
                    }
                    mx = bm;
                }
                if (final) {
                    fin[r * 64 + d] = acc.x / sum * vs;
                    fin[r * 64 + d + 1] = acc.y / sum * vs;
                } else {
                    dst[r * 66 + 2 + d] = acc.x;
                    dst[r * 66 + 3 + d] = acc.y;
                    if (d == 0) { dst[r * 66] = mx; dst[r * 66 + 1] = sum; }
                }
            }
        };
        const int ng = (ncl + GROUP - 1) / GROUP, grp = cl / GROUP;
        const int gn = min(GROUP, ncl - grp * GROUP);
        __threadfence();
        __syncthreads();
        if (tid == 0) last[0] = atomicAdd(cnt + 1 + grp, 1) == gn - 1;
        __syncthreads();
        if (!last[0]) return;
        __threadfence();
        if (tid == 0) cnt[1 + grp] = 0;
        float* const groups = all + (size_t)NPART * ROWS * 66;
        if (ng == 1) {
            merge(all, ncl, nullptr, true);
        } else {
            merge(all + (size_t)grp * GROUP * ROWS * 66, gn, groups + (size_t)grp * ROWS * 66, false);
            __threadfence();
            __syncthreads();
            if (tid == 0) last[0] = atomicAdd(cnt, 1) == ng - 1;
            __syncthreads();
            if (!last[0]) return;
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
        put(qrow(r) * 64 + d, fin[i]);
    }
}
`

// attnGateStore writes head_rms(fin) · sigmoid(gate) to out, one warp per
// query row (head).
const attnGateStore = `    for (int r = warp; r < ROWS; r += 4) {
        const float v0 = fin[r * 64 + lane], v1 = fin[r * 64 + lane + 32];
        const float inv = rsqrtf(eg_warp_sum(v0 * v0 + v1 * v1) * (1.0f / 64) + GATE_EPS);
        const size_t t = qrow(r) / HQ;
        const int h = qrow(r) % HQ;
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
	a, g, err := a.norm()
	if err != nil {
		return "", err
	}
	params, names := a.launchParams(), paramNames(a.params())
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(int batch, int max_keys, %s, cudaStream_t stream) {\n", a.Name, strings.Join(params, ", "))
	if a.DeviceSplits {
		// PlanSplits, with the device's resident blocks.
		fmt.Fprintf(&b, `    static const int slots = [] {
        int dev = 0, sms = 0, per = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, %s, 128, 0);
        return sms * per;
    }();
    const int per = batch > 0 ? (slots / (batch * %d) > 1 ? slots / (batch * %d) : 1) : 1;
    const int want = ((max_keys + per - 1) / per + 63) / 64 * 64, chunk = want > %d ? want : %d;
`, a.Name, a.Hkv/g.hpb, a.Hkv/g.hpb, a.Chunk, a.Chunk)
	} else {
		fmt.Fprintf(&b, "    constexpr int chunk = %d;\n", a.Chunk)
	}
	gridX := "(max_keys + chunk - 1) / chunk"
	if a.ClusterCombine {
		gridX = "((max_keys + chunk - 1) / chunk + 7) / 8 * 8"
	}
	smem := "0"
	if a.WGMMA {
		smem = "smem"
		sqBytes := 64 * 64
		if a.Swizzle {
			sqBytes = 64 * 128
		}
		dsmem := sqBytes + max(a.Stages*g.stageBytes(a.Depth), 4*64*HeadDim)
		fmt.Fprintf(&b, "    constexpr int smem = %d;\n", dsmem)
		fmt.Fprintf(&b, "    static const cudaError_t attr = cudaFuncSetAttribute(%s, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);\n    (void)attr;\n", a.Name)
	}
	fmt.Fprintf(&b, "    const dim3 grid(%s, %d, batch), block(128);\n", gridX, a.Hkv/g.hpb)
	fmt.Fprintf(&b, "    if (max_keys > 0 && batch > 0) EG_LAUNCH(%s, grid, block, %s, stream, %s);\n}\n", a.Name, smem, strings.Join(names, ", "))
	return b.String(), nil
}

// cudaWGMMA emits the native Hopper FP8 WGMMA form of a (normalized, a.WGMMA set).
//
// The 128-thread warpgroup (4 warps) maps the block's query rows (up to 64)
// across the 4 warps (16 rows per warp) and computes Q·Kᵀ over 64 keys per
// tile with two k=32 steps of wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3
// from shared-memory descriptors (unswizzled core-matrix layout or 128B XOR
// swizzle), followed by online softmax and P·V over all 64 keys.
func (a PagedAttention) cudaWGMMA(g attnGeom) string {
	var b strings.Builder
	if a.ClusterCombine {
		b.WriteString(ClusterPrelude)
		b.WriteString("\n")
	}
	if a.Swizzle {
		b.WriteString(wgmmaSwizzleHelpers)
	}
	wg := a.Name + "_wgmma"
	b.WriteString(wgmmaE4M3Func(wg, 64))
	fmt.Fprintf(&b, "// %s: paged FP8 GQA attention, %d/%d heads, %d query tokens, window %d, page %d, %d-key splits, depth %d, %d stages, wgmma.\n",
		a.Name, a.Hq, a.Hkv, a.QLen, a.Window, a.Page, a.Chunk, a.Depth, a.Stages)
	if a.ClusterCombine {
		fmt.Fprintf(&b, "__global__ void __cluster_dims__(%d, 1, 1) __launch_bounds__(128, 1) %s(\n    %s) {\n",
			ClusterSize, a.Name, strings.Join(a.params(), ",\n    "))
	} else {
		fmt.Fprintf(&b, "__global__ void __launch_bounds__(128) %s(\n    %s) {\n", a.Name, strings.Join(a.params(), ",\n    "))
	}
	fmt.Fprintf(&b, "    constexpr int HQ = %d, HKV = %d, G = %d, L = %d, ROWS = %d, MROWS = 64, KW = 1;\n",
		a.Hq, a.Hkv, g.G, a.QLen, g.rows)
	b.WriteString("    constexpr int HPB = 1, HKVB = HKV, ROWS1 = ROWS, D = 64, KP = 4;\n")
	fmt.Fprintf(&b, "    constexpr int WINDOW = %d, PAGE = %d, DEPTH = %d, STEP = 64 * DEPTH, STAGES = %d;\n",
		a.Window, a.Page, a.Depth, a.Stages)
	if a.DeviceSplits {
		b.WriteString("    const int CHUNK = chunk;\n")
	} else {
		fmt.Fprintf(&b, "    constexpr int CHUNK = %d;\n", a.Chunk)
	}
	sqRB := 64
	if a.Swizzle {
		sqRB = 128
	}
	fmt.Fprintf(&b, "    constexpr int PIECES = STEP * KP, NP = (PIECES + 127) / 128, GROUP = %d, RB = 64, SQ_RB = %d;\n", attnGroup, sqRB)
	b.WriteString("    constexpr int STAGE_BYTES = 2 * STEP * RB, RING_BYTES = STAGES * STAGE_BYTES > MROWS * 256 ? STAGES * STAGE_BYTES : MROWS * 256;\n")
	if a.TableAhead >= 2 {
		fmt.Fprintf(&b, "    constexpr int TABLE_AHEAD = %d;\n", a.TableAhead)
	}
	fmt.Fprintf(&b, "    const float scale = %s * k_scale[0] * 1.4426950408889634f; // log2 domain\n",
		floatLiteral(a.QScale*a.SMScale))
	if a.ClusterCombine {
		b.WriteString(attnWGMMAHeadCluster)
	} else {
		b.WriteString(attnWGMMAHead)
	}
	table := attnTableAhead1
	if a.TableAhead >= 2 {
		table = attnTableAheadN
	}
	qload := attnWGMMAQ
	issue := attnWGMMAIssue
	vload := attnWGMMAVLoad
	qkTile := fmt.Sprintf(`        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            %s(s, eg_wgmma_desc(sq + ks * 256, 128, 512), eg_wgmma_desc(kst + sub * 4096 + ks * 256, 128, 512));
        eg_wgmma_commit();
        eg_wgmma_wait<0>();
`, wg)
	if a.Swizzle {
		qload = attnWGMMAQSwizzle
		issue = attnWGMMAIssueSwizzle
		vload = attnWGMMAVLoadSwizzle
		qkTile = fmt.Sprintf(`        eg_wgmma_fence();
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            %s(s, eg_wgmma_desc_128b(sq + ks * 32, 1024), eg_wgmma_desc_128b(kst + sub * 8192 + ks * 32, 1024));
        eg_wgmma_commit();
        eg_wgmma_wait<0>();
`, wg)
	}
	nsteps := "    const int nsteps = (c1 - c0 + STEP - 1) / STEP;\n"
	tail := attnTail
	if a.ClusterCombine {
		nsteps = "    const int nsteps = c1 > c0 ? (c1 - c0 + STEP - 1) / STEP : 0;\n"
		tail = attnClusterTail
	}
	b.WriteString(qload)
	b.WriteString(strings.NewReplacer(
		"ISSUE_STAGE\n", issue,
		"NSTEPS_DECL\n", nsteps,
		"QK_TILE\n", qkTile,
		"V_LOAD\n", vload,
		"TABLE_PREFETCH\n", table[0],
		"TABLE_STEP\n", table[1],
		"LOOP_TAIL\n", tail,
	).Replace(attnWGMMALoop))
	if a.Out == BF16 {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = eg_f32_to_bf16(v); };\n")
	} else {
		b.WriteString("    auto put = [&](size_t i, float v) { out[i] = v; };\n")
	}
	if a.ClusterCombine {
		b.WriteString(attnClusterCombine)
	} else {
		b.WriteString(attnCombine)
	}
	if a.Gate {
		fmt.Fprintf(&b, "    constexpr int GATE_STRIDE = %d, GATE_OFFSET = %d;\n    constexpr float GATE_EPS = %s;\n",
			a.GateStride, a.GateOffset, floatLiteral(a.GateEps))
		b.WriteString(attnGateStore)
	} else {
		b.WriteString(attnStore)
	}
	if a.ClusterCombine {
		fmt.Fprintf(&b, "EG_REGISTER_CLUSTER(%s)\n", a.Name)
	}
	return b.String()
}

const attnWGMMAHead = `    const int split = blockIdx.x, kvb = blockIdx.y, kvh0 = kvb, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane / 4, c = lane % 4;
    const int seqlen = seqlens[b], p0 = seqlen - L;
    const int kbegin = WINDOW ? max(0, p0 - WINDOW + 1) : 0;
    const int nsplit = (seqlen - kbegin + CHUNK - 1) / CHUNK;
    if (split >= nsplit) return;
    const int c0 = kbegin + split * CHUNK, c1 = min(seqlen, c0 + CHUNK);
    const float NEG_INF = -__uint_as_float(0x7f800000u);
    auto trow = [](int pr) { return pr; };
    auto qrow = [&](int pr) { // [token, head] index of partial row pr
        return (size_t)(b * L + pr / G) * HQ + kvh0 * G + pr % G;
    };

    EG_DYNAMIC_SMEM(unsigned char, dsmem);
    unsigned char* const sq = dsmem;
    __shared__ float red_m[MROWS], red_l[MROWS];
    __shared__ int last;
    // The cp.async ring: STAGES stages of STEP K rows then STEP V rows of RB
    // bytes; after the main loop, red_o.
    unsigned char* const ring = dsmem + MROWS * SQ_RB;
    float* const red_o = reinterpret_cast<float*>(ring);
`

const attnWGMMAHeadCluster = `    const int split = blockIdx.x, kvb = blockIdx.y, kvh0 = kvb, b = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane / 4, c = lane % 4;
    const int seqlen = seqlens[b], p0 = seqlen - L;
    const int kbegin = WINDOW ? max(0, p0 - WINDOW + 1) : 0;
    const int nsplit = (seqlen - kbegin + CHUNK - 1) / CHUNK;
    const int rank = eg_cluster_rank(), cl = split / 8, ncl = (nsplit + 7) / 8;
    if (cl >= ncl) return;
    const int c0 = kbegin + split * CHUNK, c1 = min(seqlen, c0 + CHUNK);
    const float NEG_INF = -__uint_as_float(0x7f800000u);
    auto trow = [](int pr) { return pr; };
    auto qrow = [&](int pr) { // [token, head] index of partial row pr
        return (size_t)(b * L + pr / G) * HQ + kvh0 * G + pr % G;
    };

    EG_DYNAMIC_SMEM(unsigned char, dsmem);
    unsigned char* const sq = dsmem;
    EG_SHARED(float, red_m, [MROWS]);
    EG_SHARED(float, red_l, [MROWS]);
    EG_SHARED(int, last, [1]);
    // The cp.async ring: STAGES stages of STEP K rows then STEP V rows of RB
    // bytes; after the main loop, red_o.
    unsigned char* const ring = dsmem + MROWS * SQ_RB;
    float* const red_o = reinterpret_cast<float*>(ring);
`

const attnWGMMAQ = `    auto kcore = [](int r, int ch) { return ((r / 8) * 4 + ch) * 128 + (r % 8) * 16; };
    for (int i = tid; i < MROWS * KP; i += 128) {
        const int r = i / KP, ch = i % KP;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) w = eg_ldg16(q + qrow(r) * 64 + ch * 16);
        *reinterpret_cast<eg_u32x4*>(sq + kcore(r, ch)) = w;
    }
`

const attnWGMMAQSwizzle = `    auto kcore = [](int r, int ch) { return (int)eg_swizzle_128b((uint32_t)(r * 128 + ch * 16)); };
    auto vcore = [](int r, int ch) { return (int)eg_swizzle_128b((uint32_t)(r * 128 + 64 + ch * 16)); };
    for (int i = tid; i < MROWS * KP; i += 128) {
        const int r = i / KP, ch = i % KP;
        eg_u32x4 w = {0, 0, 0, 0};
        if (r < ROWS) w = eg_ldg16(q + qrow(r) * 64 + ch * 16);
        *reinterpret_cast<eg_u32x4*>(sq + kcore(r, ch)) = w;
    }
`

const attnWGMMAIssue = `            unsigned char* const dk = kd + kcore(piece / KP, piece % KP);
            unsigned char* const dv = kd + STEP * RB + (piece / KP) * RB + (piece % KP) * 16;
`

const attnWGMMAIssueSwizzle = `            unsigned char* const dk = kd + kcore(piece / KP, piece % KP);
            unsigned char* const dv = kd + vcore(piece / KP, piece % KP);
`

const attnWGMMAVLoad = `            const unsigned char* const vr = vst + (sub * 64 + ks * 16 + 2 * c) * RB + 8 * g;
            const eg_u32x2 va = *reinterpret_cast<const eg_u32x2*>(vr), vb = *reinterpret_cast<const eg_u32x2*>(vr + RB);
            const eg_u32x2 vc = *reinterpret_cast<const eg_u32x2*>(vr + 8 * RB), vd = *reinterpret_cast<const eg_u32x2*>(vr + 9 * RB);
`

const attnWGMMAVLoadSwizzle = `            const int k0 = sub * 64 + ks * 16 + 2 * c;
            const unsigned char* const vr0 = kst + vcore(k0, g >> 1) + (g & 1) * 8;
            const unsigned char* const vr1 = kst + vcore(k0 + 1, g >> 1) + (g & 1) * 8;
            const unsigned char* const vr2 = kst + vcore(k0 + 8, g >> 1) + (g & 1) * 8;
            const unsigned char* const vr3 = kst + vcore(k0 + 9, g >> 1) + (g & 1) * 8;
            const eg_u32x2 va = *reinterpret_cast<const eg_u32x2*>(vr0), vb = *reinterpret_cast<const eg_u32x2*>(vr1);
            const eg_u32x2 vc = *reinterpret_cast<const eg_u32x2*>(vr2), vd = *reinterpret_cast<const eg_u32x2*>(vr3);
`

const attnWGMMALoop = `    const int fk = tid * NP / KP;
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
ISSUE_STAGE
            if (row >= 0) {
                const size_t at = ((size_t)row * HKV + kvh0) * 64 + (piece % KP) * 16;
                eg_cp_async16(dk, k_cache + at);
                eg_cp_async16(dv, v_cache + at);
            } else {
                // Past the chunk's end P is 0, and 0·NaN is NaN: zero K and V.
                *reinterpret_cast<eg_u32x4*>(dk) = eg_u32x4{0, 0, 0, 0};
                *reinterpret_cast<eg_u32x4*>(dv) = eg_u32x4{0, 0, 0, 0};
            }
        }
    };
NSTEPS_DECL
TABLE_PREFETCH
    eg_fence_proxy_async();
    __syncthreads();

    // This lane's two rows (warp*16 + g and +8): positions and first keys.
    int pos[2], lo[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = warp * 16 + g + 8 * h;
        pos[h] = p0 + min(r / G, L - 1);
        lo[h] = WINDOW ? max(0, pos[h] - WINDOW + 1) : 0;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.0f, 0.0f}, o[8][4] = {};

    for (int it = 0; it < nsteps; ++it) {
        eg_cp_async_wait<STAGES - 2>();
        eg_fence_proxy_async();
        __syncthreads();
TABLE_STEP
        eg_cp_async_commit();
        const int kb = c0 + it * STEP;
        const unsigned char* const kst = ring + (it % STAGES) * STAGE_BYTES;
        const unsigned char* const vst = kst + STEP * RB;
#pragma unroll
        for (int sub = 0; sub < DEPTH; ++sub) {
        const int key0 = kb + sub * 64;
        if (key0 >= c1) break;
        float s[32] = {};
QK_TILE
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float mx = NEG_INF;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int key = key0 + nt * 8 + 2 * c + e;
                    float& x = s[nt * 4 + 2 * h + e];
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
            for (int nt = 0; nt < 8; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    float& x = s[nt * 4 + 2 * h + e];
                    x = exp2f(x - base);
                    l[h] += x;
                }
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const float* const sk = &s[ks * 8];
            const uint32_t pa[4] = {eg_pack_f16x2(sk[0], sk[1]), eg_pack_f16x2(sk[2], sk[3]),
                                    eg_pack_f16x2(sk[4], sk[5]), eg_pack_f16x2(sk[6], sk[7])};
V_LOAD
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const uint32_t sel = (t % 4) | (4 + t % 4) << 4;
                const uint32_t vf[2] = {eg_e4m3x2_to_f16x2(t < 4 ? eg_prmt(va.x, vb.x, sel) : eg_prmt(va.y, vb.y, sel)),
                                        eg_e4m3x2_to_f16x2(t < 4 ? eg_prmt(vc.x, vd.x, sel) : eg_prmt(vc.y, vd.y, sel))};
                eg_mma_f16(o[t], pa, vf, o[t]);
            }
        }
        }
    }
    eg_cp_async_wait<0>();
    __syncthreads();

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
        l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
        const int r = warp * 16 + g + 8 * h;
        if (c == 0) { red_m[r] = m[h]; red_l[r] = l[h]; }
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int e = 0; e < 2; ++e)
                red_o[r * 64 + 8 * (2 * c + e) + t] = o[t][2 * h + e];
    }
    __syncthreads();
    const float vs = v_scale[0];
LOOP_TAIL
`

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
