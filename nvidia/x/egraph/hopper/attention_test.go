package hopper

import (
	"cmp"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// pagedCase builds a scattered paged cache for seqlens and random queries.
func pagedCase(r *rand.Rand, a PagedAttention, seqlens []int32, kvScale float32) ([]float32, PagedCache) {
	maxPages := 0
	for _, s := range seqlens {
		maxPages = max(maxPages, (int(s)+a.Page-1)/a.Page)
	}
	total := len(seqlens) * maxPages
	c := PagedCache{
		PageTable: make([]int32, len(seqlens)*maxPages),
		MaxPages:  maxPages,
		SeqLens:   seqlens,
		KScale:    kvScale,
		VScale:    kvScale,
		K:         make([]uint8, total*a.Page*a.Hkv*HeadDim),
		V:         make([]uint8, total*a.Page*a.Hkv*HeadDim),
	}
	perm := r.Perm(total)
	for i := range c.PageTable {
		c.PageTable[i] = int32(perm[i])
	}
	for b, s := range seqlens {
		for p := (int(s) + a.Page - 1) / a.Page; p < maxPages; p++ {
			c.PageTable[b*maxPages+p] = -1
		}
	}
	live := make([]bool, total*a.Page)
	for b, s := range seqlens {
		for t := range int(s) {
			live[int(c.PageTable[b*maxPages+t/a.Page])*a.Page+t%a.Page] = true
		}
	}
	// NaN codes in rows holding no token catch a kernel that reads them.
	for i := range c.K {
		c.K[i], c.V[i] = 0x7f, 0x7f
		if live[i/(a.Hkv*HeadDim)] {
			c.K[i] = E4M3Bits(float32(r.NormFloat64()) / kvScale)
			c.V[i] = E4M3Bits(float32(r.NormFloat64()) / kvScale)
		}
	}
	q := make([]float32, len(seqlens)*a.QLen*a.Hq*HeadDim)
	for i := range q {
		q[i] = a.Q.Round(float32(r.NormFloat64()))
	}
	return q, c
}

// attnWorkspace returns a's workspace as a launch first sees it: zero
// counters, then partials holding NaN.
func attnWorkspace(a PagedAttention, batch, maxKeys int) []byte {
	ws := poison(a.WorkspaceBytes(batch, maxKeys))
	clear(ws[:attnCounterBytes(a, batch, a.Splits(maxKeys))])
	return ws
}

// attnCounterBytes returns the bytes of a's counters for batch requests
// split splits ways.
func attnCounterBytes(a PagedAttention, batch, splits int) int {
	if a.ClusterCombine {
		splits /= ClusterSize
	}
	ng := (splits + attnGroup - 1) / attnGroup
	return 4 * batch * a.Hkv / max(a.KVHeadsPerBlock, 1) * (1 + ng)
}

// runPaged launches a through its launcher on an emulated device of slots
// resident blocks (one per SM). It returns the output, the workspace and
// the launch's grid.
func runPaged(t *testing.T, a PagedAttention, q []float32, c PagedCache, maxKeys, launches, slots int) ([]float64, []byte, [3]int) {
	t.Helper()
	src, err := a.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	l, err := a.Launcher()
	if err != nil {
		t.Fatal(err)
	}
	src += "\n" + l
	batch := len(c.SeqLens)
	args := "q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, out, ws"
	bufs := []emuBuf{
		{"q", a.Q.cType(), typed{a.Q, f64s(q)}.bytes()},
		{"k_cache", "uint8_t", c.K},
		{"v_cache", "uint8_t", c.V},
		{"page_table", "int32_t", i32Bytes(c.PageTable)},
		{"seqlens", "int32_t", i32Bytes(c.SeqLens)},
		{"k_scale", "float", f32Bytes([]float32{c.KScale})},
		{"v_scale", "float", f32Bytes([]float32{c.VScale})},
	}
	if a.Gate {
		bufs = append(bufs, emuBuf{"gate", "uint16_t", typed{BF16, f64s(c.Gate)}.bytes()})
		args = "q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, gate, out, ws"
	}
	bufs = append(bufs,
		emuBuf{"out", a.Out.cType(), poison(len(q) * a.Out.Bits() / 8)},
		emuBuf{"ws", "unsigned char", attnWorkspace(a, batch, maxKeys)},
		emuBuf{"grid", "uint32_t", poison(12)})
	launch := fmt.Sprintf("eg_emu_sms = %d;\n    eg_emu_blocks_per_sm = 1", slots)
	one := fmt.Sprintf("%s_launch(%d, %d, "+args+", nullptr)", a.Name, batch, maxKeys, c.MaxPages)
	launch += ";\n    " + one
	for i := 1; i < launches; i++ {
		launch += ";\n    " + one
	}
	launch += ";\n    grid[0] = eg_emu_last_grid.x;\n    grid[1] = eg_emu_last_grid.y;\n    grid[2] = eg_emu_last_grid.z"
	res := runEmu(t, src, launch, bufs...)
	g := i32s(res["grid"])
	return decode(a.Out, res["out"]), res["ws"], [3]int{int(g[0]), int(g[1]), int(g[2])}
}

type pagedTestCase struct {
	name    string
	a       PagedAttention
	seqlens []int32
	maxKeys int
	kvScale float32
}

func runPagedCases(t *testing.T, tests []pagedTestCase) {
	t.Helper()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(31, uint64(len(tt.name))))
			q, c := pagedCase(r, tt.a, tt.seqlens, tt.kvScale)
			if tt.a.Gate {
				c.Gate = make([]float32, len(tt.seqlens)*tt.a.QLen*tt.a.GateStride)
				for i := range c.Gate {
					c.Gate[i] = BF16.Round(float32(r.NormFloat64()))
				}
			}
			want, err := tt.a.Reference(q, c)
			if err != nil {
				t.Fatal(err)
			}
			// Resident blocks of the emulated device.
			slots := cmp.Or(map[string]int{"device_splits_many": 200, "hpb2_two_level": 200}[tt.name], 8)
			got, ws, grid := runPaged(t, tt.a, q, c, tt.maxKeys, 2, slots)
			bound := 5e-4 // F16 tiles, F32 accumulation
			if tt.a.Out == BF16 {
				bound = 4e-3
			}
			if e := relErr(got, want); e > bound || math.IsNaN(e) {
				t.Errorf("relative L2 error %.2e", e)
			} else {
				t.Logf("relative L2 error %.2e", e)
			}
			splits := tt.a.Splits(tt.maxKeys)
			if tt.a.DeviceSplits {
				_, splits = tt.a.PlanSplits(len(tt.seqlens), tt.maxKeys, slots)
			}
			want3, _ := tt.a.Grid(len(tt.seqlens), tt.maxKeys)
			want3[0] = splits
			if grid != want3 {
				t.Errorf("grid %v, want %v", grid, want3)
			}
			for i, b := range ws[:attnCounterBytes(tt.a, len(tt.seqlens), splits)] {
				if b != 0 {
					t.Fatalf("counter byte %d = %d after launch, want 0", i, b)
				}
			}
		})
	}
}

func TestPagedAttentionEmulated(t *testing.T) {
	runPagedCases(t, []pagedTestCase{
		{"full_decode_splits", PagedAttention{Name: "attn_full", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: F32, QScale: 5.4730064863875, Chunk: 64}, []int32{150, 37}, 150, 1},
		{"sliding_verify", PagedAttention{Name: "attn_verify", Hq: 16, Hkv: 2, QLen: 8, Window: 100, Page: 1, Q: BF16, Out: BF16, Chunk: 64}, []int32{130, 9}, 107, 1},
		{"verify4_scaled", PagedAttention{Name: "attn_v4", Hq: 16, Hkv: 2, QLen: 4, Window: 0, Page: 64, Q: E4M3, Out: F32, QScale: 5.4730064863875, Chunk: 128}, []int32{200, 64, 5}, 200, 1.0 / 64},
		{"decode2_window", PagedAttention{Name: "attn_w", Hq: 8, Hkv: 1, QLen: 2, Window: 50, Page: 4, Q: E4M3, Out: F32, Chunk: 64}, []int32{90, 3}, 51, 1},
		{"fp8_full_splits", PagedAttention{Name: "attn_fp8", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: F32, QScale: 5.4730064863875, Chunk: 64, QKFP8: true}, []int32{150, 37}, 150, 1},
		{"fp8_verify_gate", PagedAttention{Name: "attn_fp8_v", Hq: 16, Hkv: 2, QLen: 8, Window: 100, Page: 1, Q: E4M3, Out: BF16, QScale: 5.4730064863875, Chunk: 64, QKFP8: true,
			Gate: true, GateStride: 2304, GateOffset: 1280}, []int32{130, 9}, 107, 1.0 / 64},
		{"gate_splits", PagedAttention{Name: "attn_gate", Hq: 16, Hkv: 2, QLen: 2, Window: 0, Page: 8, Q: E4M3, Out: BF16, QScale: 5.4730064863875, Chunk: 64,
			Gate: true, GateStride: 2304, GateOffset: 1280}, []int32{150, 20}, 150, 1},
		{"gate_dp", PagedAttention{Name: "attn_gate_dp", Hq: 64, Hkv: 8, QLen: 8, Window: 30, Page: 1, Q: E4M3, Out: F32, Chunk: 64,
			Gate: true, GateStride: 9216, GateOffset: 5120}, []int32{60}, 37, 1},
		// 41, 17, 16 and 1 chunks: three groups with a short last one, two
		// groups with a last group of one chunk, one full group, no split.
		{"two_level", PagedAttention{Name: "attn_2l", Hq: 16, Hkv: 2, QLen: 1, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875, Chunk: 64},
			[]int32{2600, 1025, 1024, 37}, 2600, 1},
		// 128-key steps over 192-key chunks: every other step is half empty.
		{"fp8_depth2_two_level", PagedAttention{Name: "attn_fp8_d2", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			Chunk: 192, Depth: 2, QKFP8: true, Gate: true, GateStride: 2304, GateOffset: 1280}, []int32{4000, 200}, 4000, 1.0 / 64},
		{"depth2_dp", PagedAttention{Name: "attn_d2_dp", Hq: 64, Hkv: 8, QLen: 8, Page: 1, Q: E4M3, Out: F32, Chunk: 128, Depth: 2},
			[]int32{300, 70}, 300, 1},
		// Device-sized splits: 8 slots give 3 requests x 2 KV heads one
		// split each; 200 give 33, rounded to 32 splits of 128 keys (2
		// groups).
		{"device_splits_few", PagedAttention{Name: "attn_dev", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			QKFP8: true, Gate: true, GateStride: 2304, GateOffset: 1280, DeviceSplits: true, Chunk: 64}, []int32{4000, 641, 1}, 4000, 1},
		{"device_splits_many", PagedAttention{Name: "attn_dev_many", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			QKFP8: true, Gate: true, GateStride: 2304, GateOffset: 1280, DeviceSplits: true, Chunk: 64}, []int32{4000, 641, 1}, 4000, 1},
		// 481 chunks: 31 groups, so the final merge folds two batches of
		// about equal weight. The
		// large QScale spreads logits over hundreds of binades, so a merge
		// whose running max is wrong overflows instead of passing.
		{"wide_merge", PagedAttention{Name: "attn_wide", Hq: 8, Hkv: 1, QLen: 1, Page: 1, Q: E4M3, Out: F32, Chunk: 64, QKFP8: true, QScale: 200},
			[]int32{30770}, 30770, 1},
		// 16-step chunks through a 3-stage ring, with ragged last steps.
		{"ring3_wrap", PagedAttention{Name: "attn_r3", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: F32, QScale: 5.4730064863875, Chunk: 1024, Stages: 3},
			[]int32{1000, 129, 65}, 1000, 1},
		{"ring2_bf16q", PagedAttention{Name: "attn_r2", Hq: 16, Hkv: 2, QLen: 2, Page: 4, Q: BF16, Out: F32, Chunk: 512, Stages: 2},
			[]int32{700, 31}, 700, 1},
		{"ring8_fp8_q8", PagedAttention{Name: "attn_r8", Hq: 64, Hkv: 8, QLen: 8, Page: 16, Q: E4M3, Out: F32, Chunk: 320, QKFP8: true, Stages: 8},
			[]int32{511, 300}, 511, 1},
		// Two KV heads per block: 4 query rows per head (tile rows 0-3 and
		// 8-11), 8 (the full tile, two-level merge), and 8 from 2 tokens
		// with F16 Q·Kᵀ in a window.
		{"hpb2_fp8_gate", PagedAttention{Name: "attn_h2", Hq: 16, Hkv: 4, QLen: 1, Page: 16, Q: E4M3, Out: BF16, QScale: 5.4730064863875, Chunk: 64,
			QKFP8: true, Gate: true, GateStride: 3072, GateOffset: 2048, KVHeadsPerBlock: 2}, []int32{300, 37, 1}, 300, 1},
		{"hpb2_two_level", PagedAttention{Name: "attn_h2_2l", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: F32, QScale: 5.4730064863875,
			QKFP8: true, DeviceSplits: true, Chunk: 64, KVHeadsPerBlock: 2}, []int32{4000, 641, 1}, 4000, 1},
		{"hpb2_f16_window", PagedAttention{Name: "attn_h2_w", Hq: 8, Hkv: 2, QLen: 2, Window: 50, Page: 1, Q: BF16, Out: BF16, Chunk: 64, KVHeadsPerBlock: 2},
			[]int32{130, 9}, 51, 1},
		// Multi-stage page-table register prefetch (TableAhead 3 across a
		// 4-stage cp.async ring): page sizes 1 and 64, windowed and full,
		// ragged lengths, and split-KV.
		{"ahead3_p1_window", PagedAttention{Name: "attn_ta3_p1", Hq: 4, Hkv: 1, QLen: 1, Window: 300, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875,
			Chunk: 384, Stages: 4, TableAhead: 3, QKFP8: true}, []int32{400, 129, 7}, 300, 1},
		{"ahead3_p64_full_splits", PagedAttention{Name: "attn_ta3_p64", Hq: 4, Hkv: 1, QLen: 2, Window: 0, Page: 64, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			Chunk: 256, Stages: 4, TableAhead: 3, QKFP8: true, Gate: true, GateStride: 768, GateOffset: 512}, []int32{520, 65, 5}, 520, 1.0 / 64},
		// 128B XOR-swizzled shared-memory K/V staging: 5 stages fit in 48 KiB
		// with unpadded 64B rows; covers QKFP8 true/false and KVHeadsPerBlock 1/2.
		{"swizzle_fp8_s5", PagedAttention{Name: "attn_sw_fp8", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: F32, QScale: 5.4730064863875,
			Chunk: 384, Stages: 5, TableAhead: 3, QKFP8: true, Swizzle: true}, []int32{450, 130, 19}, 450, 1},
		{"swizzle_f16_s3", PagedAttention{Name: "attn_sw_f16", Hq: 16, Hkv: 2, QLen: 2, Window: 150, Page: 4, Q: BF16, Out: F32,
			Chunk: 192, Stages: 3, Swizzle: true}, []int32{220, 75, 5}, 151, 1.0 / 64},
		{"swizzle_hpb2_fp8", PagedAttention{Name: "attn_sw_h2_fp8", Hq: 16, Hkv: 4, QLen: 1, Page: 16, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			Chunk: 128, Stages: 2, KVHeadsPerBlock: 2, QKFP8: true, Swizzle: true, Gate: true, GateStride: 3072, GateOffset: 2048}, []int32{300, 65, 3}, 300, 1},
		{"swizzle_hpb2_f16", PagedAttention{Name: "attn_sw_h2_f16", Hq: 8, Hkv: 2, QLen: 2, Window: 90, Page: 1, Q: BF16, Out: BF16,
			Chunk: 128, Stages: 2, KVHeadsPerBlock: 2, Swizzle: true}, []int32{140, 11}, 91, 1},
		// 8-CTA cluster DSMEM split-KV combine: 8 splits (single cluster, no
		// global partials), 16 splits (two clusters with fused Gate), and
		// ragged multi-group cluster splits.
		{"cluster_8_splits", PagedAttention{Name: "attn_cl8", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: F32, QScale: 5.4730064863875,
			Chunk: 64, QKFP8: true, Swizzle: true, ClusterCombine: true}, []int32{512, 300, 37}, 512, 1},
		{"cluster_16_splits_gate", PagedAttention{Name: "attn_cl16_gate", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			Chunk: 64, QKFP8: true, Swizzle: true, ClusterCombine: true, Gate: true, GateStride: 2304, GateOffset: 1280}, []int32{1024, 577, 130}, 1024, 1.0 / 64},
		{"cluster_ragged_two_level", PagedAttention{Name: "attn_cl_2l", Hq: 8, Hkv: 2, QLen: 2, Page: 8, Q: BF16, Out: F32,
			Chunk: 64, Stages: 2, KVHeadsPerBlock: 2, Swizzle: true, ClusterCombine: true}, []int32{8500, 700, 35}, 8500, 1},
	})
}

func TestPagedAttentionWGMMA(t *testing.T) {
	// Native Hopper FP8 WGMMA m64n64k32 Q·Kᵀ across QLen 1 and 8 (Hq 16,
	// Hkv 2 -> 64 rows), Swizzle true/false, Page 1/64, Window/full, Gate.
	runPagedCases(t, []pagedTestCase{
		{"wgmma_q1_unswizzled_p1", PagedAttention{Name: "attn_wg_q1", Hq: 16, Hkv: 2, QLen: 1, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875,
			Chunk: 128, QKFP8: true, WGMMA: true}, []int32{300, 75, 5}, 300, 1},
		{"wgmma_q1_swizzle_p64_gate", PagedAttention{Name: "attn_wg_q1_sw", Hq: 16, Hkv: 2, QLen: 1, Window: 180, Page: 64, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			Chunk: 128, QKFP8: true, WGMMA: true, Swizzle: true, Gate: true, GateStride: 2304, GateOffset: 1280}, []int32{250, 90, 12}, 180, 1.0 / 64},
		{"wgmma_q8_unswizzled_p64_window", PagedAttention{Name: "attn_wg_q8", Hq: 16, Hkv: 2, QLen: 8, Window: 120, Page: 64, Q: E4M3, Out: F32, QScale: 5.4730064863875,
			Chunk: 64, QKFP8: true, WGMMA: true}, []int32{180, 65, 9}, 127, 1},
		{"wgmma_q8_swizzle_p1_full_gate", PagedAttention{Name: "attn_wg_q8_sw", Hq: 16, Hkv: 2, QLen: 8, Page: 1, Q: E4M3, Out: BF16, QScale: 5.4730064863875,
			Chunk: 128, Stages: 4, TableAhead: 3, QKFP8: true, WGMMA: true, Swizzle: true, Gate: true, GateStride: 2304, GateOffset: 1280}, []int32{290, 129, 11}, 290, 1.0 / 64},
	})
}

func relErr(got, want []float64) float64 {
	var num, den float64
	for i := range want {
		num += (got[i] - want[i]) * (got[i] - want[i])
		den += want[i] * want[i]
	}
	return math.Sqrt(num / den)
}

// TestPagedAttentionFormula checks Reference against reference.py's
// formula for one query: softmax over the window, FP8 values times scales.
func TestPagedAttentionFormula(t *testing.T) {
	a := PagedAttention{Name: "attn", Hq: 4, Hkv: 2, QLen: 2, Window: 3, Page: 2, Q: BF16, Out: F32, QScale: 2}
	r := rand.New(rand.NewPCG(33, 34))
	q, c := pagedCase(r, a, []int32{7}, 0.5)
	got, err := a.Reference(q, c)
	if err != nil {
		t.Fatal(err)
	}
	// Token j=0 sits at position 5 and attends 3, 4, 5; head 3 reads KV head 1.
	const j, h, kvh = 0, 3, 1
	var logits [3]float64
	var mx = math.Inf(-1)
	for i, pos := range []int{3, 4, 5} {
		row := int(c.PageTable[pos/2])*2 + pos%2
		for d := range 64 {
			logits[i] += float64(q[(j*4+h)*64+d]) * 2 * float64(FromE4M3Bits(c.K[(row*2+kvh)*64+d])) * 0.5 / 8
		}
		mx = math.Max(mx, logits[i])
	}
	var sum float64
	for i := range logits {
		logits[i] = math.Exp(logits[i] - mx)
		sum += logits[i]
	}
	for d := range 64 {
		var want float64
		for i, pos := range []int{3, 4, 5} {
			row := int(c.PageTable[pos/2])*2 + pos%2
			want += logits[i] / sum * float64(FromE4M3Bits(c.V[(row*2+kvh)*64+d])) * 0.5
		}
		if g := got[(j*4+h)*64+d]; math.Abs(g-want) > 1e-12 {
			t.Fatalf("out[%d] = %g, want %g", d, g, want)
		}
	}
}

func TestPagedAttentionErrors(t *testing.T) {
	ok := PagedAttention{Name: "a", Hq: 16, Hkv: 2, QLen: 1, Page: 1, Q: E4M3, Out: F32}
	tests := []struct {
		name string
		edit func(*PagedAttention)
	}{
		{"name", func(a *PagedAttention) { a.Name = "eg_x" }},
		{"heads", func(a *PagedAttention) { a.Hkv = 3 }},
		{"rows", func(a *PagedAttention) { a.Hq, a.Hkv, a.QLen = 128, 16, 9 }},
		{"rows64", func(a *PagedAttention) { a.Hq, a.Hkv, a.QLen = 128, 1, 1 }},
		{"page", func(a *PagedAttention) { a.Page = 0 }},
		{"q", func(a *PagedAttention) { a.Q = F32 }},
		{"out", func(a *PagedAttention) { a.Out = E4M3 }},
		{"chunk", func(a *PagedAttention) { a.Chunk = 100 }},
		{"depth", func(a *PagedAttention) { a.Depth = 3 }},
		{"stages1", func(a *PagedAttention) { a.Stages = 1 }},
		{"stages9", func(a *PagedAttention) { a.Stages = 9 }},
		{"shared", func(a *PagedAttention) { a.Depth, a.Stages = 2, 3 }},
		{"fp8_q", func(a *PagedAttention) { a.Q, a.QKFP8 = BF16, true }},
		{"hpb3", func(a *PagedAttention) { a.KVHeadsPerBlock = 3 }},
		{"hpb2_odd", func(a *PagedAttention) { a.Hq, a.Hkv, a.KVHeadsPerBlock = 3, 3, 2 }},
		{"hpb2_rows", func(a *PagedAttention) { a.QLen, a.KVHeadsPerBlock = 2, 2 }},
		{"ahead_neg", func(a *PagedAttention) { a.TableAhead = -1 }},
		{"ahead_over", func(a *PagedAttention) { a.Stages, a.TableAhead = 3, 4 }},
		{"wgmma_bf16_q", func(a *PagedAttention) { a.Q, a.WGMMA = BF16, true }},
		{"wgmma_hpb2", func(a *PagedAttention) { a.KVHeadsPerBlock, a.WGMMA = 2, true }},
	}
	if _, err := ok.CUDA(); err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		a := ok
		tt.edit(&a)
		if _, err := a.CUDA(); err == nil {
			t.Errorf("%s: no error", tt.name)
		}
	}
}

func TestPagedAttentionGolden(t *testing.T) {
	a := PagedAttention{Name: "moe8_attn_sliding", Hq: 16, Hkv: 2, QLen: 1, Window: 2048, Page: 1, QKFP8: true, Gate: true, GateStride: 2304, GateOffset: 1280,
		Q: E4M3, Out: F32, QScale: 5.4730064863875, Chunk: 256}
	src, err := a.Source()
	if err != nil {
		t.Fatal(err)
	}
	checkGolden(t, "moe8_attn_sliding.cu", src)
}

func ExamplePagedAttention() {
	// Moe8 tp8 sliding-window decode: 16 query heads over 2 FP8 KV heads.
	a := PagedAttention{Name: "attn", Hq: 16, Hkv: 2, QLen: 1, Window: 2048, Page: 1,
		Q: E4M3, Out: F32, QScale: 5.4730064863875}
	maxKeys := 2048
	grid, block := a.Grid(32, maxKeys)
	fmt.Println(grid, block, a.WorkspaceBytes(32, maxKeys))
	// Output: [8 2 32] [128 1 1] 1217024
}

// TestPagedAttentionVariantEmulated runs a DeviceSplits variant through its
// extern "C" entry, the way a tuner or a ctypes caller does.
func TestPagedAttentionVariantEmulated(t *testing.T) {
	a := PagedAttention{Name: "attn_v", Hq: 16, Hkv: 2, QLen: 1, Page: 4, Q: E4M3, Out: F32, Chunk: 64, QKFP8: true, DeviceSplits: true}
	v, err := a.Variant()
	if err != nil {
		t.Fatal(err)
	}
	seqlens := []int32{900, 70}
	q, c := pagedCase(rand.New(rand.NewPCG(5, 6)), a, seqlens, 1)
	want, err := a.Reference(q, c)
	if err != nil {
		t.Fatal(err)
	}
	const maxKeys = 900
	res := runEmu(t, v.Source, fmt.Sprintf("eg_emu_sms = 16;\n    %s_c(%d, %d, q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, out, ws, nullptr)",
		v.Entry, len(seqlens), maxKeys, c.MaxPages),
		emuBuf{"q", "uint8_t", typed{E4M3, f64s(q)}.bytes()},
		emuBuf{"k_cache", "uint8_t", c.K},
		emuBuf{"v_cache", "uint8_t", c.V},
		emuBuf{"page_table", "int32_t", i32Bytes(c.PageTable)},
		emuBuf{"seqlens", "int32_t", i32Bytes(c.SeqLens)},
		emuBuf{"k_scale", "float", f32Bytes([]float32{c.KScale})},
		emuBuf{"v_scale", "float", f32Bytes([]float32{c.VScale})},
		emuBuf{"out", "float", poison(4 * len(q))},
		emuBuf{"ws", "unsigned char", attnWorkspace(a, len(seqlens), maxKeys)})
	if e := relErr(decode(F32, res["out"]), want); e > 5e-4 || math.IsNaN(e) {
		t.Errorf("relative L2 error %.2e", e)
	}
}

// TestPagedAttentionTableAheadMutations verifies that corrupting the
// multi-stage page-table register ring (off-by-one slot or dropped tail
// prefetch) fails the emulator.
func TestPagedAttentionTableAheadMutations(t *testing.T) {
	// With 265 keys in a 512-key chunk (5 steps of 64 keys: steps 0..3 full
	// and step 4 partially empty, keys 256..264) and Stages=3, TableAhead=2,
	// the prologue issues steps 0..1 and prefetches steps 2..3 into pg[0..1],
	// step 0 refills pg[0] with step 4, and step 1 refills pg[1] with -1.
	// Reading pg[(it + 1) % TABLE_AHEAD] in issue() or dropping the in-loop
	// prefetch corrupts the prefetched steps.
	a := PagedAttention{Name: "attn_ta_mut", Hq: 4, Hkv: 1, QLen: 1, Window: 265, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875,
		Chunk: 512, Stages: 3, TableAhead: 2, QKFP8: true}
	src, err := a.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	l, err := a.Launcher()
	if err != nil {
		t.Fatal(err)
	}
	full := src + "\n" + l
	seqlens := []int32{560}
	q, c := pagedCase(rand.New(rand.NewPCG(71, 72)), a, seqlens, 1)
	want, err := a.Reference(q, c)
	if err != nil {
		t.Fatal(err)
	}
	const maxKeys = 265
	launch := fmt.Sprintf("eg_emu_sms = 8;\n    %s_launch(%d, %d, q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, out, ws, nullptr)",
		a.Name, len(seqlens), maxKeys, c.MaxPages)
	mutants := []struct {
		name string
		old  string
		repl string
	}{
		{
			name: "off_by_one_slot",
			old:  "issue((it + STAGES - 1) % STAGES, pg[slot]);",
			repl: "issue((it + STAGES - 1) % STAGES, pg[(slot + 1) % TABLE_AHEAD]);",
		},
		{
			name: "dropped_tail_prefetch",
			old:  "pg[slot] = it + STAGES - 1 + TABLE_AHEAD < nsteps ? rowof(c0 + (it + STAGES - 1 + TABLE_AHEAD) * STEP) : -1;",
			repl: "pg[slot] = -1;",
		},
	}
	for _, mut := range mutants {
		if !strings.Contains(full, mut.old) {
			t.Fatalf("%s: target not in source", mut.name)
		}
		msrc := strings.Replace(full, mut.old, mut.repl, 1)
		res := runEmu(t, msrc, launch,
			emuBuf{"q", "uint8_t", typed{E4M3, f64s(q)}.bytes()},
			emuBuf{"k_cache", "uint8_t", c.K},
			emuBuf{"v_cache", "uint8_t", c.V},
			emuBuf{"page_table", "int32_t", i32Bytes(c.PageTable)},
			emuBuf{"seqlens", "int32_t", i32Bytes(c.SeqLens)},
			emuBuf{"k_scale", "float", f32Bytes([]float32{c.KScale})},
			emuBuf{"v_scale", "float", f32Bytes([]float32{c.VScale})},
			emuBuf{"out", "float", poison(4 * len(q))},
			emuBuf{"ws", "unsigned char", attnWorkspace(a, len(seqlens), maxKeys)})
		if e := relErr(decode(F32, res["out"]), want); !(e > 5e-4 || math.IsNaN(e)) {
			t.Errorf("mutant %s survived (relErr %.2e)", mut.name, e)
		}
	}
}

// TestPagedAttentionSwizzleAndClusterMutations verifies that removing the
// read-side XOR swizzle (when stages are written swizzled) or corrupting the
// cluster DSMEM peer rank offset in ClusterCombine fails the emulator.
func TestPagedAttentionSwizzleAndClusterMutations(t *testing.T) {
	cases := []struct {
		name    string
		a       PagedAttention
		seqlens []int32
		maxKeys int
		old     string
		repl    string
	}{
		{
			name: "unswizzled_k_fp8_read",
			a: PagedAttention{Name: "attn_mut_kfp8", Hq: 8, Hkv: 1, QLen: 1, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875,
				Chunk: 64, QKFP8: true, Swizzle: true},
			seqlens: []int32{64},
			maxKeys: 64,
			old:     "((2 * ks + (lane / 8) % 2) ^ (lane & (KP - 1))) * 8",
			repl:    "ks * 16 + 8 * ((lane / 8) % 2)",
		},
		{
			name: "unswizzled_k_f16_read",
			a: PagedAttention{Name: "attn_mut_kf16", Hq: 8, Hkv: 1, QLen: 1, Page: 1, Q: BF16, Out: F32,
				Chunk: 64, Swizzle: true},
			seqlens: []int32{64},
			maxKeys: 64,
			old:     "(ks ^ (g & (KP - 1))) * 16",
			repl:    "ks * 16",
		},
		{
			name: "unswizzled_v_read",
			a: PagedAttention{Name: "attn_mut_v", Hq: 8, Hkv: 1, QLen: 1, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875,
				Chunk: 64, QKFP8: true, Swizzle: true},
			seqlens: []int32{64},
			maxKeys: 64,
			old:     "((4 * hb + (g >> 1)) ^ ((2 * c) & (KP - 1))) * 16",
			repl:    "(4 * hb + (g >> 1)) * 16",
		},
		{
			name: "cluster_rank_offset_single",
			a: PagedAttention{Name: "attn_mut_cl1", Hq: 8, Hkv: 1, QLen: 1, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875,
				Chunk: 64, QKFP8: true, Swizzle: true, ClusterCombine: true},
			seqlens: []int32{512},
			maxKeys: 512,
			old:     "eg_map_shared_rank(red_o, rk)",
			repl:    "eg_map_shared_rank(red_o, (rk + 1) & 7)",
		},
		{
			name: "cluster_rank_offset_multi",
			a: PagedAttention{Name: "attn_mut_cl2", Hq: 8, Hkv: 1, QLen: 1, Page: 1, Q: E4M3, Out: F32, QScale: 5.4730064863875,
				Chunk: 64, QKFP8: true, Swizzle: true, ClusterCombine: true},
			seqlens: []int32{1024},
			maxKeys: 1024,
			old:     "eg_map_shared_rank(red_m, rk)",
			repl:    "eg_map_shared_rank(red_m, 0)",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src, err := tc.a.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			l, err := tc.a.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			full := src + "\n" + l
			if !strings.Contains(full, tc.old) {
				t.Fatalf("target %q not in source", tc.old)
			}
			msrc := strings.Replace(full, tc.old, tc.repl, 1)
			q, c := pagedCase(rand.New(rand.NewPCG(83, 89)), tc.a, tc.seqlens, 1)
			want, err := tc.a.Reference(q, c)
			if err != nil {
				t.Fatal(err)
			}
			launch := fmt.Sprintf("eg_emu_sms = 8;\n    %s_launch(%d, %d, q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, out, ws, nullptr)",
				tc.a.Name, len(tc.seqlens), tc.maxKeys, c.MaxPages)
			res := runEmu(t, msrc, launch,
				emuBuf{"q", tc.a.Q.cType(), typed{tc.a.Q, f64s(q)}.bytes()},
				emuBuf{"k_cache", "uint8_t", c.K},
				emuBuf{"v_cache", "uint8_t", c.V},
				emuBuf{"page_table", "int32_t", i32Bytes(c.PageTable)},
				emuBuf{"seqlens", "int32_t", i32Bytes(c.SeqLens)},
				emuBuf{"k_scale", "float", f32Bytes([]float32{c.KScale})},
				emuBuf{"v_scale", "float", f32Bytes([]float32{c.VScale})},
				emuBuf{"out", "float", poison(4 * len(q))},
				emuBuf{"ws", "unsigned char", attnWorkspace(tc.a, len(tc.seqlens), tc.maxKeys)})
			if e := relErr(decode(F32, res["out"]), want); !(e > 5e-4 || math.IsNaN(e)) {
				t.Errorf("mutant %s survived (relErr %.2e)", tc.name, e)
			}
		})
	}
}

// TestPagedAttentionWGMMAMutations verifies that dropping eg_wgmma_wait<0>() in
// PagedAttention{WGMMA: true} fails with NaN on the host emulator (when
// finalized at kernel exit) or errors on exit if uncommitted, and survives
// under EG_EMU_SYNC_WGMMA=1.
func TestPagedAttentionWGMMAMutations(t *testing.T) {
	for _, swizzle := range []bool{false, true} {
		name := "unswizzled"
		if swizzle {
			name = "swizzled"
		}
		t.Run(name, func(t *testing.T) {
			a := PagedAttention{
				Name: "attn_wg_mut", Hq: 16, Hkv: 2, QLen: 8, Page: 1,
				Q: E4M3, Out: F32, QScale: 5.4730064863875,
				Chunk: 64, QKFP8: true, WGMMA: true, Swizzle: swizzle,
			}
			src, err := a.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			l, err := a.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			full := src + "\n" + l
			seqlens := []int32{128, 64}
			const maxKeys = 128
			q, c := pagedCase(rand.New(rand.NewPCG(91, 97)), a, seqlens, 1)
			want, err := a.Reference(q, c)
			if err != nil {
				t.Fatal(err)
			}
			launch := fmt.Sprintf("eg_emu_sms = 8;\n    %s_launch(%d, %d, q, k_cache, v_cache, page_table, %d, seqlens, k_scale, v_scale, out, ws, nullptr)",
				a.Name, len(seqlens), maxKeys, c.MaxPages)
			runWithFlags := func(code string, flags ...string) []float64 {
				t.Setenv("EG_EMU_FLAGS", strings.Join(flags, " "))
				res := runEmu(t, code, launch,
					emuBuf{"q", "uint8_t", typed{E4M3, f64s(q)}.bytes()},
					emuBuf{"k_cache", "uint8_t", c.K},
					emuBuf{"v_cache", "uint8_t", c.V},
					emuBuf{"page_table", "int32_t", i32Bytes(c.PageTable)},
					emuBuf{"seqlens", "int32_t", i32Bytes(c.SeqLens)},
					emuBuf{"k_scale", "float", f32Bytes([]float32{c.KScale})},
					emuBuf{"v_scale", "float", f32Bytes([]float32{c.VScale})},
					emuBuf{"out", "float", poison(4 * len(q))},
					emuBuf{"ws", "unsigned char", attnWorkspace(a, len(seqlens), maxKeys)})
				return decode(F32, res["out"])
			}
			// 1. Unmutated kernel passes.
			if e := relErr(runWithFlags(full), want); e > 5e-4 || math.IsNaN(e) {
				t.Fatalf("unmutated relErr = %.2e", e)
			}

			// 2. Dropping eg_wgmma_wait<0>() before reading s for softmax (while
			// draining pending ops at loop exit so the thread does not abort)
			// reads the NaN-poisoned accumulator under async WGMMA, and survives
			// under -DEG_EMU_SYNC_WGMMA=1.
			const target = "        eg_wgmma_wait<0>();\n"
			if !strings.Contains(full, target) {
				t.Fatalf("target %q not in source", target)
			}
			msrc := strings.Replace(full, target, "", 1)
			msrc = strings.Replace(msrc, "    eg_cp_async_wait<0>();\n", "    eg_wgmma_wait<0>();\n    eg_cp_async_wait<0>();\n", 1)
			gotAsync := runWithFlags(msrc)
			if e := relErr(gotAsync, want); !math.IsNaN(e) {
				t.Errorf("dropped wgmma_wait mutant: relErr = %.2e, want NaN", e)
			}
			gotSync := runWithFlags(msrc, "-DEG_EMU_SYNC_WGMMA=1")
			if e := relErr(gotSync, want); e > 5e-4 || math.IsNaN(e) {
				t.Errorf("dropped wgmma_wait mutant under EG_EMU_SYNC_WGMMA=1: relErr = %.2e, want <= 5e-4", e)
			}
		})
	}
}
