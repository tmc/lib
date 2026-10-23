package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
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
	ng := (a.Splits(maxKeys) + attnGroup - 1) / attnGroup
	clear(ws[:4*batch*a.Hkv*(1+ng)])
	return ws
}

func runPaged(t *testing.T, a PagedAttention, q []float32, c PagedCache, maxKeys, launches int) ([]float64, []byte) {
	t.Helper()
	src, err := a.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	batch := len(c.SeqLens)
	grid, block := a.Grid(batch, maxKeys)
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
		emuBuf{"ws", "unsigned char", attnWorkspace(a, batch, maxKeys)})
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), 0, %s, "+args+")",
		grid[0], grid[1], grid[2], block[0], a.Name, c.MaxPages)
	one := launch
	for i := 1; i < launches; i++ {
		launch += ";\n    " + one
	}
	res := runEmu(t, src, launch, bufs...)
	return decode(a.Out, res["out"]), res["ws"]
}

func TestPagedAttentionEmulated(t *testing.T) {
	tests := []struct {
		name    string
		a       PagedAttention
		seqlens []int32
		maxKeys int
		kvScale float32
	}{
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
		// 16-step chunks through a 3-stage ring, with ragged last steps.
		{"ring3_wrap", PagedAttention{Name: "attn_r3", Hq: 16, Hkv: 2, QLen: 1, Page: 16, Q: E4M3, Out: F32, QScale: 5.4730064863875, Chunk: 1024, Stages: 3},
			[]int32{1000, 129, 65}, 1000, 1},
		{"ring2_bf16q", PagedAttention{Name: "attn_r2", Hq: 16, Hkv: 2, QLen: 2, Page: 4, Q: BF16, Out: F32, Chunk: 512, Stages: 2},
			[]int32{700, 31}, 700, 1},
		{"ring8_fp8_q8", PagedAttention{Name: "attn_r8", Hq: 64, Hkv: 8, QLen: 8, Page: 16, Q: E4M3, Out: F32, Chunk: 320, QKFP8: true, Stages: 8},
			[]int32{511, 300}, 511, 1},
	}
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
			got, ws := runPaged(t, tt.a, q, c, tt.maxKeys, 2)
			bound := 5e-4 // F16 tiles, F32 accumulation
			if tt.a.Out == BF16 {
				bound = 4e-3
			}
			if e := relErr(got, want); e > bound || math.IsNaN(e) {
				t.Errorf("relative L2 error %.2e", e)
			} else {
				t.Logf("relative L2 error %.2e", e)
			}
			ng := (tt.a.Splits(tt.maxKeys) + attnGroup - 1) / attnGroup
			for i, b := range ws[:4*len(tt.seqlens)*tt.a.Hkv*(1+ng)] {
				if b != 0 {
					t.Fatalf("counter byte %d = %d after launch, want 0", i, b)
				}
			}
		})
	}
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
