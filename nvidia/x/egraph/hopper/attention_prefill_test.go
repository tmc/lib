package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func TestPrefillAttentionEmulated(t *testing.T) {
	tests := []struct {
		name        string
		hq, hkv, d  int
		start, n    int
		page, pages int
	}{
		{"gqa4_d128", 8, 2, 128, 5, 70, 16, 8},
		{"mha_d128_one_tile", 2, 2, 128, 0, 33, 16, 4},
		{"gqa8_d64_two_tiles", 8, 1, 64, 17, 128, 16, 12},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := PrefillAttention{Name: "pfa_" + tt.name, NumHeads: tt.hq, KVHeads: tt.hkv, HeadDim: tt.d, PageSize: tt.page, MaxPages: tt.pages}
			src, err := a.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewPCG(1, uint64(tt.n)))
			rnd := func(n int) []float32 {
				v := make([]float32, n)
				for i := range v {
					v[i] = FromBF16Bits(BF16Bits(float32(rng.NormFloat64())))
				}
				return v
			}
			// A permuted page table puts position p in slot slot(p).
			table := make([]int32, tt.pages)
			for i, p := range rng.Perm(tt.pages) {
				table[i] = int32(p)
			}
			slot := func(p int) int { return int(table[p/tt.page])*tt.page + p%tt.page }
			slots := tt.pages * tt.page
			if tt.start+tt.n > slots {
				t.Fatalf("%d positions exceed %d slots", tt.start+tt.n, slots)
			}
			q := rnd(tt.n * tt.hq * tt.d)
			kc, vc := rnd(slots*tt.hkv*tt.d), rnd(slots*tt.hkv*tt.d)

			grid, block := a.Grid(tt.n)
			res := runEmu(t, src, fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), 0, %s, q, k_cache, v_cache, page_table, %d, %d, out)",
				grid[0], grid[1], grid[2], block[0], a.Name, tt.start, tt.n),
				emuBuf{"q", "uint16_t", toBF16Buf(q)},
				emuBuf{"k_cache", "uint16_t", toBF16Buf(kc)},
				emuBuf{"v_cache", "uint16_t", toBF16Buf(vc)},
				emuBuf{"page_table", "int32_t", i32Bytes(table)},
				emuBuf{"out", "uint16_t", make([]byte, 2*tt.n*tt.hq*tt.d)})
			got := bytesU16(res["out"])

			scale := 1 / math.Sqrt(float64(tt.d))
			g := tt.hq / tt.hkv
			var maxErr float64
			for i := range tt.n {
				pos := tt.start + i
				for h := range tt.hq {
					kvh := h / g
					qv := q[(i*tt.hq+h)*tt.d:][:tt.d]
					s := make([]float64, pos+1)
					mx := math.Inf(-1)
					for p := range s {
						kv := kc[(slot(p)*tt.hkv+kvh)*tt.d:][:tt.d]
						for d := range tt.d {
							s[p] += float64(qv[d]) * float64(kv[d])
						}
						s[p] *= scale
						mx = max(mx, s[p])
					}
					var sum float64
					for p := range s {
						s[p] = math.Exp(s[p] - mx)
						sum += s[p]
					}
					for d := range tt.d {
						var want float64
						for p := range s {
							want += s[p] / sum * float64(vc[(slot(p)*tt.hkv+kvh)*tt.d+d])
						}
						e := math.Abs(float64(FromBF16Bits(got[(i*tt.hq+h)*tt.d+d])) - want)
						maxErr = max(maxErr, e)
					}
				}
			}
			// P is rounded to BF16 before P·V, and out is BF16.
			if maxErr > 2e-2 {
				t.Errorf("max |out - reference| = %g, want <= 2e-2", maxErr)
			}
		})
	}
}

func TestPrefillAttentionErrors(t *testing.T) {
	ok := PrefillAttention{Name: "p", NumHeads: 8, KVHeads: 2, HeadDim: 128, PageSize: 16, MaxPages: 4}
	for _, tt := range []struct {
		name string
		mod  func(*PrefillAttention)
	}{
		{"no_name", func(a *PrefillAttention) { a.Name = "" }},
		{"heads", func(a *PrefillAttention) { a.NumHeads = 7 }},
		{"head_dim", func(a *PrefillAttention) { a.HeadDim = 96 }},
		{"page", func(a *PrefillAttention) { a.PageSize = 0 }},
	} {
		a := ok
		tt.mod(&a)
		if _, err := a.CUDA(); err == nil {
			t.Errorf("%s: no error", tt.name)
		}
	}
	if _, err := ok.CUDA(); err != nil {
		t.Error(err)
	}
}
