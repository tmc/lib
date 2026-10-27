package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestPrefillSeqLens(t *testing.T) {
	tests := []struct {
		start, n, qtok int
		want           []int32
	}{
		{0, 1, 15, []int32{15}},
		{0, 15, 15, []int32{15}},
		{0, 16, 15, []int32{15, 30}},
		{128, 1, 15, []int32{143}},
		{2048 - 129, 129, 15, []int32{1934, 1949, 1964, 1979, 1994, 2009, 2024, 2039, 2054}},
	}
	for _, tt := range tests {
		got := prefillSeqLens(tt.start, tt.n, tt.qtok)
		if fmt.Sprint(got) != fmt.Sprint(tt.want) {
			t.Errorf("prefillSeqLens(%d, %d, %d) = %v, want %v", tt.start, tt.n, tt.qtok, got, tt.want)
		}
	}
}

// TestPrefillAttentionCausal runs the generated prefill attention kernel
// on the CPU over a chunk that starts mid-sequence and ends in a partial
// group, with the lengths prefillSeqLens gives, and checks every row
// against attention over positions 0 through its own. A length off by one
// either way moves every row's window and fails.
func TestPrefillAttentionCausal(t *testing.T) {
	const (
		hq, hkv, d = 4, 2, 16
		page       = 16
		start, n   = 21, 17 // rows 21..37: groups of 5 rows, the last partial
		qtok       = 5
	)
	slots := (start + n + qtok + page - 1) / page * page
	a := hopper.MTPPagedAttention{
		Name: "pf_attn", NumHeads: hq, KVHeads: hkv, HeadDim: d,
		PageSize: page, MaxPages: slots / page, QTokensPerSeq: qtok,
	}
	lens := prefillSeqLens(start, n, qtok)
	groups := len(lens)

	r := rand.New(rand.NewPCG(1, 2))
	bf := func(n int) []uint16 {
		v := make([]uint16, n)
		for i := range v {
			v[i] = hopper.BF16Bits(float32(r.NormFloat64()))
		}
		return v
	}
	// Cache rows past the chunk are zero, as the decoder leaves them.
	k, v := bf(slots*hkv*d), bf(slots*hkv*d)
	for i := (start + n) * hkv * d; i < len(k); i++ {
		k[i], v[i] = 0, 0
	}
	q := bf(groups * qtok * hq * d)
	pages := make([]int32, groups*a.MaxPages)
	for i := range pages {
		pages[i] = int32(i % a.MaxPages)
	}
	src, err := a.Source()
	if err != nil {
		t.Fatal(err)
	}
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source: src,
		Launch: fmt.Sprintf("%s_launch(%d, q, k_cache, v_cache, page_table, seqlens, nullptr, out, nullptr)", a.Name, groups),
		Buffers: []cudahostemu.Buffer{
			{Name: "q", CType: "uint16_t", Data: u16le(q)},
			{Name: "k_cache", CType: "uint16_t", Data: u16le(k)},
			{Name: "v_cache", CType: "uint16_t", Data: u16le(v)},
			{Name: "page_table", CType: "int32_t", Data: i32le(pages)},
			{Name: "seqlens", CType: "int32_t", Data: i32le(lens)},
			{Name: "out", CType: "uint16_t", Data: cudahostemu.Poison(2 * len(q))},
		},
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		t.Fatal(err)
	}
	out := res["out"]
	for row := range n {
		pos := start + row
		for h := range hq {
			qv := make([]float64, d)
			for e := range qv {
				qv[e] = float64(hopper.FromBF16Bits(q[(row*hq+h)*d+e]))
			}
			want := causalAttention(qv, k, v, hkv, h/(hq/hkv), pos+1, func(p int) int { return p })
			for e := range d {
				i := (row*hq+h)*d + e
				got := float64(hopper.FromBF16Bits(binary.LittleEndian.Uint16(out[2*i:])))
				if math.Abs(got-want[e]) > 0.02+0.01*math.Abs(want[e]) {
					t.Fatalf("row %d (position %d) head %d dim %d = %g, want %g", row, pos, h, e, got, want[e])
				}
			}
		}
	}
}

// causalAttention returns softmax(q·Kᵀ/√d)·V over cache positions 0..n-1
// of KV head kvh, in float64. slot maps a position to its cache row; k and
// v are BF16 [rows, hkv, d].
func causalAttention(q []float64, k, v []uint16, hkv, kvh, n int, slot func(int) int) []float64 {
	d := len(q)
	s := make([]float64, n)
	mx := math.Inf(-1)
	for j := range s {
		row := (slot(j)*hkv + kvh) * d
		for e := range d {
			s[j] += q[e] * float64(hopper.FromBF16Bits(k[row+e]))
		}
		s[j] /= math.Sqrt(float64(d))
		mx = max(mx, s[j])
	}
	var sum float64
	for j := range s {
		s[j] = math.Exp(s[j] - mx)
		sum += s[j]
	}
	out := make([]float64, d)
	for j := range s {
		row := (slot(j)*hkv + kvh) * d
		for e := range d {
			out[e] += s[j] * float64(hopper.FromBF16Bits(v[row+e])) / sum
		}
	}
	return out
}

// TestDecodeAttentionSplits runs the split-KV decode attention kernel and
// its combine on the CPU over a cache whose 64-row pages are shuffled, so
// split boundaries (multiples of 16 keys) fall inside pages, and checks
// every head against attention over all positions.
func TestDecodeAttentionSplits(t *testing.T) {
	const (
		hq, hkv, d = 4, 2, 16
		page       = 64
		maxPages   = 32
		splits     = 6
	)
	a := hopper.SM120PagedAttention{
		Name: "dec_attn", NumQHeads: hq, NumKVHeads: hkv, HeadDim: d,
		PageSize: page, MaxPages: maxPages, SeqLen: page * maxPages,
		KVFormat: hopper.WeightBF16, Splits: splits, MinSplitKeys: 16,
	}
	src, err := a.Source()
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(3, 4))
	pages := make([]int32, maxPages)
	for i, p := range r.Perm(maxPages) {
		pages[i] = int32(p)
	}
	slot := func(p int) int { return int(pages[p/page])*page + p%page }
	k := make([]uint16, maxPages*page*hkv*d)
	v := make([]uint16, len(k))
	for i := range k {
		k[i] = hopper.BF16Bits(float32(r.NormFloat64()))
		v[i] = hopper.BF16Bits(float32(r.NormFloat64()))
	}
	q := make([]float32, hq*d)
	for i := range q {
		q[i] = float32(r.NormFloat64())
	}
	for _, n := range []int{1, 15, 16, 17, 2000} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
				Source: src,
				Launch: fmt.Sprintf("%s_launch(1, q, k_pages, v_pages, nullptr, nullptr, page_table, ws, out, %d)", a.Name, n),
				Buffers: []cudahostemu.Buffer{
					{Name: "q", CType: "float", Data: f32le(q)},
					{Name: "k_pages", CType: "uint8_t", Data: u16le(k)},
					{Name: "v_pages", CType: "uint8_t", Data: u16le(v)},
					{Name: "page_table", CType: "int32_t", Data: i32le(pages)},
					{Name: "ws", CType: "float", Data: cudahostemu.Poison(4 * a.WorkspaceFloats(1))},
					{Name: "out", CType: "float", Data: cudahostemu.Poison(4 * len(q))},
				},
			})
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip("no host C++ compiler; emulated kernel not run")
			}
			if err != nil {
				t.Fatal(err)
			}
			out := res["out"]
			for h := range hq {
				qv := make([]float64, d)
				for e := range qv {
					qv[e] = float64(q[h*d+e])
				}
				want := causalAttention(qv, k, v, hkv, h/(hq/hkv), n, slot)
				for e := range d {
					got := float64(math.Float32frombits(binary.LittleEndian.Uint32(out[4*(h*d+e):])))
					if math.Abs(got-want[e]) > 1e-4+1e-4*math.Abs(want[e]) {
						t.Fatalf("head %d dim %d = %g, want %g", h, e, got, want[e])
					}
				}
			}
		})
	}
}

func u16le(v []uint16) []byte {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

func i32le(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

func f32le(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}
