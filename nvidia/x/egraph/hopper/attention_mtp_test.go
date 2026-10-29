package hopper

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestCausalAndTreeDraftMask(t *testing.T) {
	causal := CausalDraftMask(4)
	wantCausal := []uint64{0b0001, 0b0011, 0b0111, 0b1111}
	if !slices.Equal(causal, wantCausal) {
		t.Fatalf("CausalDraftMask(4) = %v, want %v", causal, wantCausal)
	}

	// 6 draft tokens with two branches: [0 -> 1 -> 2, 0 -> 3 -> 4, 3 -> 5]
	// Node 0: root (-1)       -> ancestors {0}          = 0b000001
	// Node 1: parent 0        -> ancestors {0, 1}       = 0b000011
	// Node 2: parent 1        -> ancestors {0, 1, 2}    = 0b000111
	// Node 3: parent 0        -> ancestors {0, 3}       = 0b001001
	// Node 4: parent 3        -> ancestors {0, 3, 4}    = 0b011001
	// Node 5: parent 3        -> ancestors {0, 3, 5}    = 0b101001
	parents := []int{-1, 0, 1, 0, 3, 3}
	treeMask, err := TreeDraftMask(parents)
	if err != nil {
		t.Fatalf("TreeDraftMask(%v): %v", parents, err)
	}
	wantTree := []uint64{
		0b000001,
		0b000011,
		0b000111,
		0b001001,
		0b011001,
		0b101001,
	}
	if !slices.Equal(treeMask, wantTree) {
		t.Fatalf("TreeDraftMask(%v) = %06b, want %06b", parents, treeMask, wantTree)
	}

	if _, err := TreeDraftMask([]int{0, 0}); err == nil {
		t.Errorf("TreeDraftMask with parent[0] != -1 succeeded, want error")
	}
	if _, err := TreeDraftMask([]int{-1, 2, 1}); err == nil {
		t.Errorf("TreeDraftMask with parent[i] >= i succeeded, want error")
	}
}

func buildMTPTestCase(r *rand.Rand, a MTPPagedAttention, seqlens []int32, draftMask []uint64, kvScale float32) ([]uint16, MTPPagedCache) {
	batch := len(seqlens)
	totalPages := batch * a.MaxPages
	c := MTPPagedCache{
		PageTable: make([]int32, batch*a.MaxPages),
		SeqLens:   seqlens,
		DraftMask: draftMask,
		KScale:    kvScale,
		VScale:    kvScale,
	}
	perm := r.Perm(totalPages)
	for i := range c.PageTable {
		c.PageTable[i] = int32(perm[i])
	}
	for b, s := range seqlens {
		used := (int(s) + a.PageSize - 1) / a.PageSize
		for p := used; p < a.MaxPages; p++ {
			c.PageTable[b*a.MaxPages+p] = -1
		}
	}
	totalSlots := totalPages * a.PageSize
	live := make([]bool, totalSlots)
	for b, s := range seqlens {
		for t := range int(s) {
			page := int(c.PageTable[b*a.MaxPages+t/a.PageSize])
			live[page*a.PageSize+t%a.PageSize] = true
		}
	}
	kvElems := totalSlots * a.KVHeads * a.HeadDim
	if a.KVFP8 {
		c.KFP8 = make([]uint8, kvElems)
		c.VFP8 = make([]uint8, kvElems)
		for i := range kvElems {
			c.KFP8[i], c.VFP8[i] = 0x7f, 0x7f
			if live[i/(a.KVHeads*a.HeadDim)] {
				c.KFP8[i] = E4M3Bits(float32(r.NormFloat64()*0.5) / kvScale)
				c.VFP8[i] = E4M3Bits(float32(r.NormFloat64()*0.5) / kvScale)
			}
		}
	} else {
		c.KBF16 = make([]uint16, kvElems)
		c.VBF16 = make([]uint16, kvElems)
		for i := range kvElems {
			c.KBF16[i], c.VBF16[i] = 0x7fc0, 0x7fc0
			if live[i/(a.KVHeads*a.HeadDim)] {
				c.KBF16[i] = BF16Bits(float32(r.NormFloat64() * 0.5))
				c.VBF16[i] = BF16Bits(float32(r.NormFloat64() * 0.5))
			}
		}
	}
	qElems := batch * a.QTokensPerSeq * a.NumHeads * a.HeadDim
	q := make([]uint16, qElems)
	for i := range q {
		v := float32(r.NormFloat64() * 0.5)
		if a.WGMMA {
			v = FromE4M3Bits(E4M3Bits(v))
		}
		q[i] = BF16Bits(v)
	}
	if a.Gate {
		c.Gate = make([]uint16, qElems)
		for i := range c.Gate {
			c.Gate[i] = BF16Bits(float32(r.NormFloat64()))
		}
	}
	return q, c
}

func runMTPPaged(t *testing.T, a MTPPagedAttention, q []uint16, c MTPPagedCache) []float64 {
	t.Helper()
	src, err := a.Source()
	if err != nil {
		t.Fatal(err)
	}
	batch := len(c.SeqLens)
	cbufs := []cudahostemu.Buffer{
		{Name: "q", CType: "uint16_t", Data: u16Bytes(q)},
		{Name: "page_table", CType: "int32_t", Data: i32Bytes(c.PageTable)},
		{Name: "seqlens", CType: "int32_t", Data: i32Bytes(c.SeqLens)},
	}
	if a.KVFP8 {
		cbufs = append(cbufs,
			cudahostemu.Buffer{Name: "k_cache", CType: "uint8_t", Data: c.KFP8},
			cudahostemu.Buffer{Name: "v_cache", CType: "uint8_t", Data: c.VFP8},
			cudahostemu.Buffer{Name: "k_scale", CType: "float", Data: f32Bytes([]float32{c.KScale})},
			cudahostemu.Buffer{Name: "v_scale", CType: "float", Data: f32Bytes([]float32{c.VScale})},
		)
	} else {
		cbufs = append(cbufs,
			cudahostemu.Buffer{Name: "k_cache", CType: "uint16_t", Data: u16Bytes(c.KBF16)},
			cudahostemu.Buffer{Name: "v_cache", CType: "uint16_t", Data: u16Bytes(c.VBF16)},
		)
	}
	maskArg := "nullptr"
	if c.DraftMask != nil {
		cbufs = append(cbufs, cudahostemu.Buffer{Name: "draft_mask", CType: "uint64_t", Data: u64Bytes(c.DraftMask)})
		maskArg = "draft_mask"
	}
	if a.Gate {
		cbufs = append(cbufs, cudahostemu.Buffer{Name: "gate", CType: "uint16_t", Data: u16Bytes(c.Gate)})
	}
	cbufs = append(cbufs, cudahostemu.Buffer{Name: "out", CType: "uint16_t", Data: poison(2 * len(q))})

	args := []string{"q", "k_cache", "v_cache", "page_table", "seqlens", maskArg}
	if a.KVFP8 {
		args = append(args, "k_scale", "v_scale")
	}
	if a.Gate {
		args = append(args, "gate")
	}
	args = append(args, "out", "nullptr")
	launch := fmt.Sprintf("%s_launch(%d, %s)", a.Name, batch, strings.Join(args, ", "))

	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		t.Fatalf("emulated kernel: %v\n--- source ---\n%s", err, numbered(src))
	}
	return decode(BF16, res["out"])
}

func TestMTPPagedAttention(t *testing.T) {
	t.Run("CausalAndTreeDraftMask", TestCausalAndTreeDraftMask)

	// Branching speculative tree of 6 draft tokens: [0 -> 1 -> 2, 0 -> 3 -> 4, 3 -> 5]
	tree6, err := TreeDraftMask([]int{-1, 0, 1, 0, 3, 3})
	if err != nil {
		t.Fatal(err)
	}
	// 4-token speculative tree: [0 -> 1 -> 2, 0 -> 3]
	tree4, err := TreeDraftMask([]int{-1, 0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}

	repeatBatch := func(m []uint64, batch int) []uint64 {
		out := make([]uint64, 0, len(m)*batch)
		for range batch {
			out = append(out, m...)
		}
		return out
	}

	tests := []struct {
		name      string
		a         MTPPagedAttention
		seqlens   []int32
		draftMask []uint64
		kvScale   float32
		bound     float64
	}{
		{
			name: "bf16_causal_draft4",
			a: MTPPagedAttention{
				Name:          "mtp_bf16_causal",
				NumHeads:      4,
				KVHeads:       2,
				HeadDim:       32,
				PageSize:      16,
				MaxPages:      4,
				QTokensPerSeq: 4,
			},
			seqlens:   []int32{36, 19},
			draftMask: repeatBatch(CausalDraftMask(4), 2),
			kvScale:   1.0,
			bound:     2e-3,
		},
		{
			name: "bf16_tree6_swizzle_gate",
			a: MTPPagedAttention{
				Name:          "mtp_bf16_tree_swz_gate",
				NumHeads:      4,
				KVHeads:       2,
				HeadDim:       64,
				PageSize:      16,
				MaxPages:      4,
				QTokensPerSeq: 6,
				Softcap:       30.0,
				Gate:          true,
				Swizzle:       true,
			},
			seqlens:   []int32{42, 21},
			draftMask: repeatBatch(tree6, 2),
			kvScale:   1.0,
			bound:     2e-3,
		},
		{
			// G = NumHeads/KVHeads = 2 < 8, but QTokensPerSeq = 4 gives G*QTokensPerSeq = 8, unlocking WGMMA!
			name: "fp8_wgmma_tree4_g2_q4",
			a: MTPPagedAttention{
				Name:          "mtp_fp8_wgmma_tree",
				NumHeads:      4,
				KVHeads:       2,
				HeadDim:       64,
				PageSize:      16,
				MaxPages:      6,
				QTokensPerSeq: 4,
				KVFP8:         true,
				Swizzle:       true,
				WGMMA:         true,
				Gate:          true,
			},
			seqlens:   []int32{70, 25},
			draftMask: repeatBatch(tree4, 2),
			kvScale:   0.5,
			bound:     3e-3,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(101, uint64(len(tc.name))))
			q, c := buildMTPTestCase(r, tc.a, tc.seqlens, tc.draftMask, tc.kvScale)
			want, err := tc.a.Reference(q, c)
			if err != nil {
				t.Fatal(err)
			}
			got := runMTPPaged(t, tc.a, q, c)
			errRel := relErr(got, want)
			if errRel > tc.bound || math.IsNaN(errRel) {
				t.Fatalf("relative L2 error %.4e > bound %.4e", errRel, tc.bound)
			}
			t.Logf("relative L2 error %.4e (bound %.4e)", errRel, tc.bound)
		})
	}

	t.Run("null_draft_mask_defaults_causal", func(t *testing.T) {
		a := MTPPagedAttention{
			Name:          "mtp_null_mask",
			NumHeads:      4,
			KVHeads:       2,
			HeadDim:       32,
			PageSize:      16,
			MaxPages:      4,
			QTokensPerSeq: 4,
		}
		r := rand.New(rand.NewPCG(202, 203))
		q, c := buildMTPTestCase(r, a, []int32{28, 16}, nil, 1.0)
		want, err := a.Reference(q, c)
		if err != nil {
			t.Fatal(err)
		}
		got := runMTPPaged(t, a, q, c)
		if errRel := relErr(got, want); errRel > 2e-3 || math.IsNaN(errRel) {
			t.Fatalf("relative L2 error %.4e", errRel)
		}
	})

	t.Run("validation_errors", func(t *testing.T) {
		bad := []MTPPagedAttention{
			{Name: "1bad", NumHeads: 4, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4, QTokensPerSeq: 4},
			{Name: "mtp_bad_heads", NumHeads: 5, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4, QTokensPerSeq: 4},
			{Name: "mtp_bad_qtok", NumHeads: 4, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4, QTokensPerSeq: 0},
			{Name: "mtp_wgmma_no_fp8", NumHeads: 4, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4, QTokensPerSeq: 4, WGMMA: true, KVFP8: false},
			{Name: "mtp_wgmma_small_rows", NumHeads: 2, KVHeads: 2, HeadDim: 64, PageSize: 16, MaxPages: 4, QTokensPerSeq: 4, WGMMA: true, KVFP8: true},
		}
		for _, b := range bad {
			if _, err := b.Source(); err == nil {
				t.Errorf("expected error for %+v, got nil", b)
			}
		}
	})
}
