package hopper

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func buildMLATestCase(r *rand.Rand, a MLAPagedAttention, seqlens []int32, cScale, kpeScale float32) (qNope, qPe []uint16, c MLAPagedCache) {
	batch := len(seqlens)
	totalPages := batch * a.MaxPages
	c = MLAPagedCache{
		PageTable: make([]int32, batch*a.MaxPages),
		SeqLens:   seqlens,
		CScale:    cScale,
		KPEScale:  kpeScale,
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
	cElems := totalSlots * a.LatentDim
	rElems := totalSlots * a.RopeDim
	if a.KVFP8 {
		c.CKVFP8 = make([]uint8, cElems)
		c.KPEFP8 = make([]uint8, rElems)
		for i := range cElems {
			c.CKVFP8[i] = 0x7f
			if live[i/a.LatentDim] {
				c.CKVFP8[i] = E4M3Bits(float32(r.NormFloat64()*0.5) / cScale)
			}
		}
		for i := range rElems {
			c.KPEFP8[i] = 0x7f
			if live[i/a.RopeDim] {
				c.KPEFP8[i] = E4M3Bits(float32(r.NormFloat64()*0.5) / kpeScale)
			}
		}
	} else {
		c.CKVBF16 = make([]uint16, cElems)
		c.KPEBF16 = make([]uint16, rElems)
		for i := range cElems {
			c.CKVBF16[i] = 0x7fc0
			if live[i/a.LatentDim] {
				c.CKVBF16[i] = BF16Bits(float32(r.NormFloat64() * 0.5))
			}
		}
		for i := range rElems {
			c.KPEBF16[i] = 0x7fc0
			if live[i/a.RopeDim] {
				c.KPEBF16[i] = BF16Bits(float32(r.NormFloat64() * 0.5))
			}
		}
	}
	qNope = make([]uint16, batch*a.NumHeads*a.LatentDim)
	for i := range qNope {
		v := float32(r.NormFloat64() * 0.5)
		if a.WGMMA {
			v = FromE4M3Bits(E4M3Bits(v))
		}
		qNope[i] = BF16Bits(v)
	}
	qPe = make([]uint16, batch*a.NumHeads*a.RopeDim)
	for i := range qPe {
		v := float32(r.NormFloat64() * 0.5)
		if a.WGMMA {
			v = FromE4M3Bits(E4M3Bits(v))
		}
		qPe[i] = BF16Bits(v)
	}
	return qNope, qPe, c
}

func runMLAPaged(t *testing.T, a MLAPagedAttention, qNope, qPe []uint16, c MLAPagedCache) []float64 {
	t.Helper()
	src, err := a.Source()
	if err != nil {
		t.Fatal(err)
	}
	batch := len(c.SeqLens)
	cbufs := []cudahostemu.Buffer{
		{Name: "q_nope", CType: "uint16_t", Data: u16Bytes(qNope)},
		{Name: "q_pe", CType: "uint16_t", Data: u16Bytes(qPe)},
		{Name: "page_table", CType: "int32_t", Data: i32Bytes(c.PageTable)},
		{Name: "seqlens", CType: "int32_t", Data: i32Bytes(c.SeqLens)},
	}
	if a.KVFP8 {
		cbufs = append(cbufs,
			cudahostemu.Buffer{Name: "c_kv_cache", CType: "uint8_t", Data: c.CKVFP8},
			cudahostemu.Buffer{Name: "k_pe_cache", CType: "uint8_t", Data: c.KPEFP8},
			cudahostemu.Buffer{Name: "c_scale", CType: "float", Data: f32Bytes([]float32{c.CScale})},
			cudahostemu.Buffer{Name: "kpe_scale", CType: "float", Data: f32Bytes([]float32{c.KPEScale})},
		)
	} else {
		cbufs = append(cbufs,
			cudahostemu.Buffer{Name: "c_kv_cache", CType: "uint16_t", Data: u16Bytes(c.CKVBF16)},
			cudahostemu.Buffer{Name: "k_pe_cache", CType: "uint16_t", Data: u16Bytes(c.KPEBF16)},
		)
	}
	cbufs = append(cbufs, cudahostemu.Buffer{Name: "out", CType: "uint16_t", Data: poison(2 * len(qNope))})

	args := []string{"q_nope", "q_pe", "c_kv_cache", "k_pe_cache", "page_table", "seqlens"}
	if a.KVFP8 {
		args = append(args, "c_scale", "kpe_scale")
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

func TestMLAPagedAttention(t *testing.T) {
	tests := []struct {
		name     string
		a        MLAPagedAttention
		seqlens  []int32
		cScale   float32
		kpeScale float32
		bound    float64
	}{
		{
			name: "bf16_latent_kv_b2_h8_dc64_dr32",
			a: MLAPagedAttention{
				Name:      "mla_bf16_h8",
				NumHeads:  8,
				LatentDim: 64,
				RopeDim:   32,
				PageSize:  16,
				MaxPages:  4,
			},
			seqlens:  []int32{37, 19},
			cScale:   1.0,
			kpeScale: 1.0,
			bound:    2e-3,
		},
		{
			name: "bf16_swizzle_b2_h16_dc64_dr32",
			a: MLAPagedAttention{
				Name:      "mla_bf16_swz_h16",
				NumHeads:  16,
				LatentDim: 64,
				RopeDim:   32,
				PageSize:  16,
				MaxPages:  4,
				Swizzle:   true,
			},
			seqlens:  []int32{45, 22},
			cScale:   1.0,
			kpeScale: 1.0,
			bound:    2e-3,
		},
		{
			name: "fp8_wgmma_swizzle_b2_h8_dc64_dr32",
			a: MLAPagedAttention{
				Name:      "mla_fp8_wgmma_swz_h8",
				NumHeads:  8,
				LatentDim: 64,
				RopeDim:   32,
				PageSize:  16,
				MaxPages:  6,
				KVFP8:     true,
				Swizzle:   true,
				WGMMA:     true,
			},
			seqlens:  []int32{73, 29},
			cScale:   0.5,
			kpeScale: 0.25,
			bound:    3e-3,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := rand.New(rand.NewPCG(303, uint64(len(tc.name))))
			qNope, qPe, c := buildMLATestCase(r, tc.a, tc.seqlens, tc.cScale, tc.kpeScale)
			want, err := tc.a.Reference(qNope, qPe, c)
			if err != nil {
				t.Fatal(err)
			}
			got := runMLAPaged(t, tc.a, qNope, qPe, c)
			errRel := relErr(got, want)
			if errRel > tc.bound || math.IsNaN(errRel) {
				t.Fatalf("relative L2 error %.4e > bound %.4e", errRel, tc.bound)
			}
			t.Logf("relative L2 error %.4e (bound %.4e)", errRel, tc.bound)
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		bad := []MLAPagedAttention{
			{Name: "1bad", NumHeads: 8, LatentDim: 64, RopeDim: 32, PageSize: 16, MaxPages: 4},
			{Name: "mla_bad_heads", NumHeads: 0, LatentDim: 64, RopeDim: 32, PageSize: 16, MaxPages: 4},
			{Name: "mla_bad_dc", NumHeads: 8, LatentDim: 48, RopeDim: 32, PageSize: 16, MaxPages: 4, KVFP8: true, WGMMA: true},
			{Name: "mla_wgmma_no_fp8", NumHeads: 8, LatentDim: 64, RopeDim: 32, PageSize: 16, MaxPages: 4, WGMMA: true, KVFP8: false},
			{Name: "mla_wgmma_small_heads", NumHeads: 4, LatentDim: 64, RopeDim: 32, PageSize: 16, MaxPages: 4, WGMMA: true, KVFP8: true},
		}
		for _, b := range bad {
			if _, err := b.Source(); err == nil {
				t.Errorf("expected error for %+v, got nil", b)
			}
		}
	})
}
