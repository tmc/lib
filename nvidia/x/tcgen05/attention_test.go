package tcgen05

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func i32Bytes(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

func TestTCGen05PagedAttention(t *testing.T) {
	tests := []struct {
		name     string
		target   string
		ctaGroup int
		pingPong bool
		kvFormat WeightFormat
		tol      float64
		wantPTX  []string
	}{
		{
			name:     "sm100a_cta1_bf16_pingpong",
			target:   "sm_100a",
			ctaGroup: 1,
			pingPong: true,
			kvFormat: WeightBF16,
			tol:      1e-4,
			wantPTX: []string{
				"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
				"tcgen05.mma.cta_group::1",
				"tcgen05.ld.sync.aligned.32x32b.x1.b32",
				"tcgen05.st.sync.aligned.32x32b.x1.b32",
				"tcgen05.commit.cta_group::1",
				"tcgen05.dealloc.cta_group::1.sync.aligned.b32",
			},
		},
		{
			name:     "sm100a_cta2_bf16_pingpong",
			target:   "sm_100a",
			ctaGroup: 2,
			pingPong: true,
			kvFormat: WeightBF16,
			tol:      1e-4,
			wantPTX: []string{
				"__cluster_dims__(2, 1, 1)",
				"tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32",
				"tcgen05.mma.cta_group::2",
				"tcgen05.commit.cta_group::2",
				"tcgen05.dealloc.cta_group::2.sync.aligned.b32",
			},
		},
		{
			name:     "sm103a_cta1_bf16_pingpong",
			target:   "sm_103a",
			ctaGroup: 1,
			pingPong: true,
			kvFormat: WeightBF16,
			tol:      1e-4,
			wantPTX: []string{
				"tcgen05.mma.cta_group::1",
				"eg_fa4_exp2_2x",
			},
		},
		{
			name:     "sm100a_cta1_nvfp4_pingpong",
			target:   "sm_100a",
			ctaGroup: 1,
			pingPong: true,
			kvFormat: WeightNVFP4,
			tol:      5e-2,
			wantPTX: []string{
				"tcgen05.cp.cta_group::1.32x128b",
				"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale",
			},
		},
		{
			name:     "sm103a_cta1_nvfp4_scale_vec_4x",
			target:   "sm_103a",
			ctaGroup: 1,
			pingPong: true,
			kvFormat: WeightNVFP4,
			tol:      5e-2,
			wantPTX: []string{
				"tcgen05.cp.cta_group::1.32x128b",
				"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
				"scale_vec::4X",
			},
		},
		{
			name:     "sm103a_cta2_nvfp4_scale_vec_4x",
			target:   "sm_103a",
			ctaGroup: 2,
			pingPong: true,
			kvFormat: WeightNVFP4,
			tol:      5e-2,
			wantPTX: []string{
				"__cluster_dims__(2, 1, 1)",
				"tcgen05.cp.cta_group::2.32x128b",
				"tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X",
				"scale_vec::4X",
			},
		},
	}

	const (
		numQHeads  = 4
		numKVHeads = 2
		headDim    = 64
		pageSize   = 8
		maxPages   = 8
		seqLen     = 20 // spans 3 pages (8 + 8 + 4)
		totalPhys  = 6
	)
	pageTable := []int32{4, 1, 5, 0, 2, 3, 0, 0} // non-contiguous physical pages

	rng := rand.New(rand.NewPCG(2026, 901))
	q := make([]float32, numQHeads*headDim)
	kPages := make([]float32, totalPhys*pageSize*numKVHeads*headDim)
	vPages := make([]float32, totalPhys*pageSize*numKVHeads*headDim)
	for i := range q {
		q[i] = float32(rng.IntN(9)-4) * 0.125
	}
	for i := range kPages {
		kPages[i] = float32(rng.IntN(9)-4) * 0.125
	}
	for i := range vPages {
		vPages[i] = float32(rng.IntN(9)-4) * 0.125
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			kern := TCGen05PagedAttention{
				Name:       "fa4_" + tc.name,
				NumQHeads:  numQHeads,
				NumKVHeads: numKVHeads,
				HeadDim:    headDim,
				PageSize:   pageSize,
				MaxPages:   maxPages,
				SeqLen:     seqLen,
				Target:     tc.target,
				CTAGroup:   tc.ctaGroup,
				PingPong:   tc.pingPong,
				KVFormat:   tc.kvFormat,
			}

			// 1. Verify multi-page TMEM trace on target.
			trace := kern.TMEMTrace()
			sum, err := cudatmem.VerifyTrace(cudatmem.Target(tc.target), trace)
			if err != nil {
				t.Fatalf("VerifyTrace(%s) failed: %v", tc.target, err)
			}
			if sum.CTAs != tc.ctaGroup || sum.CTAGroup != tc.ctaGroup || sum.MMAs != 6 {
				t.Fatalf("unexpected trace summary: %+v", sum)
			}

			// 2. Verify PTX markers in CUDA().
			cudaSrc := kern.CUDA()
			for _, marker := range tc.wantPTX {
				if !strings.Contains(cudaSrc, marker) {
					t.Errorf("CUDA() missing marker %q", marker)
				}
			}

			// 3. Verify host emulation via cudahostemu.Run matches Reference.
			src := kern.Source()
			launch := fmt.Sprintf("%s_launch(q, k_pages, v_pages, page_table, out)", kern.Name)
			res := runEmu(t, src, launch,
				emuBuf{"q", "float", f32Bytes(q)},
				emuBuf{"k_pages", "float", f32Bytes(kPages)},
				emuBuf{"v_pages", "float", f32Bytes(vPages)},
				emuBuf{"page_table", "int32_t", i32Bytes(pageTable)},
				emuBuf{"out", "float", poison(4 * numQHeads * headDim)},
			)
			got := bytesF32(res["out"])
			want := kern.Reference(q, kPages, vPages, pageTable, seqLen)

			var maxDiff float64
			for i := range want {
				if math.IsNaN(float64(got[i])) {
					t.Fatalf("out[%d] is NaN (want %g)", i, want[i])
				}
				diff := math.Abs(float64(got[i]) - float64(want[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff > tc.tol {
					t.Fatalf("out[%d] = %g, want %g (diff %g > tol %g)", i, got[i], want[i], diff, tc.tol)
				}
			}
			t.Logf("%s: max abs diff vs float64 Reference = %g (tol %g)", tc.name, maxDiff, tc.tol)
		})
	}
}

func TestTCGen05MLAPagedAttention(t *testing.T) {
	tests := []struct {
		name     string
		target   string
		ctaGroup int
		pingPong bool
	}{
		{
			name:     "sm100a_cta1_pingpong",
			target:   "sm_100a",
			ctaGroup: 1,
			pingPong: true,
		},
		{
			name:     "sm103a_cta1_pingpong",
			target:   "sm_103a",
			ctaGroup: 1,
			pingPong: true,
		},
		{
			name:     "sm103a_cta2_pingpong",
			target:   "sm_103a",
			ctaGroup: 2,
			pingPong: true,
		},
	}

	const (
		numHeads   = 4
		kvLoRARank = 16
		ropeDim    = 8
		pageSize   = 8
		maxPages   = 8
		seqLen     = 19 // spans 3 pages (8 + 8 + 3)
		totalPhys  = 6
	)
	pageTable := []int32{3, 0, 5, 2, 1, 4, 0, 0} // non-contiguous physical pages

	rng := rand.New(rand.NewPCG(2026, 902))
	qAbsorbed := make([]float32, numHeads*kvLoRARank)
	qRope := make([]float32, numHeads*ropeDim)
	cKVPages := make([]float32, totalPhys*pageSize*kvLoRARank)
	kRopePages := make([]float32, totalPhys*pageSize*ropeDim)

	for i := range qAbsorbed {
		qAbsorbed[i] = float32(rng.NormFloat64() * 0.25)
	}
	for i := range qRope {
		qRope[i] = float32(rng.NormFloat64() * 0.25)
	}
	for i := range cKVPages {
		cKVPages[i] = float32(rng.NormFloat64() * 0.25)
	}
	for i := range kRopePages {
		kRopePages[i] = float32(rng.NormFloat64() * 0.25)
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			kern := TCGen05MLAPagedAttention{
				Name:       "mla_" + tc.name,
				NumHeads:   numHeads,
				KVLoRARank: kvLoRARank,
				RoPEDim:    ropeDim,
				VHeadDim:   kvLoRARank,
				PageSize:   pageSize,
				MaxPages:   maxPages,
				SeqLen:     seqLen,
				Target:     tc.target,
				CTAGroup:   tc.ctaGroup,
				PingPong:   tc.pingPong,
			}

			// 1. Verify multi-page TMEM trace.
			trace := kern.TMEMTrace()
			sum, err := cudatmem.VerifyTrace(cudatmem.Target(tc.target), trace)
			if err != nil {
				t.Fatalf("VerifyTrace(%s) failed: %v", tc.target, err)
			}
			if sum.CTAs != tc.ctaGroup || sum.CTAGroup != tc.ctaGroup || sum.MMAs != 9 {
				t.Fatalf("unexpected MLA trace summary: %+v", sum)
			}

			// 2. Verify host emulation via cudahostemu.Run matches Reference within < 1e-4.
			src := kern.Source()
			launch := fmt.Sprintf("%s_launch(q_absorbed, q_rope, c_kv_pages, k_rope_pages, page_table, out)", kern.Name)
			res := runEmu(t, src, launch,
				emuBuf{"q_absorbed", "float", f32Bytes(qAbsorbed)},
				emuBuf{"q_rope", "float", f32Bytes(qRope)},
				emuBuf{"c_kv_pages", "float", f32Bytes(cKVPages)},
				emuBuf{"k_rope_pages", "float", f32Bytes(kRopePages)},
				emuBuf{"page_table", "int32_t", i32Bytes(pageTable)},
				emuBuf{"out", "float", poison(4 * numHeads * kvLoRARank)},
			)
			got := bytesF32(res["out"])
			want := kern.Reference(qAbsorbed, qRope, cKVPages, kRopePages, pageTable, seqLen)

			var maxDiff float64
			for i := range want {
				if math.IsNaN(float64(got[i])) {
					t.Fatalf("out[%d] is NaN (want %g)", i, want[i])
				}
				diff := math.Abs(float64(got[i]) - float64(want[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if diff >= 1e-4 {
					t.Fatalf("out[%d] = %g, want %g (diff %g >= 1e-4)", i, got[i], want[i], diff)
				}
			}
			t.Logf("%s: max abs diff vs float64 Reference = %g", tc.name, maxDiff)
		})
	}
}

func ExampleTCGen05PagedAttention() {
	attn := TCGen05PagedAttention{
		Name:       "fa4_paged_gqa",
		NumQHeads:  4,
		NumKVHeads: 2,
		HeadDim:    64,
		PageSize:   16,
		MaxPages:   8,
		SeqLen:     32,
		Target:     "sm_103a",
		CTAGroup:   2,
		PingPong:   true,
		KVFormat:   WeightNVFP4,
	}
	sum, err := cudatmem.VerifyTrace(cudatmem.SM103A, attn.TMEMTrace())
	fmt.Printf("err=%v ctas=%d group=%d mmas=%d copies=%d has_scale_vec_4x=%v\n",
		err, sum.CTAs, sum.CTAGroup, sum.MMAs, sum.Copies,
		strings.Contains(attn.CUDA(), "scale_vec::4X"))
	// Output:
	// err=<nil> ctas=2 group=2 mmas=4 copies=2 has_scale_vec_4x=true
}

func ExampleTCGen05MLAPagedAttention() {
	mla := TCGen05MLAPagedAttention{
		Name:       "fa4_mla_paged",
		NumHeads:   4,
		KVLoRARank: 16,
		RoPEDim:    8,
		VHeadDim:   16,
		PageSize:   16,
		MaxPages:   8,
		SeqLen:     32,
		Target:     "sm_103a",
		CTAGroup:   1,
		PingPong:   true,
	}
	sum, err := cudatmem.VerifyTrace(cudatmem.SM103A, mla.TMEMTrace())
	fmt.Printf("err=%v ctas=%d group=%d mmas=%d grid=%v block=%v\n",
		err, sum.CTAs, sum.CTAGroup, sum.MMAs, mla.Grid(), mla.Block())
	// Output:
	// err=<nil> ctas=1 group=1 mmas=6 grid=[1 1 1] block=[288 1 1]
}
