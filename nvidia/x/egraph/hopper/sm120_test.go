package hopper

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestSM120BlockScaledGEMM(t *testing.T) {
	tests := []struct {
		name       string
		g          SM120BlockScaledGEMM
		activeCTAs int
	}{
		{
			name: "sm120a_clc_linear_f32_32x32x64",
			g: SM120BlockScaledGEMM{
				Name:     "sm120a_clc_linear",
				M:        32,
				N:        32,
				K:        64,
				BM:       16,
				BN:       16,
				BK:       64,
				Target:   "sm_120a",
				UseCLC:   true,
				Epilogue: TileLinear,
				Out:      F32,
			},
			activeCTAs: 2,
		},
		{
			name: "sm121a_clc_silu_f32_64x64x128",
			g: SM120BlockScaledGEMM{
				Name:     "sm121a_clc_silu",
				M:        64,
				N:        64,
				K:        128,
				BM:       32,
				BN:       32,
				BK:       64,
				Target:   "sm_121a",
				UseCLC:   true,
				Epilogue: TileSiLU,
				Out:      F32,
			},
			activeCTAs: 2,
		},
		{
			name: "sm121a_clc_glu_f32_64x64x64",
			g: SM120BlockScaledGEMM{
				Name:     "sm121a_clc_glu",
				M:        64,
				N:        64,
				K:        64,
				BM:       32,
				BN:       32,
				BK:       64,
				Target:   "sm_121a",
				UseCLC:   true,
				Epilogue: TileGLU,
				Out:      F32,
			},
			activeCTAs: 2,
		},
		{
			name: "sm120a_noclc_linear_f32_64x64x64",
			g: SM120BlockScaledGEMM{
				Name:     "sm120a_noclc_linear",
				M:        64,
				N:        64,
				K:        64,
				BM:       32,
				BN:       32,
				BK:       64,
				Target:   "sm_120a",
				UseCLC:   false,
				Epilogue: TileLinear,
				Out:      F32,
			},
			activeCTAs: 4,
		},
		{
			name: "sm121a_noclc_silu_f32_32x32x64",
			g: SM120BlockScaledGEMM{
				Name:     "sm121a_noclc_silu",
				M:        32,
				N:        32,
				K:        64,
				BM:       16,
				BN:       16,
				BK:       64,
				Target:   "sm_121a",
				UseCLC:   false,
				Epilogue: TileSiLU,
				Out:      F32,
			},
			activeCTAs: 2,
		},
		{
			name: "sm120f_clc_glu_bf16_32x32x64",
			g: SM120BlockScaledGEMM{
				Name:     "sm120f_clc_glu_bf16",
				M:        32,
				N:        32,
				K:        64,
				BM:       16,
				BN:       16,
				BK:       64,
				Target:   "sm_120f",
				UseCLC:   true,
				Epilogue: TileGLU,
				Out:      BF16,
			},
			activeCTAs: 2,
		},
	}

	scaleChoices := []float32{0.125, 0.25, 0.5, 1.0, 1.5}
	ctx := context.Background()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cudaSrc, err := tt.g.CUDA()
			if err != nil {
				t.Fatalf("CUDA() error: %v", err)
			}
			for _, want := range []string{
				"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3",
				"clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128",
				"clusterlaunchcontrol.query_cancel.is_canceled.pred.b128",
				"clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128",
				"eg_emu_mma_nvfp4_16864",
				"eg_clc_try_cancel",
			} {
				if !strings.Contains(cudaSrc, want) {
					t.Errorf("CUDA() missing marker %q", want)
				}
			}
			if strings.Contains(cudaSrc, "tcgen05.") {
				t.Errorf("CUDA() unexpectedly contains tcgen05 instruction on %s", tt.g.Target)
			}

			src, err := tt.g.Source()
			if err != nil {
				t.Fatalf("Source() error: %v", err)
			}

			rng := rand.New(rand.NewPCG(120, uint64(len(tt.name))))
			kFP4 := tt.g.K / 2
			kSF := tt.g.K / 16
			aPacked := make([]uint8, tt.g.M*kFP4)
			aScales := make([]uint8, tt.g.M*kSF)
			bPacked := make([]uint8, tt.g.N*kFP4)
			bScales := make([]uint8, tt.g.N*kSF)

			for i := range aPacked {
				c0 := uint8(rng.IntN(16))
				c1 := uint8(rng.IntN(16))
				aPacked[i] = (c0 & 0xf) | ((c1 & 0xf) << 4)
			}
			for i := range bPacked {
				c0 := uint8(rng.IntN(16))
				c1 := uint8(rng.IntN(16))
				bPacked[i] = (c0 & 0xf) | ((c1 & 0xf) << 4)
			}
			for i := range aScales {
				aScales[i] = E4M3Bits(scaleChoices[rng.IntN(len(scaleChoices))])
			}
			for i := range bScales {
				bScales[i] = E4M3Bits(scaleChoices[rng.IntN(len(scaleChoices))])
			}

			want, err := tt.g.Reference(aPacked, aScales, bPacked, bScales, tt.g.M)
			if err != nil {
				t.Fatalf("Reference() error: %v", err)
			}

			nout := tt.g.N
			if tt.g.Epilogue == TileGLU {
				nout = tt.g.N / 2
			}
			totalTiles := tt.g.TotalTiles(tt.g.M)
			block := tt.g.Block()

			runAndVerify := func(t *testing.T, kernelSrc string) {
				t.Helper()
				outBytes := poison(tt.g.M * nout * tt.g.Out.Bits() / 8)
				err := cudahostemu.RunWithCLC(
					ctx,
					kernelSrc,
					tt.activeCTAs,
					totalTiles,
					cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
					tt.g.SharedBytes(),
					cudahostemu.BufferArg("a_packed", "uint8_t", aPacked),
					cudahostemu.BufferArg("a_scales", "uint8_t", aScales),
					cudahostemu.BufferArg("b_packed", "uint8_t", bPacked),
					cudahostemu.BufferArg("b_scales", "uint8_t", bScales),
					cudahostemu.BufferArg("out", tt.g.Out.cType(), outBytes),
					cudahostemu.IntArg(tt.g.M),
				)
				if errors.Is(err, cudahostemu.ErrNoCompiler) {
					t.Skip(err)
				}
				if err != nil {
					t.Fatalf("RunWithCLC: %v", err)
				}

				gotF64 := decode(tt.g.Out, outBytes)
				var maxDiff float64
				for i, wv := range want {
					gv := gotF64[i]
					diff := math.Abs(gv - float64(wv))
					if math.IsNaN(gv) {
						t.Fatalf("out[%d] is NaN, want %g", i, wv)
					}
					if diff > maxDiff {
						maxDiff = diff
					}
				}
				if maxDiff >= 1e-5 {
					t.Fatalf("max abs diff = %g, want < 1e-5", maxDiff)
				}
			}

			runAndVerify(t, src)

			// Verify under adversarial reversed steal order when UseCLC is enabled.
			if tt.g.UseCLC {
				t.Run("adversarial_reverse_steal_order", func(t *testing.T) {
					oldLine := "clc_next_tile = (int)eg_sm120_clc_query_first_ctaid_x(clc_response);"
					revLine := fmt.Sprintf("const int raw_tile = (int)eg_sm120_clc_query_first_ctaid_x(clc_response); clc_next_tile = %d + (total_tiles - 1 - raw_tile);", tt.activeCTAs)
					if !strings.Contains(src, oldLine) {
						t.Fatalf("missing steal assignment %q", oldLine)
					}
					advSrc := strings.Replace(src, oldLine, revLine, 1)
					runAndVerify(t, advSrc)
				})
			}
		})
	}

	t.Run("clc_hazard_without_wait", func(t *testing.T) {
		g := SM120BlockScaledGEMM{
			Name:     "sm121a_hazard_check",
			M:        32,
			N:        32,
			K:        64,
			BM:       16,
			BN:       16,
			BK:       64,
			Target:   "sm_121a",
			UseCLC:   true,
			Epilogue: TileLinear,
			Out:      F32,
		}
		src, err := g.Source()
		if err != nil {
			t.Fatal(err)
		}
		badSrc := strings.Replace(src, "eg_sm120_clc_wait(&clc_mbar, clc_phase);", "", 1)
		kFP4, kSF := g.K/2, g.K/16
		block := g.Block()
		err = cudahostemu.RunWithCLC(
			ctx,
			badSrc,
			2,
			g.TotalTiles(g.M),
			cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
			g.SharedBytes(),
			cudahostemu.BufferArg("a_packed", "uint8_t", make([]uint8, g.M*kFP4)),
			cudahostemu.BufferArg("a_scales", "uint8_t", make([]uint8, g.M*kSF)),
			cudahostemu.BufferArg("b_packed", "uint8_t", make([]uint8, g.N*kFP4)),
			cudahostemu.BufferArg("b_scales", "uint8_t", make([]uint8, g.N*kSF)),
			cudahostemu.BufferArg("out", "float", make([]uint8, 4*g.M*g.N)),
			cudahostemu.IntArg(g.M),
		)
		if errors.Is(err, cudahostemu.ErrNoCompiler) {
			t.Skip(err)
		}
		if err == nil {
			t.Fatal("expected CLC mbarrier hazard error when eg_sm120_clc_wait is stripped, got nil")
		}
	})

	t.Run("validation_errors", func(t *testing.T) {
		bad := []SM120BlockScaledGEMM{
			{Name: "1bad", M: 64, N: 64, K: 64},
			{Name: "bad_target", M: 64, N: 64, K: 64, Target: "sm_90a"},
			{Name: "bad_m", M: 20, N: 64, K: 64},
			{Name: "bad_n", M: 64, N: 20, K: 64},
			{Name: "bad_k", M: 64, N: 64, K: 32},
			{Name: "bad_out", M: 64, N: 64, K: 64, Out: E4M3},
		}
		for _, b := range bad {
			if _, err := b.CUDA(); err == nil {
				t.Errorf("expected validation error for %+v, got nil", b)
			}
		}
	})
}

func TestSM120PagedAttention(t *testing.T) {
	tests := []struct {
		name     string
		a        SM120PagedAttention
		batch    int
		physPool int
	}{
		{
			name: "bf16_gqa_noncontiguous_sm120a",
			a: SM120PagedAttention{
				Name:       "sm120a_paged_attn_bf16",
				NumQHeads:  8,
				NumKVHeads: 2,
				HeadDim:    64,
				PageSize:   16,
				MaxPages:   6,
				SeqLen:     53,
				Target:     "sm_120a",
				KVFormat:   WeightBF16,
			},
			batch:    2,
			physPool: 16,
		},
		{
			// Four splits of 16 keys over 32-key pages: the boundaries at
			// keys 16 and 48 fall inside pages.
			name: "bf16_gqa_split_kv_sm121a",
			a: SM120PagedAttention{
				Name:         "sm121a_paged_attn_split",
				NumQHeads:    8,
				NumKVHeads:   2,
				HeadDim:      64,
				PageSize:     32,
				MaxPages:     3,
				SeqLen:       53,
				Target:       "sm_121a",
				KVFormat:     WeightBF16,
				Splits:       4,
				MinSplitKeys: 16,
			},
			batch:    2,
			physPool: 8,
		},
		{
			name: "nvfp4_gqa_noncontiguous_sm121a",
			a: SM120PagedAttention{
				Name:       "sm121a_paged_attn_nvfp4",
				NumQHeads:  8,
				NumKVHeads: 2,
				HeadDim:    64,
				PageSize:   16,
				MaxPages:   6,
				SeqLen:     53,
				Target:     "sm_121a",
				KVFormat:   WeightNVFP4,
			},
			batch:    2,
			physPool: 16,
		},
	}

	scaleChoices := []float32{0.125, 0.25, 0.5, 1.0}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cudaSrc, err := tt.a.CUDA()
			if err != nil {
				t.Fatalf("CUDA() error: %v", err)
			}
			for _, want := range []string{
				"exp2f(",
				"page_table[",
			} {
				if !strings.Contains(cudaSrc, want) {
					t.Errorf("CUDA() missing %q", want)
				}
			}
			if tt.a.KVFormat == WeightNVFP4 && !strings.Contains(cudaSrc, "cvt.rn.f16x2.e2m1x2") {
				t.Errorf("NVFP4 CUDA() missing cvt.rn.f16x2.e2m1x2")
			}
			if strings.Contains(cudaSrc, "tcgen05.") {
				t.Errorf("CUDA() unexpectedly contains tcgen05 instruction on %s", tt.a.Target)
			}

			src, err := tt.a.Source()
			if err != nil {
				t.Fatalf("Source() error: %v", err)
			}

			rng := rand.New(rand.NewPCG(321, uint64(len(tt.name))))
			q := make([]float32, tt.batch*tt.a.NumQHeads*tt.a.HeadDim)
			for i := range q {
				q[i] = float32(rng.NormFloat64() * 0.25)
			}

			// Non-contiguous page table mapping across a larger physical page pool.
			perm := rng.Perm(tt.physPool)
			pageTable := make([]int32, tt.batch*tt.a.MaxPages)
			usedPages := (tt.a.SeqLen + tt.a.PageSize - 1) / tt.a.PageSize
			for b := range tt.batch {
				for p := range tt.a.MaxPages {
					if p < usedPages {
						pageTable[b*tt.a.MaxPages+p] = int32(perm[b*tt.a.MaxPages+p])
					} else {
						pageTable[b*tt.a.MaxPages+p] = -1
					}
				}
			}

			totalSlots := tt.physPool * tt.a.PageSize
			liveSlot := make([]bool, totalSlots)
			for b := range tt.batch {
				for tPos := range tt.a.SeqLen {
					page := int(pageTable[b*tt.a.MaxPages+tPos/tt.a.PageSize])
					liveSlot[page*tt.a.PageSize+(tPos%tt.a.PageSize)] = true
				}
			}

			var kPages, vPages, kScales, vScales []uint8
			totalRows := totalSlots * tt.a.NumKVHeads
			switch tt.a.KVFormat {
			case WeightBF16:
				kU16 := make([]uint16, totalRows*tt.a.HeadDim)
				vU16 := make([]uint16, totalRows*tt.a.HeadDim)
				for i := range kU16 {
					slot := i / (tt.a.NumKVHeads * tt.a.HeadDim)
					if liveSlot[slot] {
						kU16[i] = BF16Bits(float32(rng.NormFloat64() * 0.5))
						vU16[i] = BF16Bits(float32(rng.NormFloat64() * 0.5))
					} else {
						// Poison unreferenced pages with NaN to verify non-contiguous page addressing.
						kU16[i] = 0x7fc0
						vU16[i] = 0x7fc0
					}
				}
				kPages = u16Bytes(kU16)
				vPages = u16Bytes(vU16)
				kScales = []uint8{0}
				vScales = []uint8{0}
			case WeightNVFP4:
				dFP4 := tt.a.HeadDim / 2
				dSF := tt.a.HeadDim / 16
				kPages = make([]uint8, totalRows*dFP4)
				vPages = make([]uint8, totalRows*dFP4)
				kScales = make([]uint8, totalRows*dSF)
				vScales = make([]uint8, totalRows*dSF)
				for row := range totalRows {
					slot := row / tt.a.NumKVHeads
					for j := range dFP4 {
						kPages[row*dFP4+j] = uint8(rng.IntN(256))
						vPages[row*dFP4+j] = uint8(rng.IntN(256))
					}
					for j := range dSF {
						if liveSlot[slot] {
							kScales[row*dSF+j] = E4M3Bits(scaleChoices[rng.IntN(len(scaleChoices))])
							vScales[row*dSF+j] = E4M3Bits(scaleChoices[rng.IntN(len(scaleChoices))])
						} else {
							// Poison unreferenced pages with E4M3 NaN (0x7f).
							kScales[row*dSF+j] = 0x7f
							vScales[row*dSF+j] = 0x7f
						}
					}
				}
			}

			want, err := tt.a.Reference(q, kPages, vPages, kScales, vScales, pageTable, tt.a.SeqLen)
			if err != nil {
				t.Fatalf("Reference() error: %v", err)
			}

			ws := ""
			buffers := []cudahostemu.Buffer{
				{Name: "q", CType: "float", Data: f32Bytes(q)},
				{Name: "k_pages", CType: "uint8_t", Data: kPages},
				{Name: "v_pages", CType: "uint8_t", Data: vPages},
				{Name: "k_scales", CType: "uint8_t", Data: kScales},
				{Name: "v_scales", CType: "uint8_t", Data: vScales},
				{Name: "page_table", CType: "int32_t", Data: i32Bytes(pageTable)},
				{Name: "out", CType: "float", Data: poison(4 * len(q))},
			}
			if n := tt.a.WorkspaceFloats(tt.batch); n > 0 {
				ws = "ws, "
				buffers = append(buffers, cudahostemu.Buffer{Name: "ws", CType: "float", Data: poison(4 * n)})
			}
			launch := fmt.Sprintf("%s_launch(%d, q, k_pages, v_pages, k_scales, v_scales, page_table, %sout, %d)",
				tt.a.Name, tt.batch, ws, tt.a.SeqLen)
			res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
				Source:  src,
				Launch:  launch,
				Buffers: buffers,
				Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
			})
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("cudahostemu.Run: %v", err)
			}

			got := bytesF32(res["out"])
			var maxDiff float64
			for i, wv := range want {
				gv := got[i]
				if math.IsNaN(float64(gv)) {
					t.Fatalf("out[%d] is NaN, want %g", i, wv)
				}
				diff := math.Abs(float64(gv - wv))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			if maxDiff >= 1e-4 {
				t.Fatalf("max abs diff = %g, want < 1e-4", maxDiff)
			}
		})
	}

	t.Run("validation_errors", func(t *testing.T) {
		bad := []SM120PagedAttention{
			{Name: "1bad", NumQHeads: 8, NumKVHeads: 2, HeadDim: 64},
			{Name: "bad_heads", NumQHeads: 7, NumKVHeads: 2, HeadDim: 64},
			{Name: "bad_dim", NumQHeads: 8, NumKVHeads: 2, HeadDim: 48, KVFormat: WeightNVFP4},
			{Name: "bad_target", NumQHeads: 8, NumKVHeads: 2, HeadDim: 64, Target: "sm_90a", KVFormat: WeightNVFP4},
			{Name: "bad_kv", NumQHeads: 8, NumKVHeads: 2, HeadDim: 64, KVFormat: WeightE4M3},
			{Name: "bad_splits", NumQHeads: 8, NumKVHeads: 2, HeadDim: 64, Splits: -1},
			{Name: "bad_split_kv", NumQHeads: 8, NumKVHeads: 2, HeadDim: 64, KVFormat: WeightNVFP4, Splits: 2},
			{Name: "bad_split_keys", NumQHeads: 8, NumKVHeads: 2, HeadDim: 64, Splits: 2, MinSplitKeys: 24},
		}
		for _, b := range bad {
			if b.Name == "bad_dim" {
				b.HeadDim = 24
			}
			if _, err := b.CUDA(); err == nil {
				t.Errorf("expected error for %+v, got nil", b)
			}
		}
	})
}

func ExampleSM120BlockScaledGEMM() {
	g := SM120BlockScaledGEMM{
		Name:     "gb10_nvfp4_gemm",
		M:        64,
		N:        64,
		K:        128,
		BM:       32,
		BN:       32,
		BK:       64,
		Target:   "sm_121a",
		UseCLC:   true,
		Epilogue: TileSiLU,
		Out:      F32,
	}
	src, err := g.CUDA()
	if err != nil {
		panic(err)
	}
	fmt.Println("grid:", g.Grid(64), "block:", g.Block())
	fmt.Println("has m16n8k64 mxf4nvf4:", strings.Contains(src, "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"))
	fmt.Println("has clc try_cancel:", strings.Contains(src, "clusterlaunchcontrol.try_cancel"))
	// Output:
	// grid: [4 1 1] block: [128 1 1]
	// has m16n8k64 mxf4nvf4: true
	// has clc try_cancel: true
}

func TestSM120PagedAttentionTarget(t *testing.T) {
	tests := []struct {
		target string
		kv     WeightFormat
		ok     bool
	}{
		{"sm_121a", WeightBF16, true},
		{"sm_121a", WeightNVFP4, true},
		{"sm_120f", WeightNVFP4, true},
		{"sm_103a", WeightBF16, true},
		{"sm_90a", WeightBF16, true},
		{"sm_100f", WeightBF16, true},
		{"sm_103a", WeightNVFP4, false},
		{"sm_75", WeightBF16, false},
		{"gb300", WeightBF16, false},
	}
	for _, tt := range tests {
		a := SM120PagedAttention{Name: "attn", Target: tt.target, KVFormat: tt.kv}
		_, err := a.norm()
		if (err == nil) != tt.ok {
			t.Errorf("target %s, kv %v: err = %v, want ok %v", tt.target, tt.kv, err, tt.ok)
		}
	}
}
