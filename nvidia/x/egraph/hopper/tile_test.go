package hopper

import (
	"bytes"
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestTileGEMMEmulated(t *testing.T) {
	tests := []struct {
		name    string
		g       TileGEMM
		m       int     // rows of out
		tokens  int     // rows of x with Gather
		offsets []int32 // Grouped
	}{
		{"bf16_64_s2", TileGEMM{N: 128, K: 192, Weight: WeightBF16, Out: F32, BM: 64, BN: 64, Stages: 2}, 70, 0, nil},
		{"bf16_128_epilogue", TileGEMM{N: 128, K: 256, Weight: WeightBF16, Out: BF16,
			Epilogue: MustLambda("(add (mul $0 $1) (mul $2 $3))"),
			Extras:   []Operand{{"scale", BF16, PerCol}, {"resid", F32, Full}, {"rw", F32, PerRow}}}, 130, 0, nil},
		{"nvfp4_64", TileGEMM{N: 64, K: 256, Weight: WeightNVFP4, Out: F32, BM: 64, BN: 64, Stages: 4}, 40, 0, nil},
		{"grouped_nvfp4_gather", TileGEMM{N: 64, K: 128, Weight: WeightNVFP4, Out: BF16, BM: 64, BN: 64,
			Epilogue: MustLambda("(silu $0)"), Grouped: true, Gather: true}, 90, 30, []int32{0, 17, 17, 90}},
		{"glu_nvfp4_grouped", TileGEMM{N: 128, K: 128, Weight: WeightNVFP4, Out: BF16, BM: 64, BN: 64,
			Epilogue: MustLambda("(mul (silu $0) $1)"), Grouped: true, Gather: true, GLU: true}, 70, 25, []int32{0, 3, 70}},
		{"glu_bf16_128", TileGEMM{N: 256, K: 64, Weight: WeightBF16, Out: F32, GLU: true,
			Epilogue: MustLambda("(add (mul (silu $0) $1) $2)"), Extras: []Operand{{"bias", F32, PerCol}}}, 20, 0, nil},
		{"wgmma_128", TileGEMM{N: 128, K: 192, Weight: WeightBF16, Out: F32, WGMMA: true, Stages: 2}, 150, 0, nil},
		{"wgmma_256_epilogue", TileGEMM{N: 256, K: 128, Weight: WeightBF16, Out: BF16, BN: 256, WGMMA: true, Stages: 4,
			Epilogue: MustLambda("(add (mul $0 $1) $2)"), Extras: []Operand{{"scale", BF16, PerCol}, {"resid", F32, Full}}}, 70, 0, nil},
		{"wgmma_glu_grouped", TileGEMM{N: 128, K: 64, Weight: WeightBF16, Out: BF16, BN: 64, WGMMA: true,
			Epilogue: MustLambda("(mul (silu $0) $1)"), Grouped: true, Gather: true, GLU: true}, 140, 33, []int32{0, 129, 129, 140}},
		{"grouped_bf16", TileGEMM{N: 128, K: 64, Weight: WeightBF16, Out: F32, BM: 64, BN: 128, Grouped: true}, 75, 0, []int32{0, 64, 65, 75}},
		// The MoE down projection with its post-norm scale and row sums of
		// squares: 32-column groups of 2 warps (BN 64) and of 1 warp (BN 128).
		{"sumsq_nvfp4_grouped", TileGEMM{N: 256, K: 128, Weight: WeightNVFP4, Out: BF16, BM: 64, BN: 64, Grouped: true, SumSq: true,
			Epilogue: MustLambda("(mul $0 (add 1 $1))"), Extras: []Operand{{"post_w", BF16, PerCol}}}, 75, 0, []int32{0, 64, 65, 75}},
		{"sumsq_bf16_128", TileGEMM{N: 256, K: 64, Weight: WeightBF16, Out: F32, SumSq: true}, 130, 0, nil},
		// GLU at BN 64: 32 output columns per block, 4 warps per group.
		{"sumsq_glu", TileGEMM{N: 128, K: 64, Weight: WeightBF16, Out: BF16, BM: 64, BN: 64, GLU: true, SumSq: true,
			Epilogue: MustLambda("(mul (silu $0) $1)")}, 40, 0, nil},
		// 128B shared-memory swizzle + WGMMA descriptor mode 1.
		{"wgmma_swizzle128b_bf16", TileGEMM{N: 128, K: 192, Weight: WeightBF16, Out: F32, WGMMA: true, Swizzle128B: true, Stages: 2}, 150, 0, nil},
		{"wgmma_swizzle128b_bf16_256", TileGEMM{N: 256, K: 128, Weight: WeightBF16, Out: BF16, BN: 256, WGMMA: true, Swizzle128B: true, Stages: 3,
			Epilogue: MustLambda("(add (mul $0 $1) $2)"), Extras: []Operand{{"scale", BF16, PerCol}, {"resid", F32, Full}}}, 70, 0, nil},
		{"wgmma_swizzle128b_nvfp4", TileGEMM{N: 128, K: 128, Weight: WeightNVFP4, Out: F32, BN: 64, WGMMA: true, Swizzle128B: true, GLU: true,
			Epilogue: MustLambda("(mul (silu $0) $1)")}, 90, 0, nil},
		{"wgmma_swizzle128b_nvfp4_repacked", TileGEMM{N: 256, K: 128, Weight: WeightNVFP4, Out: BF16, BN: 128, WGMMA: true, Repacked: true, Swizzle128B: true,
			GLU: true, Grouped: true, Gather: true, SumSq: true, Epilogue: MustLambda("(mul (silu $0) $1)")}, 70, 25, []int32{0, 19, 70}},
		// L2 GroupM tile scheduling across ragged multi-tile grids.
		{"groupm_mma_sync", TileGEMM{N: 256, K: 64, Weight: WeightBF16, Out: F32, BM: 64, BN: 64, GroupM: 4, SumSq: true}, 300, 0, nil},
		{"groupm_wgmma_glu_grouped", TileGEMM{N: 256, K: 128, Weight: WeightBF16, Out: BF16, BN: 64, WGMMA: true, Swizzle128B: true, GroupM: 4,
			GLU: true, Grouped: true, Gather: true, SumSq: true, Epilogue: MustLambda("(mul (silu $0) $1)")}, 290, 40, []int32{0, 140, 290}},
		// Native Hopper FP8 (WeightE4M3) WGMMA m64nBNk32, unswizzled and 128B-swizzled.
		{"wgmma_e4m3_64", TileGEMM{N: 128, K: 192, Weight: WeightE4M3, Out: F32, BN: 64, WGMMA: true, Stages: 2}, 135, 0, nil},
		{"wgmma_e4m3_swizzle128b", TileGEMM{N: 256, K: 256, Weight: WeightE4M3, Out: BF16, BN: 128, WGMMA: true, Swizzle128B: true, GroupM: 4,
			Epilogue: MustLambda("(add (mul $0 $1) $2)"), Extras: []Operand{{"scale", F32, PerCol}, {"bias", BF16, PerCol}}}, 150, 0, nil},
		{"wgmma_e4m3_glu_grouped", TileGEMM{N: 128, K: 128, Weight: WeightE4M3, Out: F32, BN: 64, WGMMA: true, Swizzle128B: true,
			GLU: true, Grouped: true, Gather: true, SumSq: true,
			Epilogue: MustLambda("(mul (mul (silu $0) $1) $2)"), Extras: []Operand{{"scale", F32, PerCol}}}, 140, 30, []int32{0, 129, 140}},
	}
	r := rand.New(rand.NewPCG(41, 42))
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "tile_" + tt.name
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			n, k, m := g.N, g.K, tt.m
			xrows := m
			var rows []int32
			if g.Gather {
				xrows = tt.tokens
				for range m {
					rows = append(rows, int32(r.IntN(tt.tokens)))
				}
			}
			var x []uint16
			var xE4 []uint8
			if g.Weight == WeightE4M3 {
				xE4 = make([]uint8, xrows*k)
				x = make([]uint16, xrows*k)
				for i := range xE4 {
					b := E4M3Bits(float32(r.NormFloat64() * 0.5))
					xE4[i] = b
					x[i] = uint16(b)
				}
			} else {
				x = randBF16(r, xrows*k, 1)
			}
			nout := n
			if g.GLU {
				nout = n / 2
			}
			experts := 1
			if g.Grouped {
				experts = len(tt.offsets) - 1
			}
			var ws []GEMMWeights
			var all GEMMWeights
			var repackedW []uint32
			var repackedSF []uint8
			var gs []float32
			for range experts {
				var w GEMMWeights
				switch g.Weight {
				case WeightNVFP4:
					w = randNVFP4(r, n, k)
					w.GlobalScal = float32(0.01 + r.Float64()*0.01)
					if g.WGMMA {
						w.GlobalScal = float32(1.0 / 64.0)
					}
					if g.Repacked {
						q, sf, err := RepackNVFP4WGMMA(w.Packed, w.Scales, n, k, g.GLU)
						if err != nil {
							t.Fatal(err)
						}
						repackedW = append(repackedW, q...)
						repackedSF = append(repackedSF, sf...)
					} else {
						all.Packed = append(all.Packed, w.Packed...)
						all.Scales = append(all.Scales, w.Scales...)
					}
					gs = append(gs, w.GlobalScal)
				case WeightE4M3:
					w.E4M3 = make([]uint8, n*k)
					for i := range w.E4M3 {
						w.E4M3[i] = E4M3Bits(float32(r.NormFloat64() * 0.125))
					}
					all.E4M3 = append(all.E4M3, w.E4M3...)
				default:
					w.BF16 = randBF16(r, n*k, 0.05)
					all.BF16 = append(all.BF16, w.BF16...)
				}
				ws = append(ws, w)
			}
			var bufs []emuBuf
			if g.Weight == WeightE4M3 {
				bufs = append(bufs, emuBuf{"x", "uint8_t", xE4})
			} else {
				bufs = append(bufs, emuBuf{"x", "uint16_t", u16Bytes(x)})
			}
			args := []string{fmt.Sprint(m), "x"}
			if g.Gather {
				bufs = append(bufs, emuBuf{"rows", "int32_t", i32Bytes(rows)})
				args = append(args, "rows")
			}
			switch g.Weight {
			case WeightNVFP4:
				if g.Repacked {
					bufs = append(bufs, emuBuf{"w", "uint32_t", u32Bytes(repackedW)}, emuBuf{"w_sf", "uint8_t", repackedSF},
						emuBuf{"w_gs", "float", f32Bytes(gs)})
				} else {
					bufs = append(bufs, emuBuf{"w", "uint8_t", all.Packed}, emuBuf{"w_sf", "uint8_t", all.Scales},
						emuBuf{"w_gs", "float", f32Bytes(gs)})
				}
				args = append(args, "w", "w_sf", "w_gs")
			case WeightE4M3:
				bufs = append(bufs, emuBuf{"w", "uint8_t", all.E4M3})
				args = append(args, "w")
			default:
				bufs = append(bufs, emuBuf{"w", "uint16_t", u16Bytes(all.BF16)})
				args = append(args, "w")
			}
			maxRows := m
			if g.Grouped {
				bufs = append(bufs, emuBuf{"offsets", "int32_t", i32Bytes(tt.offsets)})
				args = append(args, "offsets")
				maxRows = 0
				for e := range experts {
					maxRows = max(maxRows, int(tt.offsets[e+1]-tt.offsets[e]))
				}
			}
			var extras [][]float32
			for _, e := range g.Extras {
				size := map[Broadcast]int{PerCol: nout, PerRow: m, Full: m * nout}[e.Broadcast]
				v := make([]float32, size)
				for i := range v {
					v[i] = e.DType.Round(float32(r.NormFloat64()))
				}
				extras = append(extras, v)
				bufs = append(bufs, emuBuf{e.Name, e.DType.cType(), typed{e.DType, f64s(v)}.bytes()})
				args = append(args, e.Name)
			}
			bufs = append(bufs, emuBuf{"out", g.Out.cType(), poison(m * nout * g.Out.Bits() / 8)})
			args = append(args, "out")
			if g.SumSq {
				bufs = append(bufs, emuBuf{"ss", "float", poison(4 * m * nout / 32)})
				args = append(args, "ss")
			}
			grid, block := g.Grid(maxRows, experts)
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %s)",
				grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, strings.Join(args, ", "))
			res := runEmu(t, src, launch, bufs...)
			var offsets []int32
			if g.Grouped {
				offsets = tt.offsets
			}
			want, err := g.Reference(m, x, rows, offsets, ws, extras)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(want))
			for i, v := range decode(g.Out, res["out"]) {
				got[i] = float32(v)
			}
			rel := 1e-5
			if g.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeTo(got, want, rel, 1e-5); !ok {
				t.Fatalf("out[%d] = %g, want %g (relL2 %g)", i, got[i], want[i], relL2(got, want))
			}
			if g.SumSq {
				g32 := g
				g32.Out = F32
				y, err := g32.Reference(m, x, rows, offsets, ws, extras)
				if err != nil {
					t.Fatal(err)
				}
				wantSS := make([]float64, m*nout/32)
				for i, v := range y {
					wantSS[i/32] += float64(v) * float64(v)
				}
				gotSS := decode(F32, res["ss"])
				for i := range wantSS {
					if d := math.Abs(gotSS[i] - wantSS[i]); d > 1e-4*wantSS[i]+1e-6 {
						t.Fatalf("ss[%d] = %g, want %g", i, gotSS[i], wantSS[i])
					}
				}
			}
		})
	}
}

func TestTileGEMMErrors(t *testing.T) {
	ok := TileGEMM{Name: "g", N: 128, K: 128, Weight: WeightBF16, Out: BF16}
	tests := []struct {
		name string
		edit func(*TileGEMM)
	}{
		{"name", func(g *TileGEMM) { g.Name = "1g" }},
		{"n", func(g *TileGEMM) { g.N = 96 }},
		{"k", func(g *TileGEMM) { g.K = 96 }},
		{"tile", func(g *TileGEMM) { g.BM = 32 }},
		{"stages", func(g *TileGEMM) { g.Stages = 5 }},
		{"hilo", func(g *TileGEMM) { g.Weight = WeightBF16HiLo }},
		{"out", func(g *TileGEMM) { g.Out = E4M3 }},
		{"arity", func(g *TileGEMM) { g.Epilogue = MustLambda("(add $0 $1)") }},
		{"extra_name", func(g *TileGEMM) { g.Extras = []Operand{{"rows", F32, PerCol}} }},
		{"extra_ss", func(g *TileGEMM) { g.Extras = []Operand{{"ss", F32, PerCol}} }},
		{"wgmma_bm", func(g *TileGEMM) { g.WGMMA, g.BM = true, 64 }},
		{"repacked_no_wgmma", func(g *TileGEMM) { g.Weight, g.Repacked = WeightNVFP4, true }},
		{"repacked_bf16", func(g *TileGEMM) { g.WGMMA, g.Repacked = true, true }},
		{"swizzle128b_no_wgmma", func(g *TileGEMM) { g.Swizzle128B = true }},
		{"groupm_neg", func(g *TileGEMM) { g.GroupM = -1 }},
		{"e4m3_no_wgmma", func(g *TileGEMM) { g.Weight = WeightE4M3 }},
		{"e4m3_swizzle128b_k64", func(g *TileGEMM) { g.Weight, g.WGMMA, g.Swizzle128B, g.K = WeightE4M3, true, true, 64 }},
	}
	if _, err := ok.CUDA(); err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		g := ok
		tt.edit(&g)
		if _, err := g.CUDA(); err == nil {
			t.Errorf("%s: no error", tt.name)
		}
	}
}

func TestTileGEMMGolden(t *testing.T) {
	for _, g := range []TileGEMM{
		// Prefill qkv_gate projection (moe8_prefill_layer).
		{Name: "moe8_prefill_qkv", N: 2304, K: 8192, Weight: WeightBF16, Out: BF16},
		{Name: "moe8_prefill_qkv_wgmma", N: 2304, K: 8192, Weight: WeightBF16, Out: BF16, WGMMA: true},
		// MoE gate/up for local experts, tokens gathered by expert, SwiGLU applied later.
		{Name: "moe8_moe_w13", N: 8192, K: 4096, Weight: WeightNVFP4, Out: BF16, Grouped: true, Gather: true,
			GLU: true, Epilogue: MustLambda("(mul (silu $0) $1)")},
	} {
		src, err := g.Source()
		if err != nil {
			t.Fatal(err)
		}
		checkGolden(t, g.Name+".cu", src)
	}
}

func ExampleTileGEMM() {
	g := TileGEMM{Name: "proj", N: 2304, K: 8192, Weight: WeightBF16, Out: BF16}
	grid, block := g.Grid(753, 1)
	fmt.Println(grid, block, g.SharedBytes())
	// Output: [18 6 1] [256 1 1] 110592
}

// TestTileGEMMGroupMIdentical verifies that GroupM > 1 produces bit-for-bit
// identical output and SumSq reductions to GroupM == 0 across ragged
// multi-tile grids for both mma.sync and WGMMA.
func TestTileGEMMGroupMIdentical(t *testing.T) {
	r := rand.New(rand.NewPCG(91, 92))
	const m, n, k = 300, 256, 64
	x := randBF16(r, m*k, 1)
	w := randBF16(r, 2*n*k, 0.05)
	offsets := []int32{0, 140, 300}
	for _, wgmma := range []bool{false, true} {
		bm := 64
		if wgmma {
			bm = 128
		}
		base := TileGEMM{
			Name:     "gm_base",
			N:        n,
			K:        k,
			Weight:   WeightBF16,
			Out:      F32,
			BM:       bm,
			BN:       64,
			WGMMA:    wgmma,
			GLU:      true,
			Grouped:  true,
			SumSq:    true,
			Epilogue: MustLambda("(mul (silu $0) $1)"),
		}
		run := func(groupM int) (out, ss []byte) {
			g := base
			g.GroupM = groupM
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			grid, block := g.Grid(160, 2)
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, x, w, offsets, out, ss)",
				grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, m)
			res := runEmu(t, src, launch,
				emuBuf{"x", "uint16_t", u16Bytes(x)},
				emuBuf{"w", "uint16_t", u16Bytes(w)},
				emuBuf{"offsets", "int32_t", i32Bytes(offsets)},
				emuBuf{"out", "float", poison(4 * m * (n / 2))},
				emuBuf{"ss", "float", poison(4 * m * (n / 2) / 32)})
			return res["out"], res["ss"]
		}
		wantOut, wantSS := run(0)
		for _, gm := range []int{2, 4, 8} {
			gotOut, gotSS := run(gm)
			if !bytes.Equal(gotOut, wantOut) {
				t.Fatalf("wgmma=%t groupM=%d: out bytes differ from groupM=0", wgmma, gm)
			}
			if !bytes.Equal(gotSS, wantSS) {
				t.Fatalf("wgmma=%t groupM=%d: ss bytes differ from groupM=0", wgmma, gm)
			}
		}
	}
}

// TestTileGEMMSwizzleGroupMAndE4M3Mutations verifies that mismatching the 128B
// swizzle between cp.async staging and the WGMMA descriptor, corrupting the
// GroupM tile decomposition, or corrupting WeightE4M3 k-step offsets fails the
// host emulator test.
func TestTileGEMMSwizzleGroupMAndE4M3Mutations(t *testing.T) {
	r := rand.New(rand.NewPCG(101, 102))

	// 1. Swizzle128B and GroupM mutations on BF16 WGMMA.
	{
		g := TileGEMM{
			Name:        "tile_mut_bf16",
			N:           192,
			K:           128,
			Weight:      WeightBF16,
			Out:         F32,
			BN:          64,
			WGMMA:       true,
			Swizzle128B: true,
			GroupM:      4,
			Stages:      2,
		}
		src, err := g.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		const m = 260
		x := randBF16(r, m*g.K, 1)
		w := GEMMWeights{BF16: randBF16(r, g.N*g.K, 0.05)}
		want, err := g.Reference(m, x, nil, nil, []GEMMWeights{w}, nil)
		if err != nil {
			t.Fatal(err)
		}
		grid, block := g.Grid(m, 1)
		launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, x, w, out)",
			grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, m)
		mutants := []struct {
			name string
			old  string
			repl string
		}{
			{
				name: "mismatch_store_unswizzled_desc_swizzled",
				old:  "auto core = [](int r, int ch) { return (int)eg_swizzle_128b((uint32_t)(r * 128 + ch * 16)); };",
				repl: "auto core = [](int r, int ch) { return r * 128 + ch * 16; };",
			},
			{
				name: "mismatch_store_swizzled_desc_unswizzled",
				old:  "eg_wgmma_desc_128b(sa + wgi * 8 * 1024 + ks * 32, 1024), eg_wgmma_desc_128b(sb + ks * 32, 1024)",
				repl: "eg_wgmma_desc(sa + wgi * 8 * 1024 + ks * 256, 128, 1024), eg_wgmma_desc(sb + ks * 256, 128, 1024)",
			},
			{
				name: "corrupt_groupm_tile_n",
				old:  "const int tile_n = (pid % in_group) / gsz;",
				repl: "const int tile_n = (pid % in_group) % NUM_N;",
			},
		}
		for _, mut := range mutants {
			if !strings.Contains(src, mut.old) {
				t.Fatalf("%s: target not in source", mut.name)
			}
			msrc := strings.Replace(src, mut.old, mut.repl, 1)
			res := runEmu(t, msrc, launch,
				emuBuf{"x", "uint16_t", u16Bytes(x)},
				emuBuf{"w", "uint16_t", u16Bytes(w.BF16)},
				emuBuf{"out", "float", poison(4 * m * g.N)})
			got := bytesF32(res["out"])
			if _, ok := closeTo(got, want, 1e-5, 1e-5); ok {
				t.Errorf("mutant %s survived", mut.name)
			}
		}
	}

	// 2. WeightE4M3 k-step offset mutation.
	{
		g := TileGEMM{
			Name:        "tile_mut_e4m3",
			N:           64,
			K:           128,
			Weight:      WeightE4M3,
			Out:         F32,
			BN:          64,
			WGMMA:       true,
			Swizzle128B: true,
			Stages:      2,
		}
		src, err := g.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		const m = 64
		xE4 := make([]uint8, m*g.K)
		x := make([]uint16, m*g.K)
		for i := range xE4 {
			b := E4M3Bits(float32(r.NormFloat64() * 0.5))
			xE4[i] = b
			x[i] = uint16(b)
		}
		w := GEMMWeights{E4M3: make([]uint8, g.N*g.K)}
		for i := range w.E4M3 {
			w.E4M3[i] = E4M3Bits(float32(r.NormFloat64() * 0.125))
		}
		want, err := g.Reference(m, x, nil, nil, []GEMMWeights{w}, nil)
		if err != nil {
			t.Fatal(err)
		}
		grid, block := g.Grid(m, 1)
		launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, x, w, out)",
			grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, m)
		old := "eg_wgmma_desc_128b(sa + wgi * 8 * 1024 + ks * 32, 1024), eg_wgmma_desc_128b(sb + ks * 32, 1024)"
		repl := "eg_wgmma_desc_128b(sa + wgi * 8 * 1024 + ks * 16, 1024), eg_wgmma_desc_128b(sb + ks * 16, 1024)"
		if !strings.Contains(src, old) {
			t.Fatal("e4m3 k-step target not in source")
		}
		msrc := strings.Replace(src, old, repl, 1)
		res := runEmu(t, msrc, launch,
			emuBuf{"x", "uint8_t", xE4},
			emuBuf{"w", "uint8_t", w.E4M3},
			emuBuf{"out", "float", poison(4 * m * g.N)})
		got := bytesF32(res["out"])
		if _, ok := closeTo(got, want, 1e-5, 1e-5); ok {
			t.Error("mutant corrupt_e4m3_k_step survived")
		}
	}
}
