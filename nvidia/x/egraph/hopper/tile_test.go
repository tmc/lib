package hopper

import (
	"fmt"
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
		{"grouped_bf16", TileGEMM{N: 128, K: 64, Weight: WeightBF16, Out: F32, BM: 64, BN: 128, Grouped: true}, 75, 0, []int32{0, 64, 65, 75}},
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
			x := randBF16(r, xrows*k, 1)
			experts := 1
			if g.Grouped {
				experts = len(tt.offsets) - 1
			}
			var ws []GEMMWeights
			var all GEMMWeights
			var gs []float32
			for range experts {
				var w GEMMWeights
				if g.Weight == WeightNVFP4 {
					w = randNVFP4(r, n, k)
					w.GlobalScal = float32(0.01 + r.Float64()*0.01)
					all.Packed = append(all.Packed, w.Packed...)
					all.Scales = append(all.Scales, w.Scales...)
					gs = append(gs, w.GlobalScal)
				} else {
					w.BF16 = randBF16(r, n*k, 0.05)
					all.BF16 = append(all.BF16, w.BF16...)
				}
				ws = append(ws, w)
			}
			bufs := []emuBuf{{"x", "uint16_t", u16Bytes(x)}}
			args := []string{fmt.Sprint(m), "x"}
			if g.Gather {
				bufs = append(bufs, emuBuf{"rows", "int32_t", i32Bytes(rows)})
				args = append(args, "rows")
			}
			if g.Weight == WeightNVFP4 {
				bufs = append(bufs, emuBuf{"w", "uint8_t", all.Packed}, emuBuf{"w_sf", "uint8_t", all.Scales},
					emuBuf{"w_gs", "float", f32Bytes(gs)})
				args = append(args, "w", "w_sf", "w_gs")
			} else {
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
				size := map[Broadcast]int{PerCol: n, PerRow: m, Full: m * n}[e.Broadcast]
				v := make([]float32, size)
				for i := range v {
					v[i] = e.DType.Round(float32(r.NormFloat64()))
				}
				extras = append(extras, v)
				bufs = append(bufs, emuBuf{e.Name, e.DType.cType(), typed{e.DType, f64s(v)}.bytes()})
				args = append(args, e.Name)
			}
			bufs = append(bufs, emuBuf{"out", g.Out.cType(), make([]byte, m*n*g.Out.Bits()/8)})
			args = append(args, "out")
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
				t.Fatalf("out[%d] (row %d, col %d) = %g, want %g (relL2 %g)", i, i/n, i%n, got[i], want[i], relL2(got, want))
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
		// MoE gate/up for local experts, tokens gathered by expert, SwiGLU applied later.
		{Name: "moe8_moe_w13", N: 8192, K: 4096, Weight: WeightNVFP4, Out: BF16, Grouped: true, Gather: true},
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
