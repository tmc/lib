package hopper

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
)

func randBF16(r *rand.Rand, n int, scale float64) []uint16 {
	v := make([]uint16, n)
	for i := range v {
		v[i] = BF16Bits(float32(r.NormFloat64() * scale))
	}
	return v
}

func randNVFP4(r *rand.Rand, n, k int) GEMMWeights {
	w := GEMMWeights{Packed: make([]uint8, n*k/2), Scales: make([]uint8, n*k/16), GlobalScal: 0.0123}
	for i := range w.Packed {
		w.Packed[i] = uint8(r.IntN(256))
	}
	for i := range w.Scales {
		w.Scales[i] = E4M3Bits(float32(8 + r.Float64()*400))
	}
	return w
}

func TestSkinnyGEMMEmulated(t *testing.T) {
	tests := []struct {
		name string
		g    SkinnyGEMM
		m    int
	}{
		{"bf16", SkinnyGEMM{N: 8, K: 512, MaxM: 4, Weight: WeightBF16, Out: BF16}, 3},
		{"bf16_r2_u1", SkinnyGEMM{N: 8, K: 512, MaxM: 2, Weight: WeightBF16, Out: F32, RowsPerBlock: 2, Warps: 2, Unroll: 1}, 2},
		{"bf16_split", SkinnyGEMM{N: 4, K: 1024, MaxM: 8, Weight: WeightBF16, Out: BF16, SplitK: 2, RowsPerBlock: 2}, 8},
		{"hilo", SkinnyGEMM{N: 4, K: 256, MaxM: 2, Weight: WeightBF16HiLo, Out: F32}, 2},
		{"nvfp4", SkinnyGEMM{N: 4, K: 2048, MaxM: 2, Weight: WeightNVFP4, Out: F32, RowsPerBlock: 2}, 2},
		{"nvfp4_split", SkinnyGEMM{N: 2, K: 2048, MaxM: 1, Weight: WeightNVFP4, Out: BF16, SplitK: 2, Warps: 1}, 1},
		{"epilogue", SkinnyGEMM{N: 8, K: 256, MaxM: 2, Weight: WeightBF16, Out: BF16,
			Epilogue: MustLambda("(add (mul $0 $1) (silu $2))"),
			Extras:   []Operand{{"scale", BF16, PerCol}, {"resid", F32, Full}}}, 2},
	}
	r := rand.New(rand.NewPCG(1, 2))
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "gemm_" + tt.name
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			n, k, m := g.N, g.K, tt.m
			x := randBF16(r, m*k, 1)
			var w GEMMWeights
			bufs := []emuBuf{{"x", "uint16_t", u16Bytes(x)}}
			args := []string{"x"}
			switch g.Weight {
			case WeightBF16:
				w.BF16 = randBF16(r, n*k, 0.05)
				bufs = append(bufs, emuBuf{"w", "uint16_t", u16Bytes(w.BF16)})
				args = append(args, "w")
			case WeightBF16HiLo:
				w.BF16, w.Lo = make([]uint16, n*k), make([]uint16, n*k)
				for i := range w.BF16 {
					w.BF16[i], w.Lo[i] = SplitBF16(float32(r.NormFloat64() * 0.05))
				}
				bufs = append(bufs, emuBuf{"w_hi", "uint16_t", u16Bytes(w.BF16)}, emuBuf{"w_lo", "uint16_t", u16Bytes(w.Lo)})
				args = append(args, "w_hi", "w_lo")
			case WeightNVFP4:
				w = randNVFP4(r, n, k)
				bufs = append(bufs, emuBuf{"w", "uint8_t", w.Packed}, emuBuf{"w_sf", "uint8_t", w.Scales},
					emuBuf{"w_gs", "float", f32Bytes([]float32{w.GlobalScal})})
				args = append(args, "w", "w_sf", "w_gs")
			}
			var extras [][]float32
			for _, e := range g.Extras {
				size := n
				if e.Broadcast == Full {
					size = m * n
				}
				v := make([]float32, size)
				for i := range v {
					v[i] = e.DType.Round(float32(r.NormFloat64()))
				}
				extras = append(extras, v)
				if e.DType == BF16 {
					b := make([]uint16, size)
					for i, f := range v {
						b[i] = BF16Bits(f)
					}
					bufs = append(bufs, emuBuf{e.Name, "uint16_t", u16Bytes(b)})
				} else {
					bufs = append(bufs, emuBuf{e.Name, "float", f32Bytes(v)})
				}
				args = append(args, e.Name)
			}
			outBytes := m * n * g.Out.Bits() / 8
			bufs = append(bufs, emuBuf{"out", g.Out.cType(), poison(outBytes)})
			args = append(args, "out")
			if ws := g.WorkspaceBytes(); ws > 0 {
				bufs = append(bufs, emuBuf{"ws", "float", make([]byte, ws)})
				args = append(args, "ws")
			}
			grid, block, err := g.Grid()
			if err != nil {
				t.Fatal(err)
			}
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)", grid[0], grid[1], block[0], g.Name, m, strings.Join(args, ", "))
			if g.SplitK > 1 { // a second launch checks the counters were reset
				launch += "; " + launch
			}
			res := runEmu(t, src, launch, bufs...)
			want, err := g.Reference(m, x, w, extras)
			if err != nil {
				t.Fatal(err)
			}
			var got []float32
			if g.Out == BF16 {
				for _, b := range bytesU16(res["out"]) {
					got = append(got, FromBF16Bits(b))
				}
			} else {
				got = bytesF32(res["out"])
			}
			rel := 1e-5
			if g.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeTo(got, want, rel, 1e-6); !ok {
				t.Fatalf("out[%d] = %g, want %g (relL2 %g)", i, got[i], want[i], relL2(got, want))
			}
			if g.SplitK > 1 {
				counters := res["ws"][4*g.SplitK*g.MaxM*g.N:]
				for i, c := range counters {
					if c != 0 {
						t.Fatalf("counter byte %d = %d after launch, want 0", i, c)
					}
				}
			}
		})
	}
}

func TestSkinnyGEMMErrors(t *testing.T) {
	ok := SkinnyGEMM{Name: "g", N: 8, K: 256, MaxM: 1, Weight: WeightBF16, Out: BF16}
	for name, mutate := range map[string]func(*SkinnyGEMM){
		"name":     func(g *SkinnyGEMM) { g.Name = "eg_x" },
		"out":      func(g *SkinnyGEMM) { g.Out = E4M3 },
		"maxm":     func(g *SkinnyGEMM) { g.MaxM = 17 },
		"ragged_k": func(g *SkinnyGEMM) { g.K = 300 },
		"ragged_n": func(g *SkinnyGEMM) { g.RowsPerBlock = 3 },
		"epilogue": func(g *SkinnyGEMM) { g.Epilogue = MustLambda("(add $0 $1)") },
		"split":    func(g *SkinnyGEMM) { g.SplitK = 2 },
	} {
		g := ok
		mutate(&g)
		if _, err := g.CUDA(); err == nil {
			t.Errorf("%s: CUDA succeeded", name)
		}
	}
	if _, err := ok.CUDA(); err != nil {
		t.Fatal(err)
	}
}

// TestSkinnyGEMMGolden pins the Moe8 projection kernels' source.
func TestSkinnyGEMMGolden(t *testing.T) {
	for _, g := range []SkinnyGEMM{
		{Name: "moe8_qkv_gate", N: 2304, K: 8192, MaxM: 8, Weight: WeightBF16, Out: BF16},
		{Name: "moe8_router_hilo", N: 256, K: 8192, MaxM: 8, Weight: WeightBF16HiLo, Out: F32, SplitK: 4},
		{Name: "moe8_expert_down_nvfp4", N: 4096, K: 4096, MaxM: 4, Weight: WeightNVFP4, Out: F32},
	} {
		src, err := g.Source()
		if err != nil {
			t.Fatal(err)
		}
		checkGolden(t, g.Name+".cu", src)
	}
}

func ExampleSkinnyGEMM() {
	// moe8_bf16_proj's "o" layer: [M, 1024] x [8192, 1024]^T, BF16 out.
	g := SkinnyGEMM{Name: "o_proj", N: 8192, K: 1024, MaxM: 8, Weight: WeightBF16, Out: BF16}
	grid, block, _ := g.Grid()
	src, _ := g.Source()
	fmt.Println(grid, block, strings.Contains(src, "o_proj_launch(int m"))
	// Output:
	// [2048 1 1] [256 1 1] true
}
