package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestSkinnyMMAEmulated(t *testing.T) {
	tests := []struct {
		name string
		g    SkinnyMMA
		m    int
	}{
		{"f32", SkinnyMMA{N: 64, K: 256, MaxM: 32, Out: F32, Warps: 2, Chunks: 1, SplitK: 1}, 21},
		{"one_tile", SkinnyMMA{N: 16, K: 128, MaxM: 16, Out: BF16, Warps: 1, Tiles: 1, Chunks: 2, Steps: 1, SplitK: 1}, 16},
		{"chunks", SkinnyMMA{N: 64, K: 512, MaxM: 32, Out: F32, Warps: 2, Chunks: 1, KChunk: 128, SplitK: 1}, 32},
		{"split", SkinnyMMA{N: 64, K: 512, MaxM: 16, Out: BF16, Warps: 2, Chunks: 1, KChunk: 64, SplitK: 2}, 5},
		{"split_chosen", SkinnyMMA{N: 256, K: 1024, MaxM: 32, Out: F32, SMs: 2}, 17},
		{"residual", SkinnyMMA{N: 64, K: 256, MaxM: 32, Out: F32, Warps: 2, Chunks: 1, SplitK: 2,
			Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{{"resid", F32, Full}}}, 21},
		{"glu", SkinnyMMA{N: 128, K: 256, MaxM: 32, Out: BF16, Warps: 2, Tiles: 4, Chunks: 1, SplitK: 1, GLU: true,
			Epilogue: MustLambda("(mul (silu $0) $1)")}, 21},
		{"glu_split_bias", SkinnyMMA{N: 64, K: 512, MaxM: 32, Out: F32, Warps: 1, Tiles: 4, Chunks: 2, KChunk: 128, SplitK: 2, GLU: true,
			Epilogue: MustLambda("(add (mul (silu $0) $1) $2)"), Extras: []Operand{{"bias", BF16, PerCol}}}, 30},
	}
	r := rand.New(rand.NewPCG(5, 6))
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "skinny_mma_" + tt.name
			src, err := g.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			n, k, m := g.N, g.K, tt.m
			nout := n
			if g.GLU {
				nout = n / 2
			}
			x := randBF16(r, m*k, 1)
			w := randBF16(r, n*k, 0.05)
			bufs := []emuBuf{{"x", "uint16_t", u16Bytes(x)}, {"w", "uint16_t", u16Bytes(w)}}
			args := []string{fmt.Sprint(m), "x", "w"}
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
			ws := g.WorkspaceBytes()
			if ws > 0 {
				bufs = append(bufs, emuBuf{"ws", "float", make([]byte, ws)})
				args = append(args, "ws")
			}
			grid, block, err := g.Grid()
			if err != nil {
				t.Fatal(err)
			}
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), %d, %s, %s)",
				grid[0], grid[1], block[0], g.SharedBytes(), g.Name, strings.Join(args, ", "))
			if ws > 0 { // a second launch checks the counters were reset
				launch += "; " + launch
			}
			res := runEmu(t, src, launch, bufs...)
			want, err := g.Reference(m, x, w, extras)
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, len(want))
			for i, v := range decode(g.Out, res["out"]) {
				got[i] = float32(v)
			}
			var ms float64
			for _, v := range want {
				ms += float64(v) * float64(v)
			}
			rel, abs := 1e-5, max(1e-5, 1e-6*math.Sqrt(float64(k)*ms/float64(len(want))))
			if g.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeTo(got, want, rel, abs); !ok {
				t.Fatalf("out[%d] = %g, want %g (relL2 %g)", i, got[i], want[i], relL2(got, want))
			}
			if ws > 0 {
				nn, _ := g.norm()
				for i, c := range res["ws"][4*nn.SplitK*g.MaxM*n:] {
					if c != 0 {
						t.Fatalf("counter byte %d = %d after launch, want 0", i, c)
					}
				}
			}
		})
	}
}

func TestSkinnyMMASchedule(t *testing.T) {
	tests := []struct {
		name                 string
		g                    SkinnyMMA
		grid                 [3]int
		kchunk, smem, wsSize int
	}{
		// Llama-3-8B prefill projections on GB10's 48 SMs need no split.
		{"qkv", SkinnyMMA{N: 6144, K: 4096, MaxM: 32, Out: BF16, SMs: 48}, [3]int{192, 1, 1}, 512, 32 * 544 * 2, 0},
		{"gate_up", SkinnyMMA{N: 28672, K: 4096, MaxM: 32, Out: F32, SMs: 48}, [3]int{896, 1, 1}, 512, 32 * 544 * 2, 0},
		{"gate_up_glu", SkinnyMMA{N: 28672, K: 4096, MaxM: 32, Out: BF16, SMs: 48, GLU: true}, [3]int{448, 1, 1}, 512, 32 * 544 * 2, 0},
		{"down", SkinnyMMA{N: 4096, K: 14336, MaxM: 32, Out: F32, SMs: 48}, [3]int{128, 1, 1}, 512, 32 * 544 * 2, 0},
		// On 132 SMs they split K: o_proj 3 ways would not divide K.
		{"o_proj_132", SkinnyMMA{N: 4096, K: 4096, MaxM: 32, Out: F32}, [3]int{128, 4, 1}, 512, 32 * 544 * 2, 4*4*32*4096 + 4*128},
		{"down_132", SkinnyMMA{N: 4096, K: 14336, MaxM: 16, Out: F32}, [3]int{128, 4, 1}, 512, 16 * 544 * 2, 4*4*16*4096 + 4*128},
		{"kchunk", SkinnyMMA{N: 4096, K: 4096, MaxM: 32, Out: F32, SplitK: 16}, [3]int{128, 16, 1}, 256, 32 * 288 * 2, 4*16*32*4096 + 4*128},
	}
	for _, tt := range tests {
		tt.g.Name = tt.name
		grid, _, err := tt.g.Grid()
		if err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		n, _ := tt.g.norm()
		if grid != tt.grid || n.KChunk != tt.kchunk || tt.g.SharedBytes() != tt.smem || tt.g.WorkspaceBytes() != tt.wsSize {
			t.Errorf("%s: grid %v, KChunk %d, %d shared bytes, %d workspace bytes; want %v, %d, %d, %d",
				tt.name, grid, n.KChunk, tt.g.SharedBytes(), tt.g.WorkspaceBytes(), tt.grid, tt.kchunk, tt.smem, tt.wsSize)
		}
	}
}

func TestSkinnyMMAErrors(t *testing.T) {
	ok := SkinnyMMA{Name: "g", N: 128, K: 256, MaxM: 32, Out: BF16}
	tests := []struct {
		name string
		edit func(*SkinnyMMA)
	}{
		{"name", func(g *SkinnyMMA) { g.Name = "eg_g" }},
		{"out", func(g *SkinnyMMA) { g.Out = E4M3 }},
		{"maxm", func(g *SkinnyMMA) { g.MaxM = 33 }},
		{"n", func(g *SkinnyMMA) { g.N = 100 }},
		{"k", func(g *SkinnyMMA) { g.K = 320 }},
		{"split", func(g *SkinnyMMA) { g.SplitK = 3 }},
		{"kchunk", func(g *SkinnyMMA) { g.KChunk = 96 }},
		{"glu_tiles", func(g *SkinnyMMA) { g.GLU, g.Tiles = true, 3 }},
		{"arity", func(g *SkinnyMMA) { g.Epilogue = MustLambda("(add $0 $1)") }},
		{"extra_name", func(g *SkinnyMMA) { g.Extras = []Operand{{"ws", F32, PerCol}} }},
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

// TestSkinnyMMAGolden pins the Llama-3-8B prefill projections' source.
func TestSkinnyMMAGolden(t *testing.T) {
	for _, g := range []SkinnyMMA{
		{Name: "llama_pf_skinny_o_proj", N: 4096, K: 4096, MaxM: 32, Out: F32, SMs: 48,
			Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{{"e1", F32, Full}}},
		{Name: "llama_pf_skinny_gate_up", N: 28672, K: 4096, MaxM: 32, Out: BF16, SMs: 48, GLU: true,
			Epilogue: MustLambda("(mul (silu $0) $1)")},
	} {
		src, err := g.Source()
		if err != nil {
			t.Fatal(err)
		}
		checkGolden(t, g.Name+".cu", src)
	}
}

// TestPlanUseSkinnyMMA lowers a decoder layer's MLP over 21 of 32 rows
// with tile gemms and with skinny mmas, runs both plans emulated, and
// compares them.
func TestPlanUseSkinnyMMA(t *testing.T) {
	const m, rows, h, inter = 21, 32, 256, 512
	b := NewBuilder()
	x := b.Input("x", F32, rows, h)
	norm := b.Cast(BF16, b.RMSNorm(x, b.Input("norm_w", F32, h), h, 1e-5))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", BF16, 2*inter, h)))
	out := b.Map(MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", BF16, h, inter)), x)
	e, err := Optimize(b.G, out)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(7, 8))
	data := map[string][]byte{}
	var outs [2][]float64
	for i, use := range []bool{false, true} {
		p, err := Compile(e, "mlp", F32)
		if err != nil {
			t.Fatal(err)
		}
		gemms := 0
		if use {
			p.UseSkinnyMMA(1)
			for _, k := range p.Kernels {
				if k.Tile != nil {
					t.Errorf("%s: tile gemm left after UseSkinnyMMA", k.Tile.Name)
				}
				if k.MMA != nil {
					gemms++
				}
			}
			if gemms != 2 {
				t.Errorf("UseSkinnyMMA made %d skinny mmas, want 2", gemms)
			}
		}
		src, err := p.Source()
		if err != nil {
			t.Fatal(err)
		}
		var bufs []emuBuf
		var args []string
		for _, in := range p.Inputs {
			if data[in.Name] == nil {
				n := size(in.Shape)
				switch {
				case in.DType == BF16:
					data[in.Name] = u16Bytes(randBF16(r, n, 0.05))
				default:
					v := make([]float32, n)
					for j := range v {
						v[j] = float32(r.NormFloat64())
					}
					data[in.Name] = f32Bytes(v)
				}
			}
			bufs = append(bufs, emuBuf{in.Name, in.DType.cType(), data[in.Name]})
			args = append(args, in.Name)
		}
		for _, tmp := range append(p.Temps, p.Output) {
			bufs = append(bufs, emuBuf{tmp.Name, tmp.DType.cType(), poison(size(tmp.Shape) * tmp.DType.Bits() / 8)})
			args = append(args, tmp.Name)
		}
		// Run m of the plan's rows, as a prefill chunk does.
		src = strings.ReplaceAll(src, fmt.Sprintf("_launch(%d, ", rows), fmt.Sprintf("_launch(%d, ", m))
		res := runEmu(t, src, fmt.Sprintf("mlp_run(%s, nullptr)", strings.Join(args, ", ")), bufs...)
		outs[i] = decode(F32, res[p.Output.Name])[:m*h]
	}
	got, want := make([]float32, m*h), make([]float32, m*h)
	for i := range got {
		got[i], want[i] = float32(outs[1][i]), float32(outs[0][i])
	}
	if i, ok := closeTo(got, want, 1e-3, 1e-4); !ok {
		t.Fatalf("out[%d] = %g with skinny mmas, %g with tile gemms (relL2 %g)", i, got[i], want[i], relL2(got, want))
	}
}

func ExampleSkinnyMMA() {
	// Llama-3-8B's down projection over a 21-token prompt on 48 SMs.
	g := SkinnyMMA{Name: "down", N: 4096, K: 14336, MaxM: 32, Out: F32, SMs: 48}
	grid, block, _ := g.Grid()
	fmt.Println(grid, block, g.SharedBytes())
	// Output: [128 1 1] [128 1 1] 34816
}
