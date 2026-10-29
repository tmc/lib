package hopper

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestSkinnyMMAStagingEmulated(t *testing.T) {
	tests := []struct {
		name string
		g    SkinnyMMA
		m    int
	}{
		{"plain", SkinnyMMA{N: 64, K: 512, MaxM: 32, Out: F32, Warps: 2, SplitK: 1}, 21},
		{"short_k", SkinnyMMA{N: 16, K: 96, MaxM: 16, Out: BF16, Warps: 1, Chunks: 1, Steps: 1, SplitK: 1}, 1},
		{"glu", SkinnyMMA{N: 64, K: 512, MaxM: 32, Out: BF16, Warps: 1, Tiles: 2, GLU: true, SplitK: 1, Epilogue: MustLambda("(mul (silu $0) $1)")}, 21},
		{"split_residual", SkinnyMMA{N: 32, K: 384, MaxM: 32, Out: F32, Warps: 1, Chunks: 1, Steps: 1, SplitK: 3, Epilogue: MustLambda("(add $0 $1)"), Extras: []Operand{{"resid", F32, Full}}}, 17},
		{"glu_split_bias", SkinnyMMA{N: 64, K: 384, MaxM: 32, Out: F32, Warps: 1, Tiles: 2, Chunks: 1, Steps: 1, SplitK: 3, GLU: true, Epilogue: MustLambda("(add (mul (silu $0) $1) $2)"), Extras: []Operand{{"bias", BF16, PerCol}}}, 30},
	}
	r := rand.New(rand.NewPCG(23, 29))
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			g.Name = "stage_" + tt.name
			nout := g.N
			if g.GLU {
				nout /= 2
			}
			x := randBF16(r, tt.m*g.K, 1)
			w := randBF16(r, g.N*g.K, .05)
			bufs := []emuBuf{{"x", "uint16_t", u16Bytes(x)}, {"w", "uint16_t", u16Bytes(w)}}
			args := []string{fmt.Sprint(tt.m), "x", "w"}
			var extra [][]float32
			for _, e := range g.Extras {
				n := nout
				if e.Broadcast == Full {
					n *= tt.m
				}
				v := make([]float32, n)
				for i := range v {
					v[i] = e.DType.Round(float32(r.NormFloat64()))
				}
				extra = append(extra, v)
				bufs = append(bufs, emuBuf{e.Name, e.DType.cType(), typed{e.DType, f64s(v)}.bytes()})
				args = append(args, e.Name)
			}
			// Include inactive rows to detect an output store beyond m_total.
			bufs = append(bufs, emuBuf{"out", g.Out.cType(), poison(g.MaxM * nout * g.Out.Bits() / 8)})
			args = append(args, "out")
			if ws := g.WorkspaceBytes(); ws > 0 {
				bufs = append(bufs, emuBuf{"ws", "float", make([]byte, ws)})
				args = append(args, "ws")
			}
			var results [2][]byte
			for i, stage := range []bool{false, true} {
				g.StageWeights = stage
				src, err := g.CUDA()
				if err != nil {
					t.Fatal(err)
				}
				grid, block, err := g.Grid()
				if err != nil {
					t.Fatal(err)
				}
				call := fmt.Sprintf("eg_emu_launch(dim3(%d,%d),dim3(%d),%d,%s,%s)", grid[0], grid[1], block[0], g.SharedBytes(), g.Name, strings.Join(args, ","))
				if g.WorkspaceBytes() > 0 {
					call += "; " + call
				}
				out := runEmu(t, src, call, bufs...)
				results[i] = out["out"]
				if g.WorkspaceBytes() > 0 {
					ng, _ := g.norm()
					for j, v := range out["ws"][4*ng.SplitK*g.MaxM*g.N:] {
						if v != 0 {
							t.Fatalf("stage=%v counter byte%d=%d", stage, j, v)
						}
					}
				}
			}
			if !bytes.Equal(results[0], results[1]) {
				t.Fatal("staging changed output bits or inactive rows")
			}
			want, err := g.Reference(tt.m, x, w, extra)
			if err != nil {
				t.Fatal(err)
			}
			vals := decode(g.Out, results[1])
			got := make([]float32, len(want))
			for i := range got {
				got[i] = float32(vals[i])
			}
			rel := 1e-5
			if g.Out == BF16 {
				rel = 1.0 / 128
			}
			if i, ok := closeTo(got, want, rel, 1e-5); !ok {
				t.Fatalf("out[%d]=%g want%g", i, got[i], want[i])
			}
		})
	}
}

func TestSkinnyMMAStagingResources(t *testing.T) {
	for _, glu := range []bool{false, true} {
		g := SkinnyMMA{Name: "stage_resources", N: 4096, K: 4096, MaxM: 32, Out: F32, SMs: 48, GLU: glu, StageWeights: true}
		ng, err := g.norm()
		if err != nil {
			t.Fatal(err)
		}
		if ng.KChunk != 256 || g.SharedBytes() != 36864 {
			t.Fatalf("glu=%v KC=%d shared=%d", glu, ng.KChunk, g.SharedBytes())
		}
		g.StageWeights = false
		if g.SharedBytes() != 34816 {
			t.Fatalf("default shared=%d", g.SharedBytes())
		}
	}
	tests := []struct {
		name string
		edit func(*SkinnyMMA)
	}{
		{"split_overflow", func(g *SkinnyMMA) { g.SplitK = int(^uint(0) >> 1) }},
		{"large_chunk", func(g *SkinnyMMA) { g.KChunk = 512 }},
		{"large_step", func(g *SkinnyMMA) { g.Chunks = 9 }},
		{"nondividing_chunk", func(g *SkinnyMMA) { g.KChunk = 96 }},
		{"unaligned_n", func(g *SkinnyMMA) { g.N = 33 }},
		{"unaligned_k", func(g *SkinnyMMA) { g.K = 320 }},
		{"row_overflow", func(g *SkinnyMMA) { g.Tiles = int(^uint(0) >> 1) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := SkinnyMMA{Name: "invalid_stage", N: 64, K: 512, MaxM: 32, Out: F32, StageWeights: true}
			tt.edit(&g)
			if _, err := g.CUDA(); err == nil {
				t.Fatal("accepted invalid staging schedule")
			}
		})
	}
}
