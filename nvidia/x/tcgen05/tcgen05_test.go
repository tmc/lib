package tcgen05

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

type emuBuf struct {
	name  string
	ctype string
	data  []byte
}

func runEmu(t *testing.T, src, launch string, bufs ...emuBuf) map[string][]byte {
	t.Helper()
	cbufs := make([]cudahostemu.Buffer, len(bufs))
	for i, b := range bufs {
		cbufs[i] = cudahostemu.Buffer{Name: b.name, CType: b.ctype, Data: b.data}
	}
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  Prelude + src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		t.Fatalf("emulated kernel: %v\n--- source ---\n%s", err, Prelude+src)
	}
	return res
}

func poison(n int) []byte { return bytes.Repeat([]byte{0xff}, n) }

func f32Bytes(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}

func bytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func u16Bytes(v []uint16) []byte {
	b := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[2*i:], x)
	}
	return b
}

func randBF16(r *rand.Rand, n int, scale float64) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		out[i] = BF16Bits(float32(r.NormFloat64() * scale))
	}
	return out
}

func randNVFP4(r *rand.Rand, n, k int) GEMMWeights {
	w := GEMMWeights{
		Packed:     make([]uint8, n*(k/2)),
		Scales:     make([]uint8, n*(k/16)),
		GlobalScal: 1.0,
	}
	for i := range w.Packed {
		w.Packed[i] = uint8(r.IntN(256))
	}
	for i := range w.Scales {
		w.Scales[i] = E4M3Bits(float32(0.125 + r.Float64()*0.5))
	}
	return w
}

func decode(d DType, b []byte) []float64 {
	switch d {
	case F32:
		out := make([]float64, len(b)/4)
		for i := range out {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:])))
		}
		return out
	case BF16:
		out := make([]float64, len(b)/2)
		for i := range out {
			out[i] = float64(FromBF16Bits(binary.LittleEndian.Uint16(b[2*i:])))
		}
		return out
	case E4M3:
		out := make([]float64, len(b))
		for i, v := range b {
			out[i] = float64(FromE4M3Bits(v))
		}
		return out
	}
	return nil
}

func closeTo(got, want []float32, rel, abs float64) (int, bool) {
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) != math.IsNaN(w) {
			return i, false
		}
		if math.Abs(g-w) > abs+rel*math.Abs(w) {
			return i, false
		}
	}
	return -1, true
}

func relL2(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}

func firstDiff(d DType, got, want []byte) string {
	g, w := decode(d, got), decode(d, want)
	for i := range min(len(g), len(w)) {
		if g[i] != w[i] && !(math.IsNaN(g[i]) && math.IsNaN(w[i])) {
			return fmt.Sprintf("index %d: got %g, want %g", i, g[i], w[i])
		}
	}
	return "length mismatch"
}

func TestTCGen05GEMMSourcePTXInstructions(t *testing.T) {
	for _, wf := range []WeightFormat{WeightBF16, WeightE4M3, WeightNVFP4} {
		g := TCGen05GEMM{
			Name:     "tcgen05_" + wf.String(),
			N:        128,
			K:        64,
			BM:       128,
			BN:       128,
			BK:       64,
			Weight:   wf,
			Out:      F32,
			Epilogue: TileLinear,
		}
		src, err := g.Source()
		if err != nil {
			t.Fatalf("%v Source: %v", wf, err)
		}
		for _, want := range []string{
			"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
			"tcgen05.mma.cta_group::1",
			"tcgen05.commit",
			"tcgen05.wait::ld.sync.aligned",
			"tcgen05.ld.sync.aligned.32x32b",
			"tcgen05.dealloc.cta_group::1.sync.aligned.b32",
			"eg_emu_tmem_alloc",
			"eg_emu_tmem_dealloc",
			"eg_emu_tcgen05_commit",
			"eg_emu_tcgen05_wait",
			"eg_emu_tcgen05_ld_f32",
		} {
			if !strings.Contains(src, want) {
				t.Errorf("%v Source() missing %q", wf, want)
			}
		}
	}
}

func TestTCGen05GEMMEmulated(t *testing.T) {
	tests := []struct {
		name string
		g    TCGen05GEMM
		m    int
	}{
		{
			name: "bf16_linear_f32",
			g: TCGen05GEMM{
				Name:     "tcgen05_bf16_lin",
				N:        128,
				K:        128,
				BM:       128,
				BN:       128,
				BK:       64,
				Weight:   WeightBF16,
				Out:      F32,
				Epilogue: TileLinear,
			},
			m: 100,
		},
		{
			name: "bf16_silu_bf16_bn256",
			g: TCGen05GEMM{
				Name:     "tcgen05_bf16_silu",
				N:        256,
				K:        128,
				BM:       128,
				BN:       256,
				BK:       64,
				Weight:   WeightBF16,
				Out:      BF16,
				Epilogue: TileSiLU,
			},
			m: 70,
		},
		{
			name: "bf16_glu_f32",
			g: TCGen05GEMM{
				Name:     "tcgen05_bf16_glu",
				N:        128,
				K:        64,
				BM:       128,
				BN:       128,
				BK:       64,
				Weight:   WeightBF16,
				Out:      F32,
				Epilogue: TileGLU,
			},
			m: 64,
		},
		{
			name: "e4m3_linear_f32",
			g: TCGen05GEMM{
				Name:     "tcgen05_e4m3_lin",
				N:        128,
				K:        128,
				BM:       128,
				BN:       128,
				BK:       64,
				Weight:   WeightE4M3,
				Out:      F32,
				Epilogue: TileLinear,
			},
			m: 96,
		},
		{
			name: "nvfp4_linear_f32",
			g: TCGen05GEMM{
				Name:     "tcgen05_nvfp4_lin",
				N:        128,
				K:        128,
				BM:       128,
				BN:       128,
				BK:       64,
				Weight:   WeightNVFP4,
				Out:      F32,
				Epilogue: TileLinear,
			},
			m: 90,
		},
	}

	r := rand.New(rand.NewPCG(501, 502))
	ctx := context.Background()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := tt.g
			src, err := g.Source()
			if err != nil {
				t.Fatalf("Source: %v", err)
			}
			grid, block := g.Grid(tt.m)
			nout := g.N
			if g.Epilogue == TileGLU {
				nout = g.N / 2
			}
			outBytes := poison(tt.m * nout * g.Out.Bits() / 8)
			launch := cudahostemu.Launch{
				Kernel:      g.Name,
				Grid:        cudahostemu.Dim3{X: grid[0], Y: grid[1], Z: grid[2]},
				Block:       cudahostemu.Dim3{X: block[0], Y: block[1], Z: block[2]},
				SharedBytes: g.SharedBytes(),
			}

			var want []float32
			var args []cudahostemu.Arg
			args = append(args, cudahostemu.IntArg(tt.m))

			switch g.Weight {
			case WeightBF16:
				x := randBF16(r, tt.m*g.K, 0.5)
				w := GEMMWeights{BF16: randBF16(r, g.N*g.K, 0.25)}
				want, err = g.Reference(tt.m, x, w)
				if err != nil {
					t.Fatalf("Reference: %v", err)
				}
				args = append(args,
					cudahostemu.BufferArg("x", "uint16_t", u16Bytes(x)),
					cudahostemu.BufferArg("w", "uint16_t", u16Bytes(w.BF16)),
					cudahostemu.BufferArg("out", g.Out.cType(), outBytes),
				)
			case WeightE4M3:
				xE4 := make([]uint8, tt.m*g.K)
				xU16 := make([]uint16, tt.m*g.K)
				for i := range xE4 {
					b := E4M3Bits(float32(r.NormFloat64() * 0.5))
					xE4[i] = b
					xU16[i] = uint16(b)
				}
				w := GEMMWeights{E4M3: make([]uint8, g.N*g.K)}
				for i := range w.E4M3 {
					w.E4M3[i] = E4M3Bits(float32(r.NormFloat64() * 0.25))
				}
				want, err = g.Reference(tt.m, xU16, w)
				if err != nil {
					t.Fatalf("Reference: %v", err)
				}
				args = append(args,
					cudahostemu.BufferArg("x", "uint8_t", xE4),
					cudahostemu.BufferArg("w", "uint8_t", w.E4M3),
					cudahostemu.BufferArg("out", g.Out.cType(), outBytes),
				)
			case WeightNVFP4:
				aFP4 := make([]uint8, tt.m*(g.K/2))
				aScale := make([]uint8, tt.m*(g.K/16))
				bFP4 := make([]uint8, g.N*(g.K/2))
				bScale := make([]uint8, g.N*(g.K/16))
				for i := range aFP4 {
					aFP4[i] = uint8(r.IntN(256))
				}
				for i := range aScale {
					aScale[i] = E4M3Bits(float32(0.125 + r.Float64()*0.5))
				}
				for i := range bFP4 {
					bFP4[i] = uint8(r.IntN(256))
				}
				for i := range bScale {
					bScale[i] = E4M3Bits(float32(0.125 + r.Float64()*0.25))
				}
				want, err = g.ReferenceNVFP4(tt.m, aFP4, aScale, bFP4, bScale)
				if err != nil {
					t.Fatalf("ReferenceNVFP4: %v", err)
				}
				args = append(args,
					cudahostemu.BufferArg("a_fp4", "uint8_t", aFP4),
					cudahostemu.BufferArg("a_scale", "uint8_t", aScale),
					cudahostemu.BufferArg("b_fp4", "uint8_t", bFP4),
					cudahostemu.BufferArg("b_scale", "uint8_t", bScale),
					cudahostemu.BufferArg("out", g.Out.cType(), outBytes),
				)
			}

			err = cudahostemu.RunWithTMEM(ctx, src, launch, args...)
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("RunWithTMEM: %v", err)
			}

			got := make([]float32, len(want))
			for i, v := range decode(g.Out, outBytes) {
				got[i] = float32(v)
			}

			var ms float64
			for _, v := range want {
				ms += float64(v) * float64(v)
			}
			rel, abs := 1e-5, max(1e-5, 1e-6*math.Sqrt(float64(g.K)*ms/float64(len(want))))
			if g.Out == BF16 {
				rel = 1.0 / 128
				abs = max(abs, 1e-2)
			}
			if i, ok := closeTo(got, want, rel, abs); !ok {
				t.Fatalf("out[%d] = %g, want %g (relL2 %g)", i, got[i], want[i], relL2(got, want))
			}
		})
	}
}
