package hopper

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"
)

// The two interleaved rows can have different block scales. Swapping them
// must break the output even when the packed codes themselves are unchanged.
func TestWarpSpecializedTwoWordScaleMutation(t *testing.T) {
	r := rand.New(rand.NewPCG(723, 724))
	g := TileGEMM{
		Name: "ws_two_word_scale", N: 128, K: 128, BM: 64, BN: 128,
		Weight: WeightNVFP4, Out: F32, WGMMA: true, Swizzle128B: true,
		Repacked: true, DequantWords: 2, WarpSpecialized: true, Stages: 2,
	}
	src, err := g.CUDA()
	if err != nil {
		t.Fatal(err)
	}
	const m = 17
	x := randBF16(r, m*g.K, 1)
	w := randNVFP4(r, g.N, g.K)
	w.GlobalScal = 1.0 / 64
	wq, wsf, err := RepackNVFP4Dequant2Word(w.Packed, w.Scales, g.N, g.K, false)
	if err != nil {
		t.Fatal(err)
	}
	want, err := g.Reference(m, x, nil, nil, []GEMMWeights{w}, nil)
	if err != nil {
		t.Fatal(err)
	}
	grid, block := g.Grid(m, 1)
	launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, x, w, w_sf, w_gs, out)",
		grid[0], grid[1], grid[2], block[0], g.SharedBytes(), g.Name, m)
	const old = "decoded[4 * sub + j], (j & 1) ? s1 : s0"
	const mutant = "decoded[4 * sub + j], (j & 1) ? s0 : s1"
	if !strings.Contains(src, old) {
		t.Fatal("scale selection not found")
	}
	for _, tt := range []struct {
		name string
		src  string
		want bool
	}{
		{"original", src, true},
		{"swapped", strings.Replace(src, old, mutant, 1), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res := runEmu(t, tt.src, launch,
				emuBuf{"x", "uint16_t", u16Bytes(x)},
				emuBuf{"w", "uint32_t", u32Bytes(wq)},
				emuBuf{"w_sf", "uint8_t", wsf},
				emuBuf{"w_gs", "float", f32Bytes([]float32{w.GlobalScal})},
				emuBuf{"out", "float", poison(4 * m * g.N)})
			_, ok := closeTo(bytesF32(res["out"]), want, 1e-5, 1e-5)
			if ok != tt.want {
				t.Fatalf("reference match = %v, want %v", ok, tt.want)
			}
		})
	}
}
