package tcgen05

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func TestUE8M0AndFP6Packing(t *testing.T) {
	// 1. Verify EncodeUE8M0 and DecodeUE8M0 across powers of two, non-powers, and NaN.
	if got := EncodeUE8M0(1.0); got != 127 {
		t.Fatalf("EncodeUE8M0(1.0) = %d, want 127", got)
	}
	if got := DecodeUE8M0(127); got != 1.0 {
		t.Fatalf("DecodeUE8M0(127) = %v, want 1.0", got)
	}
	if got := EncodeUE8M0(float32(math.NaN())); got != 0xFF {
		t.Fatalf("EncodeUE8M0(NaN) = 0x%x, want 0xFF", got)
	}
	if !math.IsNaN(float64(DecodeUE8M0(0xFF))) {
		t.Fatalf("DecodeUE8M0(0xFF) = %v, want NaN", DecodeUE8M0(0xFF))
	}
	for e := -20; e <= 20; e++ {
		wantScale := float32(math.Ldexp(1.0, e))
		code := EncodeUE8M0(wantScale)
		if int(code) != e+127 {
			t.Fatalf("EncodeUE8M0(2^%d) = %d, want %d", e, code, e+127)
		}
		if got := DecodeUE8M0(code); got != wantScale {
			t.Fatalf("DecodeUE8M0(%d) = %v, want %v", code, got, wantScale)
		}
		// Verify smallest power-of-two >= scale for intermediate value 1.5 * 2^(e-1).
		mid := float32(math.Ldexp(1.5, e-1))
		if gotCode := EncodeUE8M0(mid); int(gotCode) != e+127 {
			t.Fatalf("EncodeUE8M0(%v) = %d, want %d", mid, gotCode, e+127)
		}
	}

	// 2. Verify QuantizeMXFP8E4M3 and DequantizeMXFP8E4M3 on 32-element blocks.
	src := make([]float32, 64)
	for i := range 32 {
		src[i] = float32((i%8)-4) * 2.0 // exact E4M3 values at scale = 1.0
	}
	for i := 32; i < 64; i++ {
		src[i] = float32((i%8)-4) * 16.0 // scaled block
	}
	codes, scales := QuantizeMXFP8E4M3(src)
	if len(codes) != 64 || len(scales) != 2 {
		t.Fatalf("QuantizeMXFP8E4M3 lengths = (%d, %d), want (64, 2)", len(codes), len(scales))
	}
	deq := DequantizeMXFP8E4M3(codes, scales)
	for i, want := range src {
		if deq[i] != want {
			t.Fatalf("DequantizeMXFP8E4M3[%d] = %v, want %v", i, deq[i], want)
		}
	}

	// 3. Verify EncodeFP6E3M2 / DecodeFP6E3M2 and EncodeFP6E2M3 / DecodeFP6E2M3 across all 64 codes.
	if got := DecodeFP6E3M2(31); got != 28.0 {
		t.Fatalf("DecodeFP6E3M2(31) = %v, want 28.0", got)
	}
	if got := DecodeFP6E3M2(63); got != -28.0 {
		t.Fatalf("DecodeFP6E3M2(63) = %v, want -28.0", got)
	}
	if got := DecodeFP6E2M3(31); got != 7.5 {
		t.Fatalf("DecodeFP6E2M3(31) = %v, want 7.5", got)
	}
	for code := uint8(0); code < 64; code++ {
		v32 := DecodeFP6E3M2(code)
		if got := EncodeFP6E3M2(v32); got != code {
			t.Fatalf("EncodeFP6E3M2(DecodeFP6E3M2(%d=%v)) = %d, want %d", code, v32, got, code)
		}
		v23 := DecodeFP6E2M3(code)
		if got := EncodeFP6E2M3(v23); got != code {
			t.Fatalf("EncodeFP6E2M3(DecodeFP6E2M3(%d=%v)) = %d, want %d", code, v23, got, code)
		}
	}

	// 4. Verify PackFP6x4 and UnpackFP6x4 round-trip across all 6-bit combinations.
	rng := rand.New(rand.NewPCG(11, 22))
	for trial := range 512 {
		c0 := uint8((trial + rng.IntN(64)) & 0x3f)
		c1 := uint8((trial*3 + rng.IntN(64)) & 0x3f)
		c2 := uint8((trial*7 + rng.IntN(64)) & 0x3f)
		c3 := uint8((trial*13 + rng.IntN(64)) & 0x3f)
		packed := PackFP6x4(c0, c1, c2, c3)
		wantB0 := c0 | (c1 << 6)
		wantB1 := (c1 >> 2) | (c2 << 4)
		wantB2 := (c2 >> 4) | (c3 << 2)
		if packed != [3]uint8{wantB0, wantB1, wantB2} {
			t.Fatalf("PackFP6x4(%d,%d,%d,%d) = %v, want [%d %d %d]",
				c0, c1, c2, c3, packed, wantB0, wantB1, wantB2)
		}
		unpacked := UnpackFP6x4(packed)
		if unpacked != [4]uint8{c0, c1, c2, c3} {
			t.Fatalf("UnpackFP6x4(%v) = %v, want [%d %d %d %d]",
				packed, unpacked, c0, c1, c2, c3)
		}
	}
}

func TestTCGen05MXFP8AndW4A8GEMM(t *testing.T) {
	cases := []struct {
		name         string
		mixedW4A8    bool
		useShiftDown bool
	}{
		{name: "pure_mxfp8", mixedW4A8: false, useShiftDown: false},
		{name: "mixed_w4a8", mixedW4A8: true, useShiftDown: false},
		{name: "mxfp8_shift_down", mixedW4A8: false, useShiftDown: true},
		{name: "w4a8_shift_down", mixedW4A8: true, useShiftDown: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			k := TCGen05MXFP8GEMM{
				Name:         "test_" + tc.name,
				M:            128,
				N:            128,
				K:            64,
				MixedW4A8:    tc.mixedW4A8,
				UseShiftDown: tc.useShiftDown,
			}

			for _, target := range []cudatmem.Target{cudatmem.SM100A, cudatmem.SM103A} {
				sum, err := cudatmem.VerifyTrace(target, k.TraceEvents())
				if err != nil {
					t.Fatalf("VerifyTrace(%s) failed: %v", target, err)
				}
				wantShifts := 0
				if tc.useShiftDown {
					wantShifts = 1
				}
				if sum.CTAs != 1 || sum.CTAGroup != 1 || sum.MaxAllocatedColumns != 160 || sum.Copies != 1 || sum.MMAs != 1 || sum.Shifts != wantShifts || sum.Loads != 4 {
					t.Fatalf("VerifyTrace(%s) unexpected summary: %+v", target, sum)
				}
			}

			src, err := k.CUDA()
			if err != nil {
				t.Fatalf("CUDA() failed: %v", err)
			}
			for _, want := range []string{
				"tcgen05.cp.cta_group::1.32x128b.warpx4",
				"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X",
				"tcgen05.ld.sync.aligned.32x32b.x1.b32",
				"elect.sync",
			} {
				if !strings.Contains(src, want) {
					t.Fatalf("CUDA() missing %q", want)
				}
			}
			if tc.useShiftDown && !strings.Contains(src, "tcgen05.shift.cta_group::1.down") {
				t.Fatalf("CUDA() with UseShiftDown=true missing tcgen05.shift.cta_group::1.down")
			}

			rng := rand.New(rand.NewPCG(101, 202))
			aF32 := make([]float32, k.M*k.K)
			bF32 := make([]float32, k.N*k.K)
			for i := range aF32 {
				aF32[i] = float32(rng.IntN(7)-3) * 0.5
			}
			for i := range bF32 {
				bF32[i] = float32(rng.IntN(7)-3) * 0.5
			}

			aData, aScales := QuantizeMXFP8E4M3(aF32)
			var bData, bScales []uint8
			if tc.mixedW4A8 {
				bData, bScales = QuantizeMXFP4E2M1(bF32)
			} else {
				bData, bScales = QuantizeMXFP8E4M3(bF32)
			}

			want, err := k.Reference(aData, aScales, bData, bScales)
			if err != nil {
				t.Fatalf("Reference() failed: %v", err)
			}

			grid, block := k.Grid()
			launch := fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d, %d, %d), 0, %s, a_data, a_scales, b_data, b_scales, out)",
				grid[0], grid[1], grid[2], block[0], block[1], block[2], k.Name)
			res := runEmu(t, src, launch,
				emuBuf{"a_data", "uint8_t", aData},
				emuBuf{"a_scales", "uint8_t", aScales},
				emuBuf{"b_data", "uint8_t", bData},
				emuBuf{"b_scales", "uint8_t", bScales},
				emuBuf{"out", "float", poison(4 * k.M * k.N)},
			)
			got := bytesF32(res["out"])

			var maxDiff float64
			for i := range want {
				diff := math.Abs(float64(got[i] - want[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
				if math.IsNaN(float64(got[i])) || diff != 0 {
					t.Fatalf("out[%d] = %g, want %g (diff %g)", i, got[i], want[i], diff)
				}
			}
			if maxDiff != 0 {
				t.Fatalf("maxDiff = %g, want 0", maxDiff)
			}
		})
	}
}

func ExampleTCGen05MXFP8GEMM() {
	k := TCGen05MXFP8GEMM{
		Name:         "blackwell_w4a8_shift_gemm",
		M:            128,
		N:            128,
		K:            64,
		MixedW4A8:    true,
		UseShiftDown: true,
	}
	idesc, _ := k.InstDescriptor()
	dec := DecodeInstDesc(idesc)
	sum, err := cudatmem.VerifyTrace(cudatmem.SM100A, k.TraceEvents())
	src, _ := k.CUDA()
	fmt.Println(
		err == nil,
		dec.AType,
		dec.BType,
		dec.ScaleVec,
		sum.MMAs,
		sum.Shifts,
		strings.Contains(src, "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X"),
		strings.Contains(src, "tcgen05.shift.cta_group::1.down"),
	)
	// Output:
	// true e4m3 e2m1 1X 1 1 true true
}
