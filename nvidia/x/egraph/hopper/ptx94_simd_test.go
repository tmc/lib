package hopper

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

func bytesU32(b []byte) []uint32 {
	v := make([]uint32, len(b)/4)
	for i := range v {
		v[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return v
}

func TestSubBytePackCodecs(t *testing.T) {
	for _, pt := range []SubBytePackType{
		SubBytePackE4M3x4,
		SubBytePackE5M2x4,
		SubBytePackE3M2x4,
		SubBytePackE2M3x4,
		SubBytePackE2M1x4,
		SubBytePackE2M1P4x4,
	} {
		if !pt.Valid() {
			t.Fatalf("expected %q to be valid", pt)
		}
	}
	if SubBytePackType("bf16x2").Valid() {
		t.Fatal("expected bf16x2 to be invalid SubBytePackType")
	}

	// E4M3x4 round-trip and saturation to ±448.0.
	e4Exact := [4]float32{0.0, 1.5, -3.25, 448.0}
	packedE4 := PackE4M3x4(e4Exact)
	gotE4 := UnpackE4M3x4(packedE4)
	if gotE4 != e4Exact {
		t.Fatalf("UnpackE4M3x4(PackE4M3x4(%v)) = %v, want %v", e4Exact, gotE4, e4Exact)
	}

	e4Sat := [4]float32{1024.0, -999.0, float32(math.Inf(1)), float32(math.Inf(-1))}
	gotE4Sat := UnpackE4M3x4(PackE4M3x4(e4Sat))
	wantE4Sat := [4]float32{448.0, -448.0, 448.0, -448.0}
	if gotE4Sat != wantE4Sat {
		t.Fatalf("E4M3x4 saturation = %v, want %v", gotE4Sat, wantE4Sat)
	}

	// E2M1x4 round-trip across all 15 distinct finite E2M1 values and saturation to ±6.0.
	e2Cases := [][4]float32{
		{0.0, 0.5, 1.0, 1.5},
		{2.0, 3.0, 4.0, 6.0},
		{-0.5, -1.0, -1.5, -2.0},
		{-3.0, -4.0, -6.0, 0.0},
	}
	for _, tc := range e2Cases {
		p := PackE2M1x4(tc)
		if (p & 0xf0f0f0f0) != 0 {
			t.Fatalf("PackE2M1x4(%v) = 0x%08x has non-zero p4 padding bits", tc, p)
		}
		got := UnpackE2M1x4(p)
		if got != tc {
			t.Fatalf("UnpackE2M1x4(PackE2M1x4(%v)) = %v, want %v", tc, got, tc)
		}
	}

	e2Sat := [4]float32{12.0, -9.5, float32(math.Inf(1)), float32(math.Inf(-1))}
	gotE2Sat := UnpackE2M1x4(PackE2M1x4(e2Sat))
	wantE2Sat := [4]float32{6.0, -6.0, 6.0, -6.0}
	if gotE2Sat != wantE2Sat {
		t.Fatalf("E2M1x4 saturation = %v, want %v", gotE2Sat, wantE2Sat)
	}

	// Verify EvalQuadSubByteFMA across SubBytePackType formats.
	a := [4]float32{2.0, -1.5, 4.0, 6.0}
	b := [4]float32{1.5, 2.0, 128.0, -100.0}
	c := [4]float32{0.5, -1.0, 16.0, -50.0}
	// Expected unquantized FMA: [3.5, -4.0, 528.0 (saturates to 448), -650.0 (saturates to -448)]
	fmaE4 := UnpackE4M3x4(EvalQuadSubByteFMA(a, b, c, SubBytePackE4M3x4))
	wantFMAE4 := [4]float32{3.5, -4.0, 448.0, -448.0}
	if fmaE4 != wantFMAE4 {
		t.Fatalf("EvalQuadSubByteFMA(e4m3x4) = %v, want %v", fmaE4, wantFMAE4)
	}

	// Expected E2M1P4x4 FMA: 3.5 ties-to-even -> 4.0, -4.0 -> -4.0, 528 -> 6.0, -650 -> -6.0
	fmaE2 := UnpackE2M1x4(EvalQuadSubByteFMA(a, b, c, SubBytePackE2M1P4x4))
	wantFMAE2 := [4]float32{4.0, -4.0, 6.0, -6.0}
	if fmaE2 != wantFMAE2 {
		t.Fatalf("EvalQuadSubByteFMA(e2m1p4x4) = %v, want %v", fmaE2, wantFMAE2)
	}

	for _, pt := range []SubBytePackType{SubBytePackE5M2x4, SubBytePackE3M2x4, SubBytePackE2M3x4} {
		if u := EvalQuadSubByteFMA(a, b, c, pt); u == 0 {
			t.Fatalf("EvalQuadSubByteFMA(%s) returned 0", pt)
		}
	}
}

func TestRubinQuadSubByteSIMDEpilogue(t *testing.T) {
	// Target and configuration validation.
	for _, validTarget := range []string{"", "sm_107a", "sm_107f"} {
		k := RubinQuadSubByteSIMDEpilogue{NumQuads: 64, Target: validTarget}
		if err := k.Validate(); err != nil {
			t.Fatalf("Validate(%q) unexpected error: %v", validTarget, err)
		}
	}
	if err := (RubinQuadSubByteSIMDEpilogue{Target: "sm_90a"}).Validate(); err == nil {
		t.Fatal("expected Validate error for non-Rubin target sm_90a")
	}
	if err := (RubinQuadSubByteSIMDEpilogue{NumQuads: -1, Target: "sm_107f"}).Validate(); err == nil {
		t.Fatal("expected Validate error for negative NumQuads")
	}

	const numQuads = 64
	k := RubinQuadSubByteSIMDEpilogue{
		Name:     "rubin_quad_subbyte_simd_epilogue",
		NumQuads: numQuads,
		Target:   "sm_107f",
	}

	cudaSrc, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA: %v", err)
	}
	for _, marker := range []string{
		"#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070",
		"fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4",
		"add.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4",
		"mul.ftz.rz.f16x2.f32x2.f32x2",
	} {
		if !strings.Contains(cudaSrc, marker) {
			t.Fatalf("CUDA source missing PTX 9.4 marker %q", marker)
		}
	}

	launchSrc, err := k.Launcher()
	if err != nil {
		t.Fatalf("Launcher: %v", err)
	}
	if _, err := k.Source(); err != nil {
		t.Fatalf("Source: %v", err)
	}

	aE2M1x4 := make([]uint32, numQuads)
	bE4M3x4 := make([]uint32, numQuads)
	cE4M3x4 := make([]uint32, numQuads)
	e2Palette := [8]float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
	for q := range numQuads {
		var av, bv, cv [4]float32
		for i := range 4 {
			idx := 4*q + i
			av[i] = e2Palette[idx%len(e2Palette)]
			if (idx & 1) == 1 {
				av[i] = -av[i]
			}
			bv[i] = float32(math.Sin(float64(idx)*0.31+0.2)) * 12.0
			cv[i] = float32(math.Cos(float64(idx)*0.19-0.4)) * 6.0
		}
		// Include an overflow quad to exercise .satfinite saturation in FMA and .rz in F16x2.
		if q == numQuads-1 {
			av = [4]float32{6.0, -6.0, 4.0, -3.0}
			bv = [4]float32{128.0, 128.0, 64.0, 32.0}
			cv = [4]float32{64.0, -64.0, 256.0, -16.0}
		}
		aE2M1x4[q] = PackE2M1x4(av)
		bE4M3x4[q] = PackE4M3x4(bv)
		cE4M3x4[q] = PackE4M3x4(cv)
	}

	wantFMA, wantAdd, wantF32, err := k.Reference(aE2M1x4, bE4M3x4, cE4M3x4)
	if err != nil {
		t.Fatalf("Reference: %v", err)
	}

	src := cudaSrc + "\n" + launchSrc
	launch := fmt.Sprintf("%s_launch(a_e2m1x4, b_e4m3x4, c_e4m3x4, out_fma_e4m3x4, out_add_e4m3x4, out_f32, %d, nullptr)", k.Name, numQuads)
	res := runEmu(t, src, launch,
		emuBuf{"a_e2m1x4", "unsigned int", u32Bytes(aE2M1x4)},
		emuBuf{"b_e4m3x4", "unsigned int", u32Bytes(bE4M3x4)},
		emuBuf{"c_e4m3x4", "unsigned int", u32Bytes(cE4M3x4)},
		emuBuf{"out_fma_e4m3x4", "unsigned int", poison(4 * len(wantFMA))},
		emuBuf{"out_add_e4m3x4", "unsigned int", poison(4 * len(wantAdd))},
		emuBuf{"out_f32", "float", poison(4 * len(wantF32))},
	)

	gotFMA := bytesU32(res["out_fma_e4m3x4"])
	gotAdd := bytesU32(res["out_add_e4m3x4"])
	gotF32 := bytesF32(res["out_f32"])

	for q := range numQuads {
		if gotFMA[q] != wantFMA[q] {
			t.Fatalf("out_fma_e4m3x4[%d] = 0x%08x, want 0x%08x", q, gotFMA[q], wantFMA[q])
		}
		if gotAdd[q] != wantAdd[q] {
			t.Fatalf("out_add_e4m3x4[%d] = 0x%08x, want 0x%08x", q, gotAdd[q], wantAdd[q])
		}
	}
	for i := range wantF32 {
		if diff := math.Abs(float64(gotF32[i] - wantF32[i])); diff >= 1e-5 {
			t.Fatalf("out_f32[%d] = %v, want %v (diff %e)", i, gotF32[i], wantF32[i], diff)
		}
	}
}

func ExampleRubinQuadSubByteSIMDEpilogue() {
	k := RubinQuadSubByteSIMDEpilogue{
		NumQuads: 1,
		Target:   "sm_107f",
	}
	a := []uint32{PackE2M1x4([4]float32{2.0, -1.5, 4.0, 0.5})}
	b := []uint32{PackE4M3x4([4]float32{3.0, 2.0, 1.5, -4.0})}
	c := []uint32{PackE4M3x4([4]float32{1.0, 0.5, -2.0, 1.0})}

	outFMA, outAdd, outF32, err := k.Reference(a, b, c)
	if err != nil {
		panic(err)
	}
	fmaVals := UnpackE4M3x4(outFMA[0])
	addVals := UnpackE4M3x4(outAdd[0])
	fmt.Printf("target=%s fma0=%.1f add0=%.1f f32_0=%.1f\n",
		k.Target, fmaVals[0], addVals[0], outF32[0])
	// Output:
	// target=sm_107f fma0=7.0 add0=5.0 f32_0=35.0
}
