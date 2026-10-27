package hopper

import (
	"bytes"
	"fmt"
	"math"
	"strings"
	"testing"
)

func TestRubinUE5M3CodecAndPZO(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		want uint8
		dec  float32
	}{
		{"pos_zero", 0.0, 0x00, 0.0},
		{"neg_zero_clamped", float32(math.Copysign(0, -1)), 0x00, 0.0},
		{"negative_clamped", -42.5, 0x00, 0.0},
		{"neg_inf_clamped", float32(math.Inf(-1)), 0x00, 0.0},
		{"min_subnormal", float32(math.Ldexp(1.0, -17)), 0x01, float32(math.Ldexp(1.0, -17))},
		{"max_subnormal", float32(math.Ldexp(7.0, -17)), 0x07, float32(math.Ldexp(7.0, -17))},
		{"min_normal_2m14", float32(math.Ldexp(1.0, -14)), 0x08, float32(math.Ldexp(1.0, -14))},
		{"one", 1.0, 0x78, 1.0},
		{"one_point_five", 1.5, 0x7c, 1.5},
		{"two", 2.0, 0x80, 2.0},
		{"sixteen", 16.0, 0x98, 16.0},
		{"max_normal_61440", 61440.0, 0xf7, 61440.0},
		{"overflow_saturates", 100000.0, 0xf7, 61440.0},
		{"pos_inf_saturates", float32(math.Inf(1)), 0xf7, 61440.0},
		{"nan_saturates", float32(math.NaN()), 0xf7, 61440.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ToUE5M3Bits(tc.in)
			if got != tc.want {
				t.Fatalf("ToUE5M3Bits(%v) = 0x%02x, want 0x%02x", tc.in, got, tc.want)
			}
			dec := FromUE5M3Bits(got)
			if dec != tc.dec {
				t.Fatalf("FromUE5M3Bits(0x%02x) = %v, want %v", got, dec, tc.dec)
			}
		})
	}

	// Verify round-trip across all 248 finite UE5M3 codes (0x00..0xf7) and NaN for 0xf8..0xff.
	for code := range 256 {
		b := uint8(code)
		v := FromUE5M3Bits(b)
		if b >= 0xf8 {
			if !math.IsNaN(float64(v)) {
				t.Fatalf("FromUE5M3Bits(0x%02x) = %v, want NaN", b, v)
			}
			continue
		}
		if rt := ToUE5M3Bits(v); rt != b {
			t.Fatalf("round-trip UE5M3 0x%02x -> %v -> 0x%02x", b, v, rt)
		}
	}

	// Verify PackUE5M3x2 / UnpackUE5M3x2.
	packed := PackUE5M3x2(1.5, 61440.0)
	lo, hi := UnpackUE5M3x2(packed)
	if lo != 1.5 || hi != 61440.0 {
		t.Fatalf("UnpackUE5M3x2(0x%04x) = (%v, %v), want (1.5, 61440.0)", packed, lo, hi)
	}

	// Verify ApplyPZO8 (.pzo Positive-Zero-Only).
	if got := ApplyPZO8(0x80, 0x80); got != 0x00 {
		t.Fatalf("ApplyPZO8(-0.0 E4M3 0x80, 0x80) = 0x%02x, want 0x00", got)
	}
	if got := ApplyPZO8(0xb8, 0x80); got != 0xb8 {
		t.Fatalf("ApplyPZO8(-1.0 E4M3 0xb8, 0x80) = 0x%02x, want 0xb8", got)
	}
	if got := ApplyPZO8(0x08, 0x08); got != 0x00 {
		t.Fatalf("ApplyPZO8(-0.0 E2M1 0x08, 0x08) = 0x%02x, want 0x00", got)
	}
	if got := ApplyPZO8(0x0a, 0x08); got != 0x0a {
		t.Fatalf("ApplyPZO8(-1.0 E2M1 0x0a, 0x08) = 0x%02x, want 0x0a", got)
	}

	// Verify CvtScaledN1UE8M0ToUE5M3x2 and CvtScaledN2UE8M0FromUE5M3x2.
	scaleCode := ToUE8M0(2.0) // 2^1 -> code 128
	qPair := CvtScaledN1UE8M0ToUE5M3x2(6.0, 3.0, scaleCode)
	dqLo, dqHi := CvtScaledN2UE8M0FromUE5M3x2(qPair, uint16(scaleCode)|(uint16(scaleCode)<<8))
	if dqLo != 3.0 || dqHi != 6.0 {
		t.Fatalf("scaled N1/N2 round-trip = (%v, %v), want (3.0, 6.0)", dqLo, dqHi)
	}
}

func TestRubinUE5M3ScaledQuant(t *testing.T) {
	const (
		rows = 4
		cols = 64
		eps  = 1e-5
	)
	x := make([]float32, rows*cols)
	w := make([]float32, cols)
	for i := range x {
		// Positive and negative activations to exercise UE5M3 clamping and scaling.
		v := float32(math.Sin(float64(i)*0.23+0.4))*3.5 + 2.0
		if i%11 == 0 {
			v = -0.5
		}
		x[i] = v
	}
	for i := range w {
		w[i] = 0.75 + 0.5*float32(math.Cos(float64(i)*0.13))
	}

	k := RubinUE5M3ScaledQuant{
		Name:   "rubin_ue5m3_quant_test",
		Rows:   rows,
		Cols:   cols,
		Eps:    eps,
		UsePZO: true,
		Target: "sm_107f",
	}

	cudaSrc, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA: %v", err)
	}
	for _, marker := range []string{
		"cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32",
		"cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2",
		".pzo",
	} {
		if !strings.Contains(cudaSrc, marker) {
			t.Fatalf("CUDA source missing PTX marker %q", marker)
		}
	}

	launchSrc, err := k.Launcher()
	if err != nil {
		t.Fatalf("Launcher: %v", err)
	}
	if _, err := k.Source(); err != nil {
		t.Fatalf("Source: %v", err)
	}

	wantQuant, wantScales, wantDequant, err := k.Reference(x, w)
	if err != nil {
		t.Fatalf("Reference: %v", err)
	}

	src := cudaSrc + "\n" + launchSrc
	launch := fmt.Sprintf("%s_launch(x, w, out_ue5m3, out_scales, out_dequant, nullptr)", k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"x", "float", f32Bytes(x)},
		emuBuf{"w", "float", f32Bytes(w)},
		emuBuf{"out_ue5m3", "uint8_t", poison(len(wantQuant))},
		emuBuf{"out_scales", "uint8_t", poison(len(wantScales))},
		emuBuf{"out_dequant", "float", poison(4 * len(wantDequant))},
	)

	gotQuant := res["out_ue5m3"]
	gotScales := res["out_scales"]
	gotDequant := bytesF32(res["out_dequant"])

	if !bytes.Equal(gotQuant, wantQuant) {
		t.Fatalf("out_ue5m3 mismatch:\ngot  %x\nwant %x", gotQuant, wantQuant)
	}
	if !bytes.Equal(gotScales, wantScales) {
		t.Fatalf("out_scales mismatch:\ngot  %x\nwant %x", gotScales, wantScales)
	}
	for i := range wantDequant {
		diff := math.Abs(float64(gotDequant[i] - wantDequant[i]))
		if diff >= 1e-5 {
			t.Fatalf("out_dequant[%d] = %v, want %v (diff %e)", i, gotDequant[i], wantDequant[i], diff)
		}
	}
}

func TestRubinPackedF32x2ResidualRMSNorm(t *testing.T) {
	const (
		rows = 4
		cols = 64
		eps  = 1e-5
	)
	x := make([]float32, rows*cols)
	residual := make([]float32, rows*cols)
	weight := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.19+0.2)) * 1.8
		residual[i] = float32(math.Cos(float64(i)*0.15-0.3)) * 1.1
	}
	for i := range weight {
		weight[i] = 0.6 + 0.4*float32(math.Sin(float64(i)*0.11+0.5))
	}

	k := RubinPackedF32x2ResidualRMSNorm{
		Name:   "rubin_f32x2_res_rmsnorm_test",
		Rows:   rows,
		Cols:   cols,
		Eps:    eps,
		Target: "sm_107f",
	}

	cudaSrc, err := k.CUDA()
	if err != nil {
		t.Fatalf("CUDA: %v", err)
	}
	for _, marker := range []string{
		"add.rn.f32x2",
		"fma.rn.f32x2",
	} {
		if !strings.Contains(cudaSrc, marker) {
			t.Fatalf("CUDA source missing PTX marker %q", marker)
		}
	}

	launchSrc, err := k.Launcher()
	if err != nil {
		t.Fatalf("Launcher: %v", err)
	}
	if _, err := k.Source(); err != nil {
		t.Fatalf("Source: %v", err)
	}

	wantR, wantY, err := k.Reference(x, residual, weight)
	if err != nil {
		t.Fatalf("Reference: %v", err)
	}

	src := cudaSrc + "\n" + launchSrc
	launch := fmt.Sprintf("%s_launch(x, residual, weight, r_out, y_out, nullptr)", k.Name)
	res := runEmu(t, src, launch,
		emuBuf{"x", "float", f32Bytes(x)},
		emuBuf{"residual", "float", f32Bytes(residual)},
		emuBuf{"weight", "float", f32Bytes(weight)},
		emuBuf{"r_out", "float", poison(4 * len(wantR))},
		emuBuf{"y_out", "float", poison(4 * len(wantY))},
	)

	gotR := bytesF32(res["r_out"])
	gotY := bytesF32(res["y_out"])
	for i := range wantR {
		if diff := math.Abs(float64(gotR[i] - wantR[i])); diff >= 1e-5 {
			t.Fatalf("r_out[%d] = %v, want %v (diff %e)", i, gotR[i], wantR[i], diff)
		}
		if diff := math.Abs(float64(gotY[i] - wantY[i])); diff >= 1e-5 {
			t.Fatalf("y_out[%d] = %v, want %v (diff %e)", i, gotY[i], wantY[i], diff)
		}
	}
}

func ExampleRubinUE5M3ScaledQuant() {
	k := RubinUE5M3ScaledQuant{
		Name:   "rubin_ue5m3_demo",
		Rows:   1,
		Cols:   32,
		Eps:    1e-5,
		UsePZO: true,
		Target: "sm_107f",
	}
	x := make([]float32, 32)
	w := make([]float32, 32)
	for i := range 32 {
		x[i] = 1.0
		w[i] = 1.5
	}
	quant, scales, dequant, err := k.Reference(x, w)
	if err != nil {
		panic(err)
	}
	fmt.Printf("target=%s scales=%d q0=0x%02x dq0=%.2f max_norm=%.1f\n",
		k.Target, len(scales), quant[0], dequant[0], FromUE5M3Bits(0xf7))
	// Output:
	// target=sm_107f scales=1 q0=0xf4 dq0=1.50 max_norm=61440.0
}
