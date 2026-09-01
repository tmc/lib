//go:build darwin

package metalsafetensors

import (
	"math"
	"testing"
)

func TestF32ToF16RoundTrip(t *testing.T) {
	cases := []struct {
		in   float32
		want float32
	}{
		{0, 0},
		{1, 1},
		{-1, -1},
		{0.5, 0.5},
		{2.5, 2.5},
		{-0.09375, -0.09375},
		{65504, 65504},                            // max finite half
		{100000, float32(math.Inf(1))},            // overflow saturates
		{6.103515625e-05, 6.103515625e-05},        // min normal half
		{5.960464477539063e-08, 5.960464477539063e-08}, // min subnormal half
	}
	for _, tc := range cases {
		got := f16ToF32(f32BitsToF16(math.Float32bits(tc.in)))
		if got != tc.want {
			t.Errorf("f16 round trip of %g = %g, want %g", tc.in, got, tc.want)
		}
	}
}

func TestF32ToF16Rounding(t *testing.T) {
	// 1 + 2^-11 is exactly between two representable halves; round to
	// nearest even keeps 1.0.
	in := float32(1) + float32(math.Pow(2, -11))
	if got := f16ToF32(f32BitsToF16(math.Float32bits(in))); got != 1 {
		t.Errorf("round-to-nearest-even of 1+2^-11 = %g, want 1", got)
	}
	// 1 + 3*2^-11 rounds up to 1 + 2^-9... its nearest even neighbor.
	in = float32(1) + 3*float32(math.Pow(2, -11))
	want := float32(1) + 2*float32(math.Pow(2, -10))
	if got := f16ToF32(f32BitsToF16(math.Float32bits(in))); got != want {
		t.Errorf("round-to-nearest-even of 1+3*2^-11 = %g, want %g", got, want)
	}
}
