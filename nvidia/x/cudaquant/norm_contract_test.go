package cudaquant

import (
	"math"
	"testing"
)

func TestNormQuantKnownQ81Layout(t *testing.T) {
	input := make([]float32, 32)
	for i := range input {
		input[i] = 1
	}
	blocks, err := RMSNormQ8_1(input, nil, 1, 32, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	b := blocks[0]
	// IEEE binary16 nearest-even: d=1/127, s=d*sum(q)=32.
	if b.D != 0x2008 || b.S != 0x5000 {
		t.Fatalf("d/s=%04x/%04x", b.D, b.S)
	}
	for i, q := range b.Quants {
		if q != 127 {
			t.Fatalf("q[%d]=%d", i, q)
		}
	}
}

func TestNormQuantInvalidInputs(t *testing.T) {
	for _, tc := range []struct {
		name       string
		rows, cols int
		eps        float32
		value      float32
	}{
		{"overflow", math.MaxInt, 32, 1, 0},
		{"infinite epsilon", 1, 32, float32(math.Inf(1)), 0},
		{"nan input", 1, 32, 1, float32(math.NaN())},
		{"zero denominator", 1, 32, 0, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float32, 32)
			input[0] = tc.value
			if _, err := RMSNormQ8_1(input, nil, tc.rows, tc.cols, tc.eps, false); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}

func TestBinary16Rounding(t *testing.T) {
	for _, tc := range []struct {
		value float32
		bits  uint16
	}{
		{0, 0}, {1, 0x3c00}, {-2, 0xc000}, {65504, 0x7bff},
		{float32(math.Ldexp(1, -24)), 1}, {1 + 1.0/2048, 0x3c00}, {1 + 3.0/2048, 0x3c02},
	} {
		if got := float32ToFloat16(tc.value); got != tc.bits {
			t.Errorf("%g: got %04x want %04x", tc.value, got, tc.bits)
		}
	}
}
