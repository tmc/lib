package mxfp4

import (
	"math"
	"strings"
	"testing"
)

// decodeE2M1Spec decodes a 4-bit code from the OCP E2M1 bit layout -- one sign
// bit, two exponent bits with a bias of 1, one mantissa bit -- without
// consulting E2M1Table. It is the independent oracle the table is checked
// against; a round trip through Quantize and Dequantize cannot catch a wrong
// table, because both sides read the same one.
func decodeE2M1Spec(code uint8) float64 {
	sign := 1.0
	if code&0x8 != 0 {
		sign = -1.0
	}
	exp := int((code >> 1) & 0x3)
	mant := float64(code & 0x1)
	if exp == 0 {
		// Subnormal: 2^(1-bias) * (mant/2), bias = 1.
		return sign * mant * 0.5
	}
	return sign * (1 + mant*0.5) * math.Pow(2, float64(exp-1))
}

func TestE2M1TableMatchesBitLayout(t *testing.T) {
	for code := 0; code < 16; code++ {
		want := decodeE2M1Spec(uint8(code))
		got := float64(E2M1Table[code])
		if got != want {
			t.Errorf("E2M1Table[%d] = %v, spec decode = %v", code, got, want)
		}
	}
	// Code 8 is negative zero in the spec, but Go's -0.0 constant is positive
	// zero, so the table holds +0 there. The deviation is inert -- the value is
	// only ever multiplied by a scale, and -0 and +0 compare and arithmetise
	// alike -- so it is pinned rather than corrected.
	if math.Signbit(float64(E2M1Table[8])) {
		t.Error("E2M1Table[8] is negative zero; update this test, the deviation is gone")
	}
}

// TestQuantizeKnownNibbles pins the encoding against nibbles computed by hand,
// not against Dequantize. Group max is 3.0, so scale = 3.0/6 = 0.5. Then
// 3.0/0.5 = 6.0 -> code 7, -1.5/0.5 = -3.0 -> code 8|5 = 13, and 0 -> code 0.
func TestQuantizeKnownNibbles(t *testing.T) {
	src := make([]float32, 32)
	src[0], src[1] = 3.0, -1.5

	packed, scales, err := Quantize(src, 32)
	if err != nil {
		t.Fatal(err)
	}
	if scales[0] != 0.5 {
		t.Errorf("scale = %v, want 0.5", scales[0])
	}
	if got, want := packed[0], byte(7|(13<<4)); got != want {
		t.Errorf("packed[0] = %#02x, want %#02x", got, want)
	}
	for i, b := range packed[1:] {
		if b != 0 {
			t.Errorf("packed[%d] = %#02x, want 0", i+1, b)
		}
	}
}

// TestQuantizeExactlyRepresentableIsLossless is an oracle test rather than a
// round trip: the inputs are chosen so that every one of them lands exactly on
// a grid point, so the expected output is the input itself, known in advance
// and not derived from the code under test.
func TestQuantizeExactlyRepresentableIsLossless(t *testing.T) {
	// Grid values scaled by 2, so maxAbs = 12 and scale = 2 exactly.
	grid := []float32{0, 1, 2, 3, 4, 6, 8, 12, -1, -2, -3, -4, -6, -8, -12, 0}
	src := make([]float32, 32)
	copy(src, grid)
	copy(src[16:], grid)

	packed, scales, err := Quantize(src, 32)
	if err != nil {
		t.Fatal(err)
	}
	if scales[0] != 2 {
		t.Fatalf("scale = %v, want 2", scales[0])
	}
	dst := make([]float32, 32)
	if err := Dequantize(packed, scales, 32, dst); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if dst[i] != src[i] {
			t.Errorf("dst[%d] = %v, want %v (exactly representable)", i, dst[i], src[i])
		}
	}
}

func TestQuantizeRejectsNonFinite(t *testing.T) {
	tests := []struct {
		name  string
		index int
		value float32
	}{
		{"positive infinity", 5, float32(math.Inf(1))},
		{"negative infinity", 0, float32(math.Inf(-1))},
		{"NaN", 31, float32(math.NaN())},
		{"NaN in the second group", 40, float32(math.NaN())},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := make([]float32, 64)
			for i := range src {
				src[i] = 1.0
			}
			src[tt.index] = tt.value

			packed, scales, err := Quantize(src, 32)
			if err == nil {
				t.Fatalf("Quantize returned nil error for %v at index %d", tt.value, tt.index)
			}
			if packed != nil || scales != nil {
				t.Error("Quantize returned output alongside an error")
			}
			if !strings.Contains(err.Error(), "src[") {
				t.Errorf("error does not name the offending element: %v", err)
			}
		})
	}
}

// TestQuantizeNonFiniteWouldPoisonGroup records the failure mode the check
// exists to prevent: without it, one infinity makes the scale infinite and all
// 32 elements dequantize to NaN, and a NaN quantizes to zero -- both with a
// nil error. The test drives the corruption directly through Dequantize.
func TestQuantizeNonFiniteWouldPoisonGroup(t *testing.T) {
	packed := make([]byte, 16)
	scales := []float32{float32(math.Inf(1))}
	dst := make([]float32, 32)
	if err := Dequantize(packed, scales, 32, dst); err == nil {
		t.Fatal("Dequantize accepted an infinite scale")
	}
	for i, v := range dst {
		if math.IsNaN(float64(v)) {
			t.Errorf("dst[%d] is NaN: an infinite scale was propagated", i)
			break
		}
	}
}

func TestDequantizeRejectsNonFiniteScale(t *testing.T) {
	for _, s := range []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())} {
		dst := make([]float32, 32)
		if err := Dequantize(make([]byte, 16), []float32{s}, 32, dst); err == nil {
			t.Errorf("Dequantize accepted scale %v", s)
		}
	}
}

// TestQuantizeE2M1TieBreak pins the rounding rule against an independently
// computed table. quantizeE2M1 uses a strict "dist < bestDist" comparison, so
// every midpoint rounds toward the lower code, i.e. toward zero -- not
// round-half-to-even. The two rules disagree at 0.75, 1.75 and 3.5, where RNE
// rounds up to the value with an even significand.
func TestQuantizeE2M1TieBreak(t *testing.T) {
	for _, tt := range []struct {
		in      float32
		want    uint8 // ties toward zero, as implemented
		wantRNE uint8 // round-half-to-even, for contrast
	}{
		{0.25, 0, 0},
		{0.75, 1, 2},
		{1.25, 2, 2},
		{1.75, 3, 4},
		{2.5, 4, 4},
		{3.5, 5, 6},
		{5.0, 6, 6},
	} {
		got := quantizeE2M1(tt.in)
		if got != tt.want {
			t.Errorf("quantizeE2M1(%v) = %d (%v), want %d (%v)", tt.in, got, E2M1Table[got], tt.want, E2M1Table[tt.want])
		}
		if tt.want != tt.wantRNE && got == tt.wantRNE {
			t.Errorf("quantizeE2M1(%v) now rounds half to even; update this test", tt.in)
		}
	}
}
