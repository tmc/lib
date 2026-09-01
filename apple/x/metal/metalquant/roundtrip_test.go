// SPDX-License-Identifier: Apache-2.0

package metalquant

import (
	"math"
	"testing"
)

// TestFloat16KnownValues checks the fp16 codec against hand-computed bit
// patterns, including the subnormal and infinity edges.
func TestFloat16KnownValues(t *testing.T) {
	tests := []struct {
		name string
		bits uint16
		want float32
	}{
		{"zero", 0x0000, 0},
		{"negative zero", 0x8000, float32(math.Copysign(0, -1))},
		{"one", 0x3C00, 1},
		{"negative two", 0xC000, -2},
		{"half", 0x3800, 0.5},
		{"largest normal", 0x7BFF, 65504},
		{"smallest normal", 0x0400, 6.103515625e-05},
		{"largest subnormal", 0x03FF, 6.097555160522461e-05},
		{"smallest subnormal", 0x0001, 5.960464477539063e-08},
		{"infinity", 0x7C00, float32(math.Inf(1))},
		{"negative infinity", 0xFC00, float32(math.Inf(-1))},
	}
	for _, tt := range tests {
		got := float16ToFloat32(tt.bits)
		if got != tt.want {
			t.Errorf("%s: float16ToFloat32(%#04x) = %v, want %v", tt.name, tt.bits, got, tt.want)
		}
		if back := float32ToFloat16(got); back != tt.bits {
			t.Errorf("%s: float32ToFloat16(%v) = %#04x, want %#04x", tt.name, got, back, tt.bits)
		}
	}

	if nan := float16ToFloat32(0x7E00); !math.IsNaN(float64(nan)) {
		t.Errorf("float16ToFloat32(0x7E00) = %v, want NaN", nan)
	}
	if bits := float32ToFloat16(float32(math.NaN())); bits&0x7C00 != 0x7C00 || bits&0x03FF == 0 {
		t.Errorf("float32ToFloat16(NaN) = %#04x, want a NaN pattern", bits)
	}
}

// TestFloat16ExhaustiveRoundTrip walks every bit pattern: decoding to float32
// and re-encoding must return the same pattern for everything except NaNs,
// whose payload is not preserved.
func TestFloat16ExhaustiveRoundTrip(t *testing.T) {
	for i := 0; i <= 0xFFFF; i++ {
		bits := uint16(i)
		f := float16ToFloat32(bits)
		if math.IsNaN(float64(f)) {
			continue
		}
		if got := float32ToFloat16(f); got != bits {
			t.Fatalf("%#04x decoded to %v and re-encoded to %#04x", bits, f, got)
		}
	}
}

// TestFloat32ToFloat16Rounding checks that values between two representable
// halves round to nearest, ties to even, and that out-of-range magnitudes
// saturate rather than wrapping.
func TestFloat32ToFloat16Rounding(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		want uint16
	}{
		{"exact", 1, 0x3C00},
		{"just above one rounds down", 1 + 1.0/4096, 0x3C00}, // tie, 0x3C00 is even
		{"three quarters of an ulp rounds up", 1 + 3.0/4096, 0x3C01},
		{"one plus one ulp", 1 + 1.0/1024, 0x3C01},
		{"tie to even rounds up to even", 1 + 3.0/2048, 0x3C02},
		{"overflow saturates", 1e30, 0x7C00},
		{"negative overflow saturates", -1e30, 0xFC00},
		{"underflow to zero", 1e-30, 0x0000},
		{"negative underflow keeps sign", -1e-30, 0x8000},
		{"subnormal", 5.960464477539063e-08, 0x0001},
	}
	for _, tt := range tests {
		if got := float32ToFloat16(tt.in); got != tt.want {
			t.Errorf("%s: float32ToFloat16(%v) = %#04x, want %#04x", tt.name, tt.in, got, tt.want)
		}
	}
}

func testValues(n int) []float32 {
	// A deterministic spread of magnitudes and both signs.
	vals := make([]float32, n)
	for i := range vals {
		x := float32(i%37) - 18
		vals[i] = x * float32(math.Pow(1.7, float64(i%11))) / 64
	}
	return vals
}

// TestQ4_0RoundTripIdempotent is the inverse property that matters for a lossy
// codec: dequantizing and re-quantizing must reproduce the exact same bytes,
// so a weight file survives repeated conversion unchanged.
func TestQ4_0RoundTripIdempotent(t *testing.T) {
	vals := testValues(4 * BlockSize)

	packed, err := QuantizeQ4_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ4_0: %v", err)
	}
	if want := 4 * blockBytesQ4_0; len(packed) != want {
		t.Fatalf("packed length = %d, want %d", len(packed), want)
	}

	round1, err := DequantizeQ4_0(packed, len(vals))
	if err != nil {
		t.Fatalf("DequantizeQ4_0: %v", err)
	}
	repacked, err := QuantizeQ4_0(round1)
	if err != nil {
		t.Fatalf("QuantizeQ4_0 (second pass): %v", err)
	}
	for i := range packed {
		if packed[i] != repacked[i] {
			t.Fatalf("byte %d changed on re-quantization: %#02x -> %#02x", i, packed[i], repacked[i])
		}
	}

	round2, err := DequantizeQ4_0(repacked, len(vals))
	if err != nil {
		t.Fatalf("DequantizeQ4_0 (second pass): %v", err)
	}
	for i := range round1 {
		if round1[i] != round2[i] {
			t.Errorf("value %d changed on the second round trip: %v -> %v", i, round1[i], round2[i])
		}
	}
}

// TestQ4_0RoundTripError bounds the quantization error at half a step of the
// block scale, per block rather than globally.
func TestQ4_0RoundTripError(t *testing.T) {
	vals := testValues(4 * BlockSize)
	packed, err := QuantizeQ4_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ4_0: %v", err)
	}
	got, err := DequantizeQ4_0(packed, len(vals))
	if err != nil {
		t.Fatalf("DequantizeQ4_0: %v", err)
	}

	for b := 0; b*BlockSize < len(vals); b++ {
		block := vals[b*BlockSize : (b+1)*BlockSize]
		var amax float64
		for _, v := range block {
			amax = math.Max(amax, math.Abs(float64(v)))
		}
		// The scale comes from the signed extreme, so the representable range
		// is asymmetric: it reaches 8 steps toward the extreme's sign but only
		// 7 the other way. In-range values round to within half a step, and a
		// value clamped at the short end is off by at most a full step, amax/8.
		tol := amax/8*1.02 + 1e-6
		for i, want := range block {
			if diff := math.Abs(float64(got[b*BlockSize+i] - want)); diff > tol {
				t.Errorf("block %d element %d: got %v, want %v (error %g > %g)",
					b, i, got[b*BlockSize+i], want, diff, tol)
			}
		}
	}
}

// TestQ4_0ExactlyRepresentable pins the layout: values that land exactly on
// nibble steps must survive the round trip bit for bit.
func TestQ4_0ExactlyRepresentable(t *testing.T) {
	vals := make([]float32, BlockSize)
	for i := range vals {
		vals[i] = float32(i%16-8) * 0.5 // steps of 0.5, extreme -8*0.5 = -4
	}
	packed, err := QuantizeQ4_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ4_0: %v", err)
	}
	got, err := DequantizeQ4_0(packed, len(vals))
	if err != nil {
		t.Fatalf("DequantizeQ4_0: %v", err)
	}
	for i := range vals {
		if got[i] != vals[i] {
			t.Errorf("element %d: got %v, want %v", i, got[i], vals[i])
		}
	}
}

// TestQ4_0NibbleOrder verifies that element i occupies the low nibble and
// element i+16 the high nibble, matching what DequantizeQ4_0 reads.
func TestQ4_0NibbleOrder(t *testing.T) {
	vals := make([]float32, BlockSize)
	vals[0] = -8  // the signed extreme, so scale = 1
	vals[16] = -7 // must land in the high nibble of byte 0

	packed, err := QuantizeQ4_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ4_0: %v", err)
	}
	if got, want := packed[2]&0x0F, byte(0); got != want {
		t.Errorf("low nibble = %d, want %d (element 0 = -8 biases to 0)", got, want)
	}
	if got, want := packed[2]>>4, byte(1); got != want {
		t.Errorf("high nibble = %d, want %d (element 16 = -7 biases to 1)", got, want)
	}
}

func TestQ8_0RoundTripIdempotent(t *testing.T) {
	vals := testValues(3 * BlockSize)

	packed, err := QuantizeQ8_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ8_0: %v", err)
	}
	if want := 3 * blockBytesQ8_0; len(packed) != want {
		t.Fatalf("packed length = %d, want %d", len(packed), want)
	}

	round1, err := DequantizeQ8_0(packed, len(vals))
	if err != nil {
		t.Fatalf("DequantizeQ8_0: %v", err)
	}
	repacked, err := QuantizeQ8_0(round1)
	if err != nil {
		t.Fatalf("QuantizeQ8_0 (second pass): %v", err)
	}
	for i := range packed {
		if packed[i] != repacked[i] {
			t.Fatalf("byte %d changed on re-quantization: %#02x -> %#02x", i, packed[i], repacked[i])
		}
	}
}

func TestQ8_0RoundTripError(t *testing.T) {
	vals := testValues(3 * BlockSize)
	packed, err := QuantizeQ8_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ8_0: %v", err)
	}
	got, err := DequantizeQ8_0(packed, len(vals))
	if err != nil {
		t.Fatalf("DequantizeQ8_0: %v", err)
	}

	for b := 0; b*BlockSize < len(vals); b++ {
		block := vals[b*BlockSize : (b+1)*BlockSize]
		var amax float64
		for _, v := range block {
			amax = math.Max(amax, math.Abs(float64(v)))
		}
		tol := amax/254*1.02 + 1e-6
		for i, want := range block {
			if diff := math.Abs(float64(got[b*BlockSize+i] - want)); diff > tol {
				t.Errorf("block %d element %d: got %v, want %v (error %g > %g)",
					b, i, got[b*BlockSize+i], want, diff, tol)
			}
		}
	}
}

// TestZeroBlock covers the degenerate all-zero block, where the scale is zero
// and the reciprocal must not become an infinity.
func TestZeroBlock(t *testing.T) {
	vals := make([]float32, BlockSize)

	packed4, err := QuantizeQ4_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ4_0: %v", err)
	}
	got4, err := DequantizeQ4_0(packed4, BlockSize)
	if err != nil {
		t.Fatalf("DequantizeQ4_0: %v", err)
	}
	for i, v := range got4 {
		if v != 0 {
			t.Errorf("Q4_0 zero block element %d = %v, want 0", i, v)
		}
	}

	packed8, err := QuantizeQ8_0(vals)
	if err != nil {
		t.Fatalf("QuantizeQ8_0: %v", err)
	}
	got8, err := DequantizeQ8_0(packed8, BlockSize)
	if err != nil {
		t.Fatalf("DequantizeQ8_0: %v", err)
	}
	for i, v := range got8 {
		if v != 0 {
			t.Errorf("Q8_0 zero block element %d = %v, want 0", i, v)
		}
	}
}

// TestPartialBlockRejected covers the silent truncation: a length that is not a
// whole number of blocks used to return a slice whose tail stayed zero.
func TestPartialBlockRejected(t *testing.T) {
	if _, err := QuantizeQ4_0(make([]float32, 33)); err == nil {
		t.Error("QuantizeQ4_0 accepted 33 values, want an error")
	}
	if _, err := QuantizeQ8_0(make([]float32, 1)); err == nil {
		t.Error("QuantizeQ8_0 accepted 1 value, want an error")
	}
	if _, err := DequantizeQ4_0(make([]byte, 64), 33); err == nil {
		t.Error("DequantizeQ4_0 accepted 33 elements, want an error")
	}
	if _, err := DequantizeQ8_0(make([]byte, 64), 33); err == nil {
		t.Error("DequantizeQ8_0 accepted 33 elements, want an error")
	}
}
