package safetensors

import (
	"math"
	"testing"
)

// TestFloat16RoundTrip checks every binary16 bit pattern: widening is exact,
// so narrowing the widened value must give the pattern back.
func TestFloat16RoundTrip(t *testing.T) {
	for h := range 1 << 16 {
		f := Float16ToFloat32(uint16(h))
		got := Float32ToFloat16(f)
		if math.IsNaN(float64(f)) {
			if h&0x7c00 != 0x7c00 || h&0x3ff == 0 {
				t.Fatalf("%#04x widened to NaN", h)
			}
			if got&0x7c00 != 0x7c00 || got&0x3ff == 0 {
				t.Fatalf("NaN %#04x narrowed to %#04x", h, got)
			}
			continue
		}
		if got != uint16(h) {
			t.Fatalf("Float32ToFloat16(Float16ToFloat32(%#04x)=%v) = %#04x", h, f, got)
		}
	}
}

// TestFloat16Ties checks rounding at and around the midpoint of every pair of
// adjacent positive binary16 values, including the pair (max finite, +Inf) and
// (0, smallest subnormal). Each midpoint is exact in float32.
func TestFloat16Ties(t *testing.T) {
	for h := uint16(0); h < 0x7c00; h++ {
		lo := float64(Float16ToFloat32(h))
		var hi float64
		if h+1 == 0x7c00 {
			hi = 65536 // the next binade's first value, where rounding overflows
		} else {
			hi = float64(Float16ToFloat32(h + 1))
		}
		mid := float32((lo + hi) / 2)
		if float64(mid) != (lo+hi)/2 {
			t.Fatalf("midpoint of %#04x not exact in float32", h)
		}
		even := h
		if h&1 == 1 {
			even = h + 1
		}
		for _, tt := range []struct {
			in   float32
			want uint16
		}{
			{mid, even},
			{math.Nextafter32(mid, 0), h},
			{math.Nextafter32(mid, float32(math.Inf(1))), h + 1},
		} {
			if got := Float32ToFloat16(tt.in); got != tt.want {
				t.Fatalf("Float32ToFloat16(%v) = %#04x, want %#04x", tt.in, got, tt.want)
			}
			if got := Float32ToFloat16(-tt.in); got != tt.want|0x8000 {
				t.Fatalf("Float32ToFloat16(%v) = %#04x, want %#04x", -tt.in, got, tt.want|0x8000)
			}
		}
	}
}

func TestFloat16Specials(t *testing.T) {
	tests := []struct {
		in   float32
		want uint16
	}{
		{0, 0},
		{float32(math.Copysign(0, -1)), 0x8000},
		{float32(math.Inf(1)), 0x7c00},
		{float32(math.Inf(-1)), 0xfc00},
		{1e10, 0x7c00},
		{math.SmallestNonzeroFloat32, 0},
		{1, 0x3c00},
		{65504, 0x7bff},
		{0x1p-24, 0x0001},
		{0x1p-14, 0x0400},
	}
	for _, tt := range tests {
		if got := Float32ToFloat16(tt.in); got != tt.want {
			t.Errorf("Float32ToFloat16(%v) = %#04x, want %#04x", tt.in, got, tt.want)
		}
	}
	if got := Float32ToFloat16(float32(math.NaN())); got&0x7c00 != 0x7c00 || got&0x3ff == 0 {
		t.Errorf("Float32ToFloat16(NaN) = %#04x, not a NaN", got)
	}
}

func TestBFloat16(t *testing.T) {
	for h := range 1 << 16 {
		f := BFloat16ToFloat32(uint16(h))
		got := Float32ToBFloat16(f)
		if math.IsNaN(float64(f)) {
			if !math.IsNaN(float64(BFloat16ToFloat32(got))) {
				t.Fatalf("NaN %#04x narrowed to %#04x", h, got)
			}
			continue
		}
		if got != uint16(h) {
			t.Fatalf("Float32ToBFloat16(BFloat16ToFloat32(%#04x)) = %#04x", h, got)
		}
		// The midpoint between h and h+1 has bit 15 set and nothing below.
		if h&0x7fff >= 0x7f80 {
			continue
		}
		mid := math.Float32frombits(uint32(h)<<16 | 0x8000)
		even := uint16(h)
		if h&1 == 1 {
			even++
		}
		if got := Float32ToBFloat16(mid); got != even {
			t.Fatalf("Float32ToBFloat16(%v) = %#04x, want %#04x", mid, got, even)
		}
		if got := Float32ToBFloat16(math.Float32frombits(uint32(h)<<16 | 0x8001)); got != uint16(h)+1 {
			t.Fatalf("above midpoint of %#04x rounded to %#04x", h, got)
		}
		if got := Float32ToBFloat16(math.Float32frombits(uint32(h)<<16 | 0x7fff)); got != uint16(h) {
			t.Fatalf("below midpoint of %#04x rounded to %#04x", h, got)
		}
	}
}

func TestEncodeDecodeFloat32(t *testing.T) {
	src := []float32{0, 1, -2.5, 1024, 0.1}
	for _, dtype := range []Dtype{F16, BF16, F32, F64} {
		data, err := EncodeFloat32(dtype, src)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != len(src)*dtype.Size() {
			t.Fatalf("%s: %d bytes", dtype, len(data))
		}
		got, err := DecodeFloat32(dtype, data)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 4 { // the first four values are exact in every dtype
			if got[i] != src[i] {
				t.Errorf("%s: element %d = %v, want %v", dtype, i, got[i], src[i])
			}
		}
	}
	if _, err := EncodeFloat32(I32, src); err == nil {
		t.Error("EncodeFloat32 accepted I32")
	}
	if _, err := DecodeFloat32(F16, []byte{1}); err == nil {
		t.Error("DecodeFloat32 accepted a partial element")
	}
}
