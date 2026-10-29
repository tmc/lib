package hopper

import (
	"math"
	"testing"
)

func TestE4M3Bits(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		want uint8
	}{
		{"zero", 0, 0x00},
		{"negzero", float32(math.Copysign(0, -1)), 0x80},
		{"one", 1, 0x38},
		{"tenth", 0.1, 0x1d}, // 1.625 * 2^-4 = 0.1015625
		{"max", 448, 0x7e},
		{"saturate", 1e6, 0x7e},
		{"negsaturate", -500, 0xfe},
		{"inf", float32(math.Inf(1)), 0x7e},
		{"nan", float32(math.NaN()), 0x7f},
		{"minsub", 1.0 / 512, 0x01},
		{"halfminsub", 1.0 / 1024, 0x00}, // tie to even zero
		{"subtonormal", 0.0156, 0x08},    // rounds up into 2^-6
		{"tie_even_down", 1.0625, 0x38},  // between 1 and 1.125
		{"tie_even_up", 1.1875, 0x3a},    // between 1.125 and 1.25
	}
	for _, tt := range tests {
		if got := E4M3Bits(tt.in); got != tt.want {
			t.Errorf("%s: E4M3Bits(%g) = %#x, want %#x", tt.name, tt.in, got, tt.want)
		}
	}
}

// TestE4M3Exhaustive checks every encoding round-trips and every midpoint
// between neighbours rounds to the one with an even mantissa.
func TestE4M3Exhaustive(t *testing.T) {
	for b := 0; b < 0x7e; b++ {
		lo, hi := FromE4M3Bits(uint8(b)), FromE4M3Bits(uint8(b+1))
		if got := E4M3Bits(lo); got != uint8(b) {
			t.Fatalf("round trip %#x: got %#x", b, got)
		}
		if got := E4M3Bits(-lo); lo != 0 && got != uint8(b)|0x80 {
			t.Fatalf("negative round trip %#x: got %#x", b, got)
		}
		mid := float32((float64(lo) + float64(hi)) / 2)
		want := uint8(b)
		if b&1 == 1 {
			want = uint8(b + 1)
		}
		if got := E4M3Bits(mid); got != want {
			t.Fatalf("midpoint %g of %#x,%#x: got %#x want %#x", mid, b, b+1, got, want)
		}
		if got := E4M3Bits(math.Nextafter32(mid, 0)); got != uint8(b) {
			t.Fatalf("below midpoint of %#x: got %#x", b, got)
		}
		if got := E4M3Bits(math.Nextafter32(mid, 1000)); got != uint8(b+1) {
			t.Fatalf("above midpoint of %#x: got %#x", b, got)
		}
	}
	if !math.IsNaN(float64(FromE4M3Bits(0x7f))) || !math.IsNaN(float64(FromE4M3Bits(0xff))) {
		t.Fatal("0x7f and 0xff must decode to NaN")
	}
}

func TestBF16Bits(t *testing.T) {
	tests := []struct {
		in   float32
		want uint16
	}{
		{1, 0x3f80},
		{-2, 0xc000},
		{1 + 1.0/256, 0x3f80}, // tie, even stays
		{1 + 3.0/256, 0x3f82}, // tie, odd rounds up
		{1 + 1.5/256, 0x3f81}, // above tie
		{float32(math.Inf(-1)), 0xff80},
		{math.MaxFloat32, 0x7f80}, // rounds to infinity
	}
	for _, tt := range tests {
		if got := BF16Bits(tt.in); got != tt.want {
			t.Errorf("BF16Bits(%g) = %#x, want %#x", tt.in, got, tt.want)
		}
	}
	if got := BF16Bits(float32(math.NaN())); got&0x7fff <= 0x7f80 {
		t.Errorf("NaN encoded as %#x", got)
	}
}

func TestSplitBF16(t *testing.T) {
	for _, f := range []float32{1.0 / 3, -0.0123456789, 3.14159265, 1e-20, 12345.678} {
		hi, lo := SplitBF16(f)
		sum := float64(FromBF16Bits(hi)) + float64(FromBF16Bits(lo))
		if rel := math.Abs(sum-float64(f)) / math.Abs(float64(f)); rel > 1.0/(1<<16) {
			t.Errorf("SplitBF16(%g): hi+lo = %g, relative error %g", f, sum, rel)
		}
	}
}

func TestFromE2M1(t *testing.T) {
	want := []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
	for c, w := range want {
		if got := FromE2M1(uint8(c)); got != w {
			t.Errorf("FromE2M1(%d) = %g, want %g", c, got, w)
		}
	}
}

func TestDequantNVFP4(t *testing.T) {
	packed := make([]uint8, 8)
	packed[0] = 0x71 // element 0 code 1 (0.5), element 1 code 7 (6)
	packed[7] = 0xe0 // element 14 code 0, element 15 code 14 (-4)
	scales := []uint8{E4M3Bits(2)}
	dst := make([]float32, 16)
	if err := DequantNVFP4(dst, packed, scales, 0.25); err != nil {
		t.Fatal(err)
	}
	for i, want := range map[int]float32{0: 0.25, 1: 3, 2: 0, 15: -2} {
		if dst[i] != want {
			t.Errorf("dst[%d] = %g, want %g", i, dst[i], want)
		}
	}
	if err := DequantNVFP4(make([]float32, 15), packed, scales, 1); err == nil {
		t.Error("ragged row accepted")
	}
}
