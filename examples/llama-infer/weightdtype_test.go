package main

import (
	"bufio"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestWeightDTypeParsing checks that the mode is selected explicitly and that
// an unrecognized name is refused.
//
// A silent default to f32 on a misspelled mode would let a run report the
// precision it was asked for while executing another one, which is exactly the
// failure this control exists to prevent.
func TestWeightDTypeParsing(t *testing.T) {
	ok := map[string]WeightDType{
		"":         WeightF32,
		"f32":      WeightF32,
		"F32":      WeightF32,
		"float32":  WeightF32,
		"f16":      WeightF16,
		"F16":      WeightF16,
		"half":     WeightF16,
		"float16":  WeightF16,
		"bf16":     WeightBF16,
		"BF16":     WeightBF16,
		"bfloat16": WeightBF16,
	}
	for in, want := range ok {
		got, err := ParseWeightDType(in)
		if err != nil {
			t.Errorf("ParseWeightDType(%q) = error %v", in, err)
			continue
		}
		if got != want {
			t.Errorf("ParseWeightDType(%q) = %v, want %v", in, got, want)
		}
	}
	for _, in := range []string{"fp16", "float", "int8", "f8", "16", "F32 ", "auto"} {
		if got, err := ParseWeightDType(in); err == nil {
			t.Errorf("ParseWeightDType(%q) = %v, want an error", in, got)
		}
	}

	for _, tt := range []struct {
		d      WeightDType
		name   string
		bytes  int
		suffix string
		ckpt   string
	}{
		{WeightF32, "f32", 4, "", "F32"},
		{WeightF16, "f16", 2, "F16", "F16"},
		{WeightBF16, "bf16", 2, "BF16", "BF16"},
	} {
		if got := tt.d.String(); got != tt.name {
			t.Errorf("String() = %q, want %q", got, tt.name)
		}
		if got := tt.d.Bytes(); got != tt.bytes {
			t.Errorf("%s Bytes() = %d, want %d", tt.name, got, tt.bytes)
		}
		if got := tt.d.Suffix(); got != tt.suffix {
			t.Errorf("%s Suffix() = %q, want %q", tt.name, got, tt.suffix)
		}
		if got := tt.d.CheckpointDType(); got != tt.ckpt {
			t.Errorf("%s CheckpointDType() = %q, want %q", tt.name, got, tt.ckpt)
		}
	}
}

// TestFloat16MatchesIndependentOracle checks float32ToFloat16 against vectors
// produced by an implementation this package had no part in writing.
//
// testdata/f16_vectors.txt was generated with Python's struct.pack("e", ...),
// which is CPython's own IEEE binary16 encoder. Each line is the float32 input
// bit pattern and the expected half bit pattern, both as decimal integers.
// Checking against a foreign implementation is the point: a table this package
// wrote from its own output would confirm nothing.
func TestFloat16MatchesIndependentOracle(t *testing.T) {
	f, err := os.Open("testdata/f16_vectors.txt")
	if err != nil {
		t.Fatalf("open vectors: %v", err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	var n int
	for line := 1; sc.Scan(); line++ {
		fields := strings.Fields(sc.Text())
		if len(fields) != 2 {
			continue
		}
		bits, err := strconv.ParseUint(fields[0], 10, 32)
		if err != nil {
			t.Fatalf("line %d: %v", line, err)
		}
		want, err := strconv.ParseUint(fields[1], 10, 16)
		if err != nil {
			t.Fatalf("line %d: %v", line, err)
		}
		in := math.Float32frombits(uint32(bits))
		if got := float32ToFloat16(in); got != uint16(want) {
			t.Errorf("float32ToFloat16(%v) = 0x%04x, want 0x%04x", in, got, want)
		}
		n++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n < 500 {
		t.Fatalf("only %d vectors read, want the full file", n)
	}
	t.Logf("checked %d oracle vectors", n)
}

// TestFloat16RoundTripsEveryValue checks widening and narrowing over the entire
// half domain.
//
// Every one of the 65536 bit patterns is exercised, so subnormals, both zeros,
// both infinities and the boundaries between them are covered without having to
// enumerate them by hand. NaN payloads are excluded because narrowing is not
// required to preserve which NaN a value was.
func TestFloat16RoundTripsEveryValue(t *testing.T) {
	var checked int
	for i := 0; i < 1<<16; i++ {
		h := uint16(i)
		if h&0x7c00 == 0x7c00 && h&0x3ff != 0 {
			continue // NaN
		}
		if got := float32ToFloat16(float16ToFloat32(h)); got != h {
			t.Fatalf("half 0x%04x widened and narrowed to 0x%04x", h, got)
		}
		checked++
	}
	if checked < 60000 {
		t.Fatalf("only %d values checked", checked)
	}
}

// TestBfloat16RoundTripsEveryValue is the bfloat16 counterpart.
func TestBfloat16RoundTripsEveryValue(t *testing.T) {
	for i := 0; i < 1<<16; i++ {
		b := uint16(i)
		if b&0x7f80 == 0x7f80 && b&0x7f != 0 {
			continue // NaN
		}
		if got := float32ToBfloat16(bfloat16ToFloat32(b)); got != b {
			t.Fatalf("bfloat16 0x%04x widened and narrowed to 0x%04x", b, got)
		}
	}
}

// TestHalfConversionEdgeCases pins the behaviors that a table of ordinary
// values would not exercise.
func TestHalfConversionEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		f16  uint16
		bf16 uint16
	}{
		{"positive zero", 0, 0x0000, 0x0000},
		{"negative zero", float32(math.Copysign(0, -1)), 0x8000, 0x8000},
		{"one", 1, 0x3c00, 0x3f80},
		{"minus one", -1, 0xbc00, 0xbf80},
		{"largest finite half", 65504, 0x7bff, 0x4780},

		// 65520 is exactly halfway between the largest finite half and the
		// next power of two, so round-to-nearest-even takes it to infinity
		// rather than clamping to the largest finite value.
		{"half overflow boundary", 65520, 0x7c00, 0x4780},
		{"beyond half range", 1e30, 0x7c00, 0x714a},
		{"beyond half range negative", -1e30, 0xfc00, 0xf14a},

		// Underflow to a signed zero rather than to a positive one.
		{"underflows half", 1e-9, 0x0000, 0x3089},
		{"underflows half negative", -1e-9, 0x8000, 0xb089},

		// The smallest half subnormal and the value that rounds to it.
		{"smallest half subnormal", 5.9604645e-08, 0x0001, 0x3380},

		{"positive infinity", float32(math.Inf(1)), 0x7c00, 0x7f80},
		{"negative infinity", float32(math.Inf(-1)), 0xfc00, 0xff80},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := float32ToFloat16(tt.in); got != tt.f16 {
				t.Errorf("float32ToFloat16(%v) = 0x%04x, want 0x%04x", tt.in, got, tt.f16)
			}
			if got := float32ToBfloat16(tt.in); got != tt.bf16 {
				t.Errorf("float32ToBfloat16(%v) = 0x%04x, want 0x%04x", tt.in, got, tt.bf16)
			}
		})
	}

	// A NaN must stay a NaN in both formats rather than becoming an infinity.
	nan := float32(math.NaN())
	if got := float16ToFloat32(float32ToFloat16(nan)); !math.IsNaN(float64(got)) {
		t.Errorf("f16 NaN became %v", got)
	}
	if got := bfloat16ToFloat32(float32ToBfloat16(nan)); !math.IsNaN(float64(got)) {
		t.Errorf("bf16 NaN became %v", got)
	}
}

// TestEncodeWeightsStoresTwoBytesPerHalf checks the storage width and that
// encoding then decoding gives the values a kernel will actually multiply by.
func TestEncodeWeightsStoresTwoBytesPerHalf(t *testing.T) {
	src := make([]float32, 257) // deliberately not a multiple of 4
	for i := range src {
		src[i] = float32(i-128) / 7
	}
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		t.Run(d.String(), func(t *testing.T) {
			raw := d.EncodeWeights(src)
			if got, want := len(raw), len(src)*d.Bytes(); got != want {
				t.Fatalf("encoded %d bytes, want %d", got, want)
			}
			got := d.DecodeWeights(raw)
			if len(got) != len(src) {
				t.Fatalf("decoded %d values, want %d", len(got), len(src))
			}
			rt := d.RoundTrip(src)
			for i := range got {
				if got[i] != rt[i] {
					t.Fatalf("element %d: decode %v, RoundTrip %v", i, got[i], rt[i])
				}
			}
			if d == WeightF32 {
				for i := range got {
					if got[i] != src[i] {
						t.Fatalf("f32 element %d changed: %v, want %v", i, got[i], src[i])
					}
				}
				return
			}
			// Rounding must actually have happened, or the half path is
			// secretly keeping float32 precision.
			var changed int
			for i := range got {
				if got[i] != src[i] {
					changed++
				}
			}
			if changed == 0 {
				t.Fatal("no value changed, so nothing was rounded to half precision")
			}
		})
	}
}

// TestRoundTripIsIdempotent checks that rounding a already-rounded weight does
// not move it again.
//
// This is what makes "rounded once" a meaningful claim: if a second rounding
// changed the value, the error would depend on how many times a weight passed
// through the conversion.
func TestRoundTripIsIdempotent(t *testing.T) {
	src := make([]float32, 4096)
	for i := range src {
		src[i] = float32(math.Ldexp(float64(i%1000-500)/500, i%40-20))
	}
	for _, d := range []WeightDType{WeightF16, WeightBF16} {
		t.Run(d.String(), func(t *testing.T) {
			once := d.RoundTrip(src)
			twice := d.RoundTrip(once)
			for i := range once {
				if once[i] != twice[i] {
					t.Fatalf("element %d moved on a second rounding: %v then %v", i, once[i], twice[i])
				}
			}
		})
	}
}

// TestInvalidWeightDTypeIsRejected checks that a value outside the three named
// modes is refused rather than silently behaving like one of them.
//
// This matters because the zero value is f32 but an out-of-range value is not
// inert: Bytes reports 2 for it and Suffix reports the empty string, so it
// would select two-byte device storage and then bind the unsuffixed float32
// kernels, which read those bytes as float32. The result would be silent
// garbage rather than an error, which is exactly the failure the explicit-mode
// rule exists to prevent.
func TestInvalidWeightDTypeIsRejected(t *testing.T) {
	for _, d := range []WeightDType{-1, 3, 7, 1 << 20} {
		if d.Valid() {
			t.Errorf("WeightDType(%d).Valid() = true", int(d))
		}
		if err := d.check(); err == nil {
			t.Errorf("WeightDType(%d).check() = nil, want an error", int(d))
		}
	}
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		if !d.Valid() {
			t.Errorf("%v.Valid() = false", d)
		}
		if err := d.check(); err != nil {
			t.Errorf("%v.check() = %v", d, err)
		}
	}

	// The specific hazard, stated directly: an invalid value must never be able
	// to pair a 2-byte width with the unsuffixed kernel names.
	for _, d := range []WeightDType{-1, 3, 7} {
		if d.Bytes() == 2 && d.Suffix() == "" && d.Valid() {
			t.Errorf("WeightDType(%d) would store 2-byte weights and bind f32 kernels", int(d))
		}
	}
}
