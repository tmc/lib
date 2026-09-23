package main

import (
	"math"
	"testing"
)

func TestFP8KVSwitches(t *testing.T) {
	t.Setenv("LLAMA_INFER_FP8_KV", "")
	if fp8KVEnabled() {
		t.Fatal("fp8 kv enabled with the variable unset")
	}
	t.Setenv("LLAMA_INFER_FP8_KV", "true")
	if fp8KVEnabled() {
		t.Fatal(`fp8 kv enabled by "true"; only "1" selects it`)
	}
	t.Setenv("LLAMA_INFER_FP8_KV", "1")
	if !fp8KVEnabled() {
		t.Fatal("fp8 kv not enabled by 1")
	}
	for _, test := range []struct {
		env  string
		want float32
		ok   bool
	}{
		{"", 1, true},
		{"0.25", 0.25, true},
		{"8", 8, true},
		{"0", 0, false},
		{"-1", 0, false},
		{"inf", 0, false},
		{"nan", 0, false},
		{"x", 0, false},
	} {
		t.Setenv("LLAMA_INFER_FP8_KV_SCALE", test.env)
		got, err := fp8KVScale()
		if (err == nil) != test.ok || got != test.want {
			t.Errorf("LLAMA_INFER_FP8_KV_SCALE=%q: got %v, %v; want %v, ok=%v", test.env, got, err, test.want, test.ok)
		}
	}
}

func TestFP8E4M3Values(t *testing.T) {
	tests := []struct {
		x    float32
		code uint8
	}{
		{0, 0x00},
		{float32(math.Copysign(0, -1)), 0x80},
		{1, 0x38},
		{-1, 0xb8},
		{1.125, 0x39},
		{1.0625, 0x38}, // tie between 1 and 1.125 rounds to the even code
		{1.1875, 0x3a}, // tie between 1.125 and 1.25 rounds to the even code
		{1.07, 0x39},   // above the tie rounds up
		{2, 0x40},
		{448, 0x7e},
		{447.9, 0x7e},
		{449, 0x7e},
		{1e9, 0x7e},
		{float32(math.Inf(1)), 0x7e},
		{float32(math.Inf(-1)), 0xfe},
		{0.015625, 0x08},     // 2^-6, the smallest normal
		{0.001953125, 0x01},  // 2^-9, the smallest subnormal
		{0.0009765625, 0x00}, // 2^-10 ties down to zero
		{0.001, 0x01},
		{0.013671875, 0x07},  // 7 * 2^-9, the largest subnormal
		{0.0146484375, 0x08}, // 7.5 * 2^-9 ties up to the smallest normal
		{1e-30, 0x00},
	}
	for _, test := range tests {
		if got := fp8E4M3(test.x); got != test.code {
			t.Errorf("fp8E4M3(%g) = %#02x, want %#02x", test.x, got, test.code)
		}
	}
	if got := fp8E4M3(float32(math.NaN())); got&0x7f != 0x7f {
		t.Errorf("fp8E4M3(NaN) = %#02x, want the NaN code", got)
	}
	if v := fp8E4M3Value(0x7f); !math.IsNaN(float64(v)) {
		t.Errorf("fp8E4M3Value(0x7f) = %g, want NaN", v)
	}
	if v := fp8E4M3Value(0x7e); v != fp8E4M3Max {
		t.Errorf("fp8E4M3Value(0x7e) = %g, want %d", v, fp8E4M3Max)
	}
}

// TestFP8E4M3RoundTrip decodes every finite code and encodes it back, and
// checks that the codes are strictly ordered by value.
func TestFP8E4M3RoundTrip(t *testing.T) {
	prev := float32(math.Inf(-1))
	for c := 0; c < 256; c++ {
		code := uint8(c)
		if code&0x7f == 0x7f {
			continue
		}
		v := fp8E4M3Value(code)
		if got := fp8E4M3(v); got != code && !(v == 0 && got&0x7f == 0) {
			t.Errorf("fp8E4M3(fp8E4M3Value(%#02x) = %g) = %#02x", code, v, got)
		}
		if code < 0x80 {
			if v <= prev && code != 0 {
				t.Errorf("code %#02x decodes to %g, not above %g", code, v, prev)
			}
			prev = v
		}
	}
}

// TestFP8E4M3Nearest checks the encoder against a search over every finite
// code: the chosen code is the nearest, and a tie goes to the even code.
func TestFP8E4M3Nearest(t *testing.T) {
	var finite []uint8
	for c := 0; c < 256; c++ {
		if uint8(c)&0x7f != 0x7f {
			finite = append(finite, uint8(c))
		}
	}
	nearest := func(x float32) uint8 {
		best, bestDist := uint8(0), math.Inf(1)
		for _, c := range finite {
			d := math.Abs(float64(fp8E4M3Value(c)) - float64(x))
			if d < bestDist || (d == bestDist && c&1 == 0) {
				best, bestDist = c, d
			}
		}
		return best
	}
	// A deterministic sweep: every power of two from 2^-12 to 2^9 with a
	// fine fan of mantissas, positive and negative.
	for e := -12; e <= 9; e++ {
		for i := 0; i < 64; i++ {
			for _, sign := range []float32{1, -1} {
				x := sign * float32(math.Ldexp(1+float64(i)/64, e))
				if math.Abs(float64(x)) >= fp8E4M3Max {
					continue
				}
				want := nearest(x)
				got := fp8E4M3(x)
				if got != want && !(fp8E4M3Value(got) == fp8E4M3Value(want)) {
					t.Errorf("fp8E4M3(%g) = %#02x (%g), nearest is %#02x (%g)", x, got, fp8E4M3Value(got), want, fp8E4M3Value(want))
				}
			}
		}
	}
	if got := fp8KVRoundTrip(3, 0.5); got != 3 {
		t.Errorf("fp8KVRoundTrip(3, 0.5) = %g, want 3", got)
	}
	if got := fp8KVRoundTrip(1000, 1); got != fp8E4M3Max {
		t.Errorf("fp8KVRoundTrip(1000, 1) = %g, want saturation at %d", got, fp8E4M3Max)
	}
}
