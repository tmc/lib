package main

import (
	"math"
	"testing"
)

func TestFloat32to16RoundTrip(t *testing.T) {
	for h := range 1 << 16 {
		f := float16to32(uint16(h))
		if math.IsNaN(float64(f)) {
			if got := float32to16(f); got&0x7c00 != 0x7c00 || got&0x3ff == 0 {
				t.Errorf("float32to16(NaN %#04x) = %#04x, want a NaN", h, got)
			}
			continue
		}
		if got := float32to16(f); got != uint16(h) {
			t.Errorf("float32to16(%g) = %#04x, want %#04x", f, got, h)
		}
	}
}

func TestFloat32to16(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		want uint16
	}{
		{"one", 1, 0x3c00},
		{"tie to even down", 1 + 1.0/2048, 0x3c00},
		{"tie to even up", 1 + 3.0/2048, 0x3c02},
		{"above tie", 1 + 1.0/2048 + 1.0/65536, 0x3c01},
		{"max", 65504, 0x7bff},
		{"rounds to inf", 65520, 0x7c00},
		{"below inf tie", 65519, 0x7bff},
		{"overflow", 1e6, 0x7c00},
		{"negative overflow", -1e6, 0xfc00},
		{"smallest subnormal", 0x1p-24, 0x0001},
		{"half smallest subnormal", 0x1p-25, 0x0000},
		{"above half smallest", 0x1.01p-25, 0x0001},
		{"subnormal tie to even", 3 * 0x1p-25, 0x0002},
		{"largest subnormal rounds to normal", 0x1.ffffp-15, 0x0400},
		{"underflow", 1e-10, 0x0000},
		{"negative zero", float32(math.Copysign(0, -1)), 0x8000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := float32to16(tt.in); got != tt.want {
				t.Errorf("float32to16(%g) = %#04x, want %#04x", tt.in, got, tt.want)
			}
		})
	}
}

func TestFloat16to32(t *testing.T) {
	tests := []struct {
		in   uint16
		want float32
	}{
		{0x3c00, 1},
		{0xc000, -2},
		{0x7bff, 65504},
		{0x0400, 0x1p-14},
		{0x0001, 0x1p-24},
		{0x0200, 0x1p-15},
		{0x83ff, -0x3ffp-24},
		{0x7c00, float32(math.Inf(1))},
	}
	for _, tt := range tests {
		if got := float16to32(tt.in); got != tt.want {
			t.Errorf("float16to32(%#04x) = %g, want %g", tt.in, got, tt.want)
		}
	}
}
