package main

import (
	"math"
	"testing"
)

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
