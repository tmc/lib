package main

import (
	"math"
	"testing"
)

func TestUlps(t *testing.T) {
	next32 := float64(math.Nextafter32(0.5, 1))
	tests := []struct {
		x, y float64
		want string
	}{
		{0.5, 0.5, "0 float32 ulp"},
		{0.5, next32, "1 float32 ulp"},
		{next32, 0.5, "1 float32 ulp"},
		{0.1, math.Nextafter(0.1, 1), "0 float32 ulp, 1 float64 ulp"},
		{0.1, math.Nextafter(math.Nextafter(0.1, 0), 0), "0 float32 ulp, 2 float64 ulp"},
		{0.25, 0.25 + 0x1p-20, "32 float32 ulp"},
		{math.Copysign(0, -1), 0, "0 float32 ulp"}, // bits differ; values do not
		{-float64(math.SmallestNonzeroFloat32), float64(math.SmallestNonzeroFloat32), "2 float32 ulp"},
		{1, math.NaN(), "NaN"},
	}
	for _, tt := range tests {
		if got := ulps(tt.x, tt.y); got != tt.want {
			t.Errorf("ulps(%v, %v) = %q, want %q", tt.x, tt.y, got, tt.want)
		}
	}
}

func TestOrdered(t *testing.T) {
	xs := []float64{math.Inf(-1), -2, -1, -math.SmallestNonzeroFloat64, 0, math.SmallestNonzeroFloat64, 1, 2, math.Inf(1)}
	for i := 1; i < len(xs); i++ {
		if a, b := ordered64(xs[i-1]), ordered64(xs[i]); a >= b {
			t.Errorf("ordered64(%v) = %d >= ordered64(%v) = %d", xs[i-1], a, xs[i], b)
		}
		x, y := float32(xs[i-1]), float32(xs[i])
		if x != y {
			if a, b := ordered32(x), ordered32(y); a >= b {
				t.Errorf("ordered32(%v) = %d >= ordered32(%v) = %d", x, a, y, b)
			}
		}
	}
}
