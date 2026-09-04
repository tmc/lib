//go:build linux

package main

import (
	"math"
	"testing"
)

// These controls exercise the same comparator used for the physical CUDA
// differentials. Empty samples and invalid tolerances are errors, not evidence.
func TestCUDAQualificationComparator(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	tests := []struct {
		name       string
		got, want  []float32
		atol, rtol float64
		wantError  bool
	}{
		{name: "equal", got: []float32{0, -2, 3}, want: []float32{0, -2, 3}},
		{name: "absolute", got: []float32{0.125}, want: []float32{0}, atol: 0.25},
		{name: "relative", got: []float32{-10.125}, want: []float32{-10}, rtol: 0.025},
		{name: "boundary", got: []float32{1.25}, want: []float32{1}, atol: 0.125, rtol: 0.125},
		{name: "wrong output", got: []float32{1, 3}, want: []float32{1, 2}, atol: 1e-5, rtol: 1e-5, wantError: true},
		{name: "wrong near zero", got: []float32{0.01}, want: []float32{0}, atol: 1e-5, rtol: 1e-5, wantError: true},
		{name: "short candidate", got: []float32{1}, want: []float32{1, 2}, wantError: true},
		{name: "long candidate", got: []float32{1, 2}, want: []float32{1}, wantError: true},
		{name: "empty", wantError: true},
		{name: "nan candidate", got: []float32{1, nan}, want: []float32{1, 2}, wantError: true},
		{name: "nan oracle", got: []float32{1, 2}, want: []float32{1, nan}, wantError: true},
		{name: "nan both", got: []float32{nan}, want: []float32{nan}, wantError: true},
		{name: "positive infinity candidate", got: []float32{inf}, want: []float32{1}, wantError: true},
		{name: "negative infinity candidate", got: []float32{-inf}, want: []float32{1}, wantError: true},
		{name: "positive infinity oracle", got: []float32{1}, want: []float32{inf}, wantError: true},
		{name: "negative infinity oracle", got: []float32{1}, want: []float32{-inf}, wantError: true},
		{name: "equal infinity", got: []float32{inf}, want: []float32{inf}, wantError: true},
		{name: "nan absolute tolerance", got: []float32{1}, want: []float32{1}, atol: math.NaN(), wantError: true},
		{name: "nan relative tolerance", got: []float32{1}, want: []float32{1}, rtol: math.NaN(), wantError: true},
		{name: "infinite absolute tolerance", got: []float32{1}, want: []float32{1}, atol: math.Inf(1), wantError: true},
		{name: "infinite relative tolerance", got: []float32{1}, want: []float32{1}, rtol: math.Inf(1), wantError: true},
		{name: "negative absolute tolerance", got: []float32{1}, want: []float32{1}, atol: -1, wantError: true},
		{name: "negative relative tolerance", got: []float32{1}, want: []float32{1}, rtol: -1, wantError: true},
		{name: "overflowed tolerance", got: []float32{2}, want: []float32{2}, rtol: math.MaxFloat64, wantError: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := cudaQualificationClose(tt.got, tt.want, tt.atol, tt.rtol)
			if (err != nil) != tt.wantError {
				t.Fatalf("comparison error = %v, want error %v", err, tt.wantError)
			}
		})
	}
}
