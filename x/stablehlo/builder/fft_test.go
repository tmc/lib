package builder

import (
	"slices"
	"testing"
)

func TestFFT(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   Type
		typ  FFTType
		len  []int64
		want Type
	}{
		{"fft", Type{Shape: []int64{2, 4}, ElementType: "complex<f32>"}, FFTForward, []int64{4}, Type{Shape: []int64{2, 4}, ElementType: "complex<f32>"}},
		{"rfft", Type{Shape: []int64{2, 5}, ElementType: "f64"}, FFTForwardReal, []int64{5}, Type{Shape: []int64{2, 3}, ElementType: "complex<f64>"}},
		{"irfft", Type{Shape: []int64{2, 3}, ElementType: "complex<f64>"}, FFTInverseReal, []int64{5}, Type{Shape: []int64{2, 5}, ElementType: "f64"}},
		{"rfft2", Type{Shape: []int64{2, 3, 4}, ElementType: "f32"}, FFTForwardReal, []int64{3, 4}, Type{Shape: []int64{2, 3, 3}, ElementType: "complex<f32>"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := NewFunc("main", tc.in)
			got, err := b.FFT(b.Arg(0), tc.typ, tc.len)
			if err != nil {
				t.Fatal(err)
			}
			if got.Type.ElementType != tc.want.ElementType || !slices.Equal(got.Type.Shape, tc.want.Shape) {
				t.Fatalf("type %v, want %v", got.Type, tc.want)
			}
		})
	}
}

func TestFFTValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   Type
		typ  FFTType
		len  []int64
	}{
		{"scalar", Type{ElementType: "complex<f32>"}, FFTForward, []int64{1}},
		{"integer", Type{Shape: []int64{4}, ElementType: "i32"}, FFTForward, []int64{4}},
		{"rfft length", Type{Shape: []int64{4}, ElementType: "f32"}, FFTForwardReal, []int64{3}},
		{"irfft length", Type{Shape: []int64{3}, ElementType: "complex<f32>"}, FFTInverseReal, []int64{6}},
		{"irfft multidimensional length", Type{Shape: []int64{2, 3}, ElementType: "complex<f32>"}, FFTInverseReal, []int64{4, 4}},
		{"empty length", Type{Shape: []int64{4}, ElementType: "complex<f32>"}, FFTForward, nil},
		{"too many axes", Type{Shape: []int64{1, 1, 1, 1}, ElementType: "complex<f32>"}, FFTForward, []int64{1, 1, 1, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := NewFunc("main", tc.in)
			if _, err := b.FFT(b.Arg(0), tc.typ, tc.len); err == nil {
				t.Fatal("accepted invalid FFT")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatal("invalid FFT changed builder")
			}
		})
	}
}
