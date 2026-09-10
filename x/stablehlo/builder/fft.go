package builder

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// FFTType identifies a StableHLO Fourier transform.
type FFTType string

const (
	// FFTForward computes a complex forward transform.
	FFTForward FFTType = "FFT"
	// FFTInverse computes a complex inverse transform.
	FFTInverse FFTType = "IFFT"
	// FFTForwardReal computes a real forward transform.
	FFTForwardReal FFTType = "RFFT"
	// FFTInverseReal computes a real inverse transform.
	FFTInverseReal FFTType = "IRFFT"
)

// FFT emits stablehlo.fft for the trailing dimensions in fftLength.
func (b *FuncBuilder) FFT(operand Value, fftType FFTType, fftLength []int64) (Value, error) {
	if err := checkType(operand.Type); err != nil {
		return Value{}, fmt.Errorf("fft: %w", err)
	}
	if len(fftLength) == 0 || len(fftLength) > 3 || len(fftLength) > len(operand.Type.Shape) {
		return Value{}, fmt.Errorf("fft: length rank must be between 1 and min(3, operand rank)")
	}
	for _, n := range fftLength {
		if n < 0 {
			return Value{}, fmt.Errorf("fft: lengths must be nonnegative")
		}
	}

	result := Type{Shape: slices.Clone(operand.Type.Shape)}
	first := len(result.Shape) - len(fftLength)
	switch fftType {
	case FFTForward, FFTInverse:
		if !complexType(operand.Type.ElementType) {
			return Value{}, fmt.Errorf("fft: complex operand required")
		}
		result.ElementType = operand.Type.ElementType
	case FFTForwardReal:
		if !fftRealType(operand.Type.ElementType) {
			return Value{}, fmt.Errorf("rfft: f32 or f64 operand required")
		}
		for i, n := range fftLength {
			if !compatibleDim(operand.Type.Shape[first+i], n) {
				return Value{}, fmt.Errorf("rfft: operand dimension %d does not match length", first+i)
			}
		}
		result.ElementType = "complex<" + operand.Type.ElementType + ">"
		if operand.Type.Shape[len(operand.Type.Shape)-1] == 0 {
			result.Shape[len(result.Shape)-1] = 0
		} else {
			result.Shape[len(result.Shape)-1] = fftLength[len(fftLength)-1]/2 + 1
		}
	case FFTInverseReal:
		if !complexType(operand.Type.ElementType) {
			return Value{}, fmt.Errorf("irfft: complex operand required")
		}
		result.ElementType = realFFTType(operand.Type.ElementType)
		for i, n := range fftLength {
			if i < len(fftLength)-1 && !compatibleDim(operand.Type.Shape[first+i], n) {
				return Value{}, fmt.Errorf("irfft: operand dimension %d does not match length", first+i)
			}
			result.Shape[first+i] = n
		}
		want := fftLength[len(fftLength)-1]/2 + 1
		if fftLength[len(fftLength)-1] == 0 {
			want = 0
		}
		if !compatibleDim(operand.Type.Shape[len(operand.Type.Shape)-1], want) {
			return Value{}, fmt.Errorf("irfft: operand final dimension does not match length")
		}
	default:
		return Value{}, fmt.Errorf("fft: invalid type %q", fftType)
	}

	out := Value{ID: b.next(), Type: result}
	b.Ops = append(b.Ops, fmt.Sprintf("%s = \"stablehlo.fft\"(%s) {fft_type = #stablehlo<fft_type %s>, fft_length = array<i64: %s>} : (%s) -> %s", out.ID, operand.ID, fftType, fftLengthText(fftLength), operand.Type, out.Type))
	return out, nil
}

func complexType(typ string) bool { return typ == "complex<f32>" || typ == "complex<f64>" }

func fftRealType(typ string) bool { return typ == "f32" || typ == "f64" }

func realFFTType(typ string) string {
	if typ == "complex<f32>" {
		return "f32"
	}
	return "f64"
}

func fftLengthText(lengths []int64) string {
	text := make([]string, len(lengths))
	for i, n := range lengths {
		text[i] = strconv.FormatInt(n, 10)
	}
	return strings.Join(text, ", ")
}
