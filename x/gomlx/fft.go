package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) FFT(operand compute.Value, fftType compute.FFTType, fftLength []int) (compute.Value, error) {
	if f.returned {
		return nil, fmt.Errorf("function already returned")
	}
	x, err := f.value(operand)
	if err != nil {
		return nil, err
	}
	lengths, err := normalizeFFTLength(x.Type, fftType, fftLength)
	if err != nil {
		return nil, err
	}
	typ, err := stablehloFFTType(fftType)
	if err != nil {
		return nil, err
	}
	z, err := f.raw.FFT(x, typ, lengths)
	if err != nil {
		return nil, err
	}
	return &node{f, z}, nil
}

func normalizeFFTLength(typ hlo.Type, fftType compute.FFTType, lengths []int) ([]int64, error) {
	if len(typ.Shape) == 0 {
		return nil, fmt.Errorf("fft: scalar operand")
	}
	if len(lengths) == 0 {
		n := typ.Shape[len(typ.Shape)-1]
		if fftType == compute.FFTInverseReal {
			if n == 0 {
				n = 0
			} else {
				n = 2 * (n - 1)
			}
		}
		lengths = []int{int(n)}
	}
	if len(lengths) == 0 || len(lengths) > 3 || len(lengths) > len(typ.Shape) {
		return nil, fmt.Errorf("fft: length rank must be between 1 and min(3, operand rank)")
	}
	out := make([]int64, len(lengths))
	for i, n := range lengths {
		if n < 0 {
			return nil, fmt.Errorf("fft: lengths must be nonnegative")
		}
		out[i] = int64(n)
	}
	return out, nil
}

func stablehloFFTType(typ compute.FFTType) (hlo.FFTType, error) {
	switch typ {
	case compute.FFTForward:
		return hlo.FFTForward, nil
	case compute.FFTInverse:
		return hlo.FFTInverse, nil
	case compute.FFTForwardReal:
		return hlo.FFTForwardReal, nil
	case compute.FFTInverseReal:
		return hlo.FFTInverseReal, nil
	}
	return "", fmt.Errorf("fft: invalid type %d", typ)
}
