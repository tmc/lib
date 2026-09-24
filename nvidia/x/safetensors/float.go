package safetensors

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Float32ToFloat16 returns the IEEE 754 binary16 value nearest to f, with ties
// to even. Values beyond the binary16 range become infinities, values below
// half the smallest subnormal become signed zeros, and NaNs stay NaNs.
func Float32ToFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23) & 0xff
	mant := b & 0x7fffff
	switch {
	case exp == 0xff && mant != 0:
		return sign | 0x7e00 | uint16(mant>>13)
	case exp == 0xff:
		return sign | 0x7c00
	case exp == 0:
		return sign // float32 subnormals are far below the binary16 range
	}
	if e := exp - 127 + 15; e > 0 {
		if e >= 0x1f {
			return sign | 0x7c00
		}
		h := uint16(e)<<10 | uint16(mant>>13)
		if rem := mant & 0x1fff; rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
			h++ // a carry out of the mantissa correctly bumps the exponent
		}
		return sign | h
	}
	// The result is subnormal: count units of 2^-24.
	m := mant | 0x800000
	shift := uint(126 - exp)
	if shift > 24 {
		return sign
	}
	h := uint16(m >> shift)
	rem, half := m&(1<<shift-1), uint32(1)<<(shift-1)
	if rem > half || rem == half && h&1 == 1 {
		h++
	}
	return sign | h
}

// Float16ToFloat32 returns the float32 value of the binary16 value h.
// The conversion is exact.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	case exp != 0:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	case mant == 0:
		return math.Float32frombits(sign)
	}
	f := float32(mant) * (1.0 / (1 << 24))
	if sign != 0 {
		f = -f
	}
	return f
}

// Float32ToBFloat16 returns the bfloat16 value nearest to f, with ties to even.
// NaNs stay NaNs.
func Float32ToBFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7fffffff > 0x7f800000 {
		return uint16(b>>16) | 0x40
	}
	return uint16((b + 0x7fff + (b>>16)&1) >> 16)
}

// BFloat16ToFloat32 returns the float32 value of the bfloat16 value h.
// The conversion is exact.
func BFloat16ToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// EncodeFloat32 returns src encoded as little-endian elements of dtype, which
// must be F16, BF16, F32 or F64. Narrowing conversions round to nearest even.
func EncodeFloat32(dtype Dtype, src []float32) ([]byte, error) {
	out := make([]byte, len(src)*dtype.Size())
	switch dtype {
	case F16:
		for i, v := range src {
			binary.LittleEndian.PutUint16(out[2*i:], Float32ToFloat16(v))
		}
	case BF16:
		for i, v := range src {
			binary.LittleEndian.PutUint16(out[2*i:], Float32ToBFloat16(v))
		}
	case F32:
		for i, v := range src {
			binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
		}
	case F64:
		for i, v := range src {
			binary.LittleEndian.PutUint64(out[8*i:], math.Float64bits(float64(v)))
		}
	default:
		return nil, fmt.Errorf("safetensors: cannot encode float32 as %q", dtype)
	}
	return out, nil
}

// DecodeFloat32 returns the elements of data, little-endian values of dtype
// F16, BF16, F32 or F64, as float32. F64 values are rounded to float32.
func DecodeFloat32(dtype Dtype, data []byte) ([]float32, error) {
	size := dtype.Size()
	if size == 0 || len(data)%size != 0 {
		return nil, fmt.Errorf("safetensors: cannot decode %d bytes of %q", len(data), dtype)
	}
	out := make([]float32, len(data)/size)
	switch dtype {
	case F16:
		for i := range out {
			out[i] = Float16ToFloat32(binary.LittleEndian.Uint16(data[2*i:]))
		}
	case BF16:
		for i := range out {
			out[i] = BFloat16ToFloat32(binary.LittleEndian.Uint16(data[2*i:]))
		}
	case F32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
	case F64:
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:])))
		}
	default:
		return nil, fmt.Errorf("safetensors: cannot decode %q as float32", dtype)
	}
	return out, nil
}
