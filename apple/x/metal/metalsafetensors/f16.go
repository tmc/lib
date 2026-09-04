//go:build darwin

package metalsafetensors

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TensorF16 reads or converts a named tensor into IEEE 754 half-precision
// bits. BF16 and F32 sources are converted with round-to-nearest; values
// outside the float16 range saturate to infinity.
func (w *Weights) TensorF16(name string) ([]uint16, error) {
	info, ok := w.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("metalsafetensors: tensor %q not found", name)
	}
	start := int(w.dataBase + int64(info.OffsetStart))
	end := int(w.dataBase + int64(info.OffsetEnd))
	if start < 0 || end > len(w.dataMMap) || start > end {
		return nil, fmt.Errorf("metalsafetensors: invalid offset range [%d, %d]", start, end)
	}
	raw := w.dataMMap[start:end]
	elemSize, ok := tensorElementSize(info.DType)
	if !ok || uint64(len(raw))%elemSize != 0 {
		return nil, fmt.Errorf("metalsafetensors: tensor %q: invalid %s byte length %d", name, info.DType, len(raw))
	}

	switch info.DType {
	case "F16":
		out := make([]uint16, len(raw)/2)
		for i := range out {
			out[i] = binary.LittleEndian.Uint16(raw[i*2:])
		}
		return out, nil
	case "BF16":
		out := make([]uint16, len(raw)/2)
		for i := range out {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = f32BitsToF16(bits)
		}
		return out, nil
	case "F32":
		out := make([]uint16, len(raw)/4)
		for i := range out {
			out[i] = f32BitsToF16(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	}
	return nil, fmt.Errorf("metalsafetensors: cannot convert dtype %q to f16", info.DType)
}

// TensorMetalSliceF16 allocates a shared Metal unified memory slice
// holding the tensor as half-precision bits.
func (w *Weights) TensorMetalSliceF16(dev *metal.Device, name string) (metalmem.Slice[uint16], error) {
	halves, err := w.TensorF16(name)
	if err != nil {
		return metalmem.Slice[uint16]{}, err
	}
	return metalmem.AllocFromSlice(dev, halves)
}

// f32BitsToF16 converts float32 bits to float16 bits with
// round-to-nearest-even.
func f32BitsToF16(bits uint32) uint16 {
	sign := uint16(bits >> 16 & 0x8000)
	exp := int32(bits>>23&0xff) - 127 + 15
	mant := bits & 0x7fffff

	switch {
	case bits&0x7fffffff == 0:
		return sign
	case exp >= 0x1f:
		if int32(bits>>23&0xff) == 0xff && mant != 0 {
			return sign | 0x7e00 // NaN
		}
		return sign | 0x7c00 // overflow to infinity
	case exp <= 0:
		if exp < -10 {
			return sign // underflow to zero
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		round := mant >> (shift - 1) & 1
		sticky := mant&(1<<(shift-1)-1) != 0
		if round == 1 && (sticky || half&1 == 1) {
			half++
		}
		return sign | half
	}

	half := sign | uint16(exp)<<10 | uint16(mant>>13)
	round := mant >> 12 & 1
	sticky := mant&0xfff != 0
	if round == 1 && (sticky || half&1 == 1) {
		half++
	}
	return half
}

// F32ToF16 converts a float32 value into IEEE 754 half-precision bits.
func F32ToF16(v float32) uint16 {
	return f32BitsToF16(math.Float32bits(v))
}

// F16ToF32 converts IEEE 754 half-precision bits into a float32 value.
func F16ToF32(h uint16) float32 {
	return f16ToF32(h)
}

// f16ToF32 converts float16 bits to float32, for tests and readback.
func f16ToF32(h uint16) float32 {
	return math.Float32frombits(f16BitsToF32Bits(h))
}

func f16BitsToF32Bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h >> 10 & 0x1f)
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return sign
		}
		// subnormal: normalize
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		return sign | e<<23 | (mant&0x3ff)<<13
	case 0x1f:
		return sign | 0xff<<23 | mant<<13
	}
	return sign | (exp+127-15)<<23 | mant<<13
}
