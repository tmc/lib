// SPDX-License-Identifier: Apache-2.0

package metalquant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// QuantFormat specifies the sub-byte weight quantization format.
type QuantFormat string

const (
	FormatQ4_0 QuantFormat = "Q4_0" // 4-bit symmetric with float16 scale per 32 elements
	FormatQ8_0 QuantFormat = "Q8_0" // 8-bit symmetric with float16 scale per 32 elements
	FormatAWQ  QuantFormat = "AWQ"  // 4-bit asymmetric with per-group FP16 scale and zero
)

// BlockQ4_0 represents a 32-element Q4_0 quantized block (18 bytes total).
type BlockQ4_0 struct {
	Scale  uint16   // float16 scale factor
	Quants [16]byte // 32 4-bit nibbles (low/high)
}

// BlockQ8_0 represents a 32-element Q8_0 quantized block (34 bytes total).
type BlockQ8_0 struct {
	Scale  uint16   // float16 scale factor
	Quants [32]int8 // 32 8-bit integers
}

// BlockSize is the number of elements sharing one quantization scale.
const BlockSize = 32

// Block byte sizes: an fp16 scale followed by the packed quants.
const (
	blockBytesQ4_0 = 2 + BlockSize/2
	blockBytesQ8_0 = 2 + BlockSize
)

// QuantizeQ4_0 packs values into Q4_0 blocks: per 32 elements, an fp16 scale
// followed by 16 bytes holding element i in the low nibble and element i+16 in
// the high nibble, each biased by 8. len(values) must be a multiple of
// BlockSize.
func QuantizeQ4_0(values []float32) ([]byte, error) {
	if len(values)%BlockSize != 0 {
		return nil, fmt.Errorf("metalquant: %d values is not a multiple of the %d-element block", len(values), BlockSize)
	}

	out := make([]byte, len(values)/BlockSize*blockBytesQ4_0)
	for b := 0; b*BlockSize < len(values); b++ {
		block := values[b*BlockSize : (b+1)*BlockSize]
		offset := b * blockBytesQ4_0

		// The scale is derived from the signed extreme so that it maps onto
		// the asymmetric [-8, 7] range of a biased nibble.
		var extreme, amax float32
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax, extreme = a, v
			}
		}

		// Round the scale through fp16 before using it, so quantizing an
		// already-dequantized block reproduces these bytes exactly.
		scaleBits := float32ToFloat16(extreme / -8)
		scale := float16ToFloat32(scaleBits)
		binary.LittleEndian.PutUint16(out[offset:offset+2], scaleBits)

		var inv float32
		if scale != 0 {
			inv = 1 / scale
		}
		for i := 0; i < BlockSize/2; i++ {
			lo := quantNibble(block[i], inv)
			hi := quantNibble(block[i+BlockSize/2], inv)
			out[offset+2+i] = lo | hi<<4
		}
	}
	return out, nil
}

// quantNibble maps v onto the 4-bit range [0, 15] biased by 8.
func quantNibble(v, inv float32) byte {
	q := int(v*inv + 8.5)
	if q < 0 {
		q = 0
	}
	if q > 15 {
		q = 15
	}
	return byte(q)
}

// QuantizeQ8_0 packs values into Q8_0 blocks: per 32 elements, an fp16 scale
// followed by 32 int8 quants. len(values) must be a multiple of BlockSize.
func QuantizeQ8_0(values []float32) ([]byte, error) {
	if len(values)%BlockSize != 0 {
		return nil, fmt.Errorf("metalquant: %d values is not a multiple of the %d-element block", len(values), BlockSize)
	}

	out := make([]byte, len(values)/BlockSize*blockBytesQ8_0)
	for b := 0; b*BlockSize < len(values); b++ {
		block := values[b*BlockSize : (b+1)*BlockSize]
		offset := b * blockBytesQ8_0

		var amax float32
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}

		scaleBits := float32ToFloat16(amax / 127)
		scale := float16ToFloat32(scaleBits)
		binary.LittleEndian.PutUint16(out[offset:offset+2], scaleBits)

		var inv float32
		if scale != 0 {
			inv = 1 / scale
		}
		for i, v := range block {
			q := int(math.Round(float64(v * inv)))
			if q < -127 {
				q = -127
			}
			if q > 127 {
				q = 127
			}
			out[offset+2+i] = byte(int8(q))
		}
	}
	return out, nil
}

// DequantizeQ4_0 unpacks a slice of Q4_0 blocks into float32 array.
func DequantizeQ4_0(data []byte, numElements int) ([]float32, error) {
	if numElements%BlockSize != 0 {
		return nil, fmt.Errorf("metalquant: %d elements is not a multiple of the %d-element block", numElements, BlockSize)
	}
	numBlocks := numElements / 32
	if len(data) < numBlocks*18 {
		return nil, fmt.Errorf("metalquant: buffer length %d insufficient for %d Q4_0 blocks", len(data), numBlocks)
	}

	out := make([]float32, numElements)
	for b := 0; b < numBlocks; b++ {
		offset := b * 18
		scaleBits := binary.LittleEndian.Uint16(data[offset : offset+2])
		scale := float16ToFloat32(scaleBits)

		for i := 0; i < 16; i++ {
			byteVal := data[offset+2+i]
			// Low nibble
			q0 := int8(byteVal&0x0F) - 8
			// High nibble
			q1 := int8(byteVal>>4) - 8

			out[b*32+i] = float32(q0) * scale
			out[b*32+16+i] = float32(q1) * scale
		}
	}
	return out, nil
}

// DequantizeQ8_0 unpacks a slice of Q8_0 blocks into float32 array.
func DequantizeQ8_0(data []byte, numElements int) ([]float32, error) {
	if numElements%BlockSize != 0 {
		return nil, fmt.Errorf("metalquant: %d elements is not a multiple of the %d-element block", numElements, BlockSize)
	}
	numBlocks := numElements / 32
	if len(data) < numBlocks*34 {
		return nil, fmt.Errorf("metalquant: buffer length %d insufficient for %d Q8_0 blocks", len(data), numBlocks)
	}

	out := make([]float32, numElements)
	for b := 0; b < numBlocks; b++ {
		offset := b * 34
		scaleBits := binary.LittleEndian.Uint16(data[offset : offset+2])
		scale := float16ToFloat32(scaleBits)

		for i := 0; i < 32; i++ {
			q := int8(data[offset+2+i])
			out[b*32+i] = float32(q) * scale
		}
	}
	return out, nil
}

// float32ToFloat16 converts f to IEEE 754 binary16, rounding to nearest even.
// Values too large for binary16 become infinities and subnormals round toward
// zero, so it is the inverse of float16ToFloat32 on the representable range.
func float32ToFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b >> 16 & 0x8000)
	exp := int32(b>>23&0xFF) - 127
	mant := b & 0x7FFFFF

	switch {
	case exp == 128: // Inf or NaN
		if mant != 0 {
			// Keep the payload non-zero so the result stays a NaN.
			return sign | 0x7C00 | uint16(mant>>13) | 0x0200
		}
		return sign | 0x7C00
	case exp > 15: // overflows binary16
		return sign | 0x7C00
	case exp >= -14: // normal
		half := sign | uint16(exp+15)<<10 | uint16(mant>>13)
		return half + roundBit(mant, 13, uint32(half))
	case exp >= -24: // subnormal
		// Restore the implicit leading one, then shift into place.
		m := mant | 0x800000
		shift := uint(-exp - 14 + 13)
		half := sign | uint16(m>>shift)
		return half + roundBit(m, shift, uint32(half))
	default: // underflows to zero
		return sign
	}
}

// roundBit reports the round-to-nearest-even increment for discarding the low
// shift bits of mant, given the truncated result half.
func roundBit(mant uint32, shift uint, half uint32) uint16 {
	rest := mant & (1<<shift - 1)
	halfway := uint32(1) << (shift - 1)
	if rest > halfway || (rest == halfway && half&1 == 1) {
		return 1
	}
	return 0
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mant := uint32(h & 0x03FF)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03FF
	} else if exp == 0x1F {
		// Inf or NaN
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	}

	exp = exp + (127 - 15)
	mant = mant << 13
	return math.Float32frombits(sign | (exp << 23) | mant)
}
