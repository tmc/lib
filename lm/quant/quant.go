// SPDX-License-Identifier: Apache-2.0

package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// BlockSize is the number of elements every block format quantizes together.
const BlockSize = 32

// Bytes per quantized block: an fp16 scale followed by the packed quants.
const (
	blockBytesQ4_0 = 2 + BlockSize/2
	blockBytesQ8_0 = 2 + BlockSize
)

// QuantFormat specifies the sub-byte weight quantization format.
type QuantFormat string

const (
	FormatQ4_0 QuantFormat = "Q4_0" // 4-bit symmetric with float16 scale per 32 elements
	FormatQ8_0 QuantFormat = "Q8_0" // 8-bit symmetric with float16 scale per 32 elements
	FormatAWQ  QuantFormat = "AWQ"  // 4-bit asymmetric with per-group FP16 scale and zero
)

// BlockQ4_0 is one Q4_0 block: an fp16 scale and [BlockSize] 4-bit nibbles.
type BlockQ4_0 struct {
	Scale  uint16              // float16 scale factor
	Quants [BlockSize / 2]byte // BlockSize nibbles, low half then high half
}

// BlockQ8_0 is one Q8_0 block: an fp16 scale and [BlockSize] int8 quants.
type BlockQ8_0 struct {
	Scale  uint16          // float16 scale factor
	Quants [BlockSize]int8 // BlockSize 8-bit integers
}

// DequantizeQ4_0 unpacks numElements values from Q4_0 blocks. numElements must
// be a multiple of [BlockSize]: a block is the smallest unit that carries a
// scale, so a partial one cannot be decoded.
func DequantizeQ4_0(data []byte, numElements int) ([]float32, error) {
	if numElements < 0 || numElements%BlockSize != 0 {
		return nil, fmt.Errorf("quant: %d elements is not a multiple of the %d-element block", numElements, BlockSize)
	}
	numBlocks := numElements / BlockSize
	if len(data) < numBlocks*blockBytesQ4_0 {
		return nil, fmt.Errorf("quant: buffer length %d insufficient for %d Q4_0 blocks", len(data), numBlocks)
	}

	out := make([]float32, numElements)
	for b := range numBlocks {
		offset := b * blockBytesQ4_0
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(data[offset : offset+2]))

		for i := range BlockSize / 2 {
			byteVal := data[offset+2+i]
			lo := int8(byteVal&0x0F) - 8
			hi := int8(byteVal>>4) - 8

			out[b*BlockSize+i] = float32(lo) * scale
			out[b*BlockSize+BlockSize/2+i] = float32(hi) * scale
		}
	}
	return out, nil
}

// DequantizeQ8_0 unpacks numElements values from Q8_0 blocks. numElements must
// be a multiple of [BlockSize].
func DequantizeQ8_0(data []byte, numElements int) ([]float32, error) {
	if numElements < 0 || numElements%BlockSize != 0 {
		return nil, fmt.Errorf("quant: %d elements is not a multiple of the %d-element block", numElements, BlockSize)
	}
	numBlocks := numElements / BlockSize
	if len(data) < numBlocks*blockBytesQ8_0 {
		return nil, fmt.Errorf("quant: buffer length %d insufficient for %d Q8_0 blocks", len(data), numBlocks)
	}

	out := make([]float32, numElements)
	for b := range numBlocks {
		offset := b * blockBytesQ8_0
		scale := Float16ToFloat32(binary.LittleEndian.Uint16(data[offset : offset+2]))

		for i := range BlockSize {
			out[b*BlockSize+i] = float32(int8(data[offset+2+i])) * scale
		}
	}
	return out, nil
}

// Float16ToFloat32 converts IEEE 754 half-precision bits to float32.
func Float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mant := uint32(h & 0x03FF)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03FF
	} else if exp == 0x1F {
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	}

	exp = exp + (127 - 15)
	mant = mant << 13
	return math.Float32frombits(sign | (exp << 23) | mant)
}
