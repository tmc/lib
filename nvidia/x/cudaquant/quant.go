// SPDX-License-Identifier: Apache-2.0

package cudaquant

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

// DequantizeQ4_0 unpacks a slice of Q4_0 blocks into float32 array.
func DequantizeQ4_0(data []byte, numElements int) ([]float32, error) {
	numBlocks := numElements / 32
	if len(data) < numBlocks*18 {
		return nil, fmt.Errorf("cudaquant: buffer length %d insufficient for %d Q4_0 blocks", len(data), numBlocks)
	}

	out := make([]float32, numElements)
	for b := 0; b < numBlocks; b++ {
		offset := b * 18
		scaleBits := binary.LittleEndian.Uint16(data[offset : offset+2])
		scale := float16ToFloat32(scaleBits)

		for i := 0; i < 16; i++ {
			byteVal := data[offset+2+i]
			q0 := int8(byteVal&0x0F) - 8
			q1 := int8(byteVal>>4) - 8

			out[b*32+i] = float32(q0) * scale
			out[b*32+16+i] = float32(q1) * scale
		}
	}
	return out, nil
}

// DequantizeQ8_0 unpacks a slice of Q8_0 blocks into float32 array.
func DequantizeQ8_0(data []byte, numElements int) ([]float32, error) {
	numBlocks := numElements / 32
	if len(data) < numBlocks*34 {
		return nil, fmt.Errorf("cudaquant: buffer length %d insufficient for %d Q8_0 blocks", len(data), numBlocks)
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

func float16ToFloat32(h uint16) float32 {
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
