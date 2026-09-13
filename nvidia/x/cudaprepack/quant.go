// SPDX-License-Identifier: Apache-2.0

package cudaprepack

import (
	"encoding/binary"
	"fmt"
	"math"
)

// OCP Microscaling FP4 (E2M1) lookup table.
// Nibble values 0..7 are positive, 8..15 are negative.
var e2m1Table = [16]float32{
	0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
	-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
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

// Float32ToFloat16 converts float32 to IEEE 754 half-precision bits.
func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x007FFFFF

	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant = (mant | 0x00800000) >> (1 - exp)
		return sign | uint16((mant+0x00001000)>>13)
	} else if exp >= 0x1F {
		return sign | 0x7C00
	}
	return sign | uint16(exp<<10) | uint16((mant+0x00001000)>>13)
}

// Float32ToE4M3 converts float32 to FP8 E4M3FN format (1 sign, 4 exp bias 7, 3 mantissa).
func Float32ToE4M3(v float32) byte {
	if math.IsNaN(float64(v)) {
		return 0x7F
	}
	bits := math.Float32bits(v)
	sign := byte((bits >> 24) & 0x80)
	absVal := float32(math.Abs(float64(v)))

	if absVal > 448.0 {
		return sign | 0x7E // saturate to max finite
	}
	if absVal < 0.001953125 { // min subnormal 2^-9
		return sign
	}

	exp := int((bits>>23)&0xFF) - 127
	mant := bits & 0x007FFFFF

	if exp < -6 {
		// Subnormal in E4M3 (exp field = 0)
		shift := -6 - exp
		val := (mant | 0x00800000) >> (shift + 20)
		return sign | byte(val&0x07)
	}

	expE4M3 := exp + 7
	if expE4M3 > 15 {
		return sign | 0x7E
	}
	mantE4M3 := byte((mant >> 20) & 0x07)
	return sign | byte(expE4M3<<3) | mantE4M3
}

// E4M3ToFloat32 converts FP8 E4M3FN to float32.
func E4M3ToFloat32(b byte) float32 {
	sign := float32(1.0)
	if (b & 0x80) != 0 {
		sign = -1.0
	}
	exp := (b >> 3) & 0x0F
	mant := b & 0x07

	if exp == 0x0F && mant == 0x07 {
		return float32(math.NaN())
	}
	if exp == 0 {
		if mant == 0 {
			return sign * 0.0
		}
		// Subnormal: 2^-6 * (mant / 8)
		return sign * float32(math.Pow(2, -6)) * (float32(mant) / 8.0)
	}
	// Normal: 2^(exp - 7) * (1 + mant / 8)
	return sign * float32(math.Pow(2, float64(int(exp)-7))) * (1.0 + float32(mant)/8.0)
}

// Float32ToE5M2 converts float32 to FP8 E5M2 format (1 sign, 5 exp bias 15, 2 mantissa).
func Float32ToE5M2(v float32) byte {
	if math.IsNaN(float64(v)) {
		return 0x7F
	}
	bits := math.Float32bits(v)
	sign := byte((bits >> 24) & 0x80)
	absVal := float32(math.Abs(float64(v)))

	if math.IsInf(float64(v), 0) || absVal > 57344.0 {
		return sign | 0x7C // Inf
	}
	if absVal < 0.0000152587890625 { // min subnormal 2^-16
		return sign
	}

	exp := int((bits>>23)&0xFF) - 127
	mant := bits & 0x007FFFFF

	if exp < -14 {
		// Subnormal
		shift := -14 - exp
		val := (mant | 0x00800000) >> (shift + 21)
		return sign | byte(val&0x03)
	}

	expE5M2 := exp + 15
	if expE5M2 >= 31 {
		return sign | 0x7C
	}
	mantE5M2 := byte((mant >> 21) & 0x03)
	return sign | byte(expE5M2<<2) | mantE5M2
}

// E5M2ToFloat32 converts FP8 E5M2 to float32.
func E5M2ToFloat32(b byte) float32 {
	sign := float32(1.0)
	if (b & 0x80) != 0 {
		sign = -1.0
	}
	exp := (b >> 2) & 0x1F
	mant := b & 0x03

	if exp == 0x1F {
		if mant == 0 {
			return float32(math.Inf(int(sign)))
		}
		return float32(math.NaN())
	}
	if exp == 0 {
		if mant == 0 {
			return sign * 0.0
		}
		// Subnormal: 2^-14 * (mant / 4)
		return sign * float32(math.Pow(2, -14)) * (float32(mant) / 4.0)
	}
	// Normal: 2^(exp - 15) * (1 + mant / 4)
	return sign * float32(math.Pow(2, float64(int(exp)-15))) * (1.0 + float32(mant)/4.0)
}

func quantizeE2M1(v float32) byte {
	if v >= 5.0 {
		return 7 // 6.0
	} else if v >= 3.5 {
		return 6 // 4.0
	} else if v >= 2.5 {
		return 5 // 3.0
	} else if v >= 1.75 {
		return 4 // 2.0
	} else if v >= 1.25 {
		return 3 // 1.5
	} else if v >= 0.75 {
		return 2 // 1.0
	} else if v >= 0.25 {
		return 1 // 0.5
	} else if v >= 0.0 {
		return 0 // 0.0
	} else if v >= -0.25 {
		return 8 // -0.0
	} else if v >= -0.75 {
		return 9 // -0.5
	} else if v >= -1.25 {
		return 10 // -1.0
	} else if v >= -1.75 {
		return 11 // -1.5
	} else if v >= -2.5 {
		return 12 // -2.0
	} else if v >= -3.5 {
		return 13 // -3.0
	} else if v >= -5.0 {
		return 14 // -4.0
	}
	return 15 // -6.0
}

// PackMXFP4 quantizes float32 data into OCP MXFP4 format and repacks into contiguous
// hardware tiles conforming to sm_121a 16-byte alignment and 128-byte box width limits.
func PackMXFP4(src []float32, rows, cols int, groupSize int) (*PrepackedTensor, error) {
	if groupSize <= 0 || groupSize%32 != 0 {
		return nil, fmt.Errorf("cudaprepack: groupSize must be a multiple of 32, got %d", groupSize)
	}
	if cols%groupSize != 0 {
		return nil, fmt.Errorf("cudaprepack: cols (%d) must be a multiple of groupSize (%d)", cols, groupSize)
	}
	if len(src) < rows*cols {
		return nil, fmt.Errorf("cudaprepack: source buffer too short")
	}

	numGroupsPerRow := cols / groupSize
	totalGroups := rows * numGroupsPerRow

	// 32 elements = 16 bytes payload
	bytesPerGroup := groupSize / 2
	rawRowBytes := numGroupsPerRow * bytesPerGroup
	paddedRowStride := ((rawRowBytes + MinAddressAlignment - 1) / MinAddressAlignment) * MinAddressAlignment

	payload := make([]byte, rows*paddedRowStride)
	scales := make([]byte, totalGroups*2) // Float16 scale per group

	for r := 0; r < rows; r++ {
		for g := 0; g < numGroupsPerRow; g++ {
			elemStart := r*cols + g*groupSize
			group := src[elemStart : elemStart+groupSize]

			maxAbs := float32(0.0)
			for _, v := range group {
				abs := float32(math.Abs(float64(v)))
				if abs > maxAbs {
					maxAbs = abs
				}
			}

			scale := maxAbs / 6.0
			if scale == 0 {
				scale = 1.0
			}
			invScale := 1.0 / scale

			scaleH := Float32ToFloat16(scale)
			scaleIdx := (r*numGroupsPerRow + g) * 2
			binary.LittleEndian.PutUint16(scales[scaleIdx:], scaleH)

			dstStart := r*paddedRowStride + g*bytesPerGroup
			for i := 0; i < groupSize; i += 2 {
				n0 := quantizeE2M1(group[i] * invScale)
				n1 := quantizeE2M1(group[i+1] * invScale)
				payload[dstStart+i/2] = n0 | (n1 << 4)
			}
		}
	}

	boxWidthBytes := 128
	if paddedRowStride < 128 {
		boxWidthBytes = paddedRowStride
	}
	box0 := uint32(boxWidthBytes * 2) // 2 nibbles per byte
	box1 := uint32(32)
	if box1 > uint32(rows) {
		box1 = uint32(rows)
	}

	desc := BoxDescriptor{
		Rank:           2,
		DataType:       TypeMXFP4,
		GlobalDims:     [3]uint64{uint64(cols), uint64(rows), 1},
		GlobalStrides:  [3]int64{int64(paddedRowStride), 0, 0},
		BoxDims:        [3]uint32{box0, box1, 1},
		ElementStrides: [3]uint32{1, 1, 1},
		Swizzle:        SwizzleNone,
		Interleave:     InterleaveNone,
	}

	return &PrepackedTensor{
		Arch:    ArchSM121a,
		Type:    TypeMXFP4,
		BoxDesc: desc,
		Payload: payload,
		Scales:  scales,
	}, nil
}

// DequantizeMXFP4 unpacks pre-packed MXFP4 tiles and scales into dst.
func DequantizeMXFP4(p *PrepackedTensor, groupSize int, dst []float32) error {
	if p.Type != TypeMXFP4 {
		return fmt.Errorf("cudaprepack: tensor type is %s, want MXFP4", p.Type)
	}
	cols := int(p.BoxDesc.GlobalDims[0])
	rows := int(p.BoxDesc.GlobalDims[1])
	paddedRowStride := int(p.BoxDesc.GlobalStrides[0])
	numGroupsPerRow := cols / groupSize
	bytesPerGroup := groupSize / 2

	if len(dst) < rows*cols {
		return fmt.Errorf("cudaprepack: destination slice too short")
	}

	for r := 0; r < rows; r++ {
		for g := 0; g < numGroupsPerRow; g++ {
			scaleIdx := (r*numGroupsPerRow + g) * 2
			scaleH := binary.LittleEndian.Uint16(p.Scales[scaleIdx:])
			scale := Float16ToFloat32(scaleH)

			srcStart := r*paddedRowStride + g*bytesPerGroup
			dstStart := r*cols + g*groupSize

			for i := 0; i < groupSize; i += 2 {
				b := p.Payload[srcStart+i/2]
				n0 := b & 0x0F
				n1 := (b >> 4) & 0x0F
				dst[dstStart+i] = e2m1Table[n0] * scale
				dst[dstStart+i+1] = e2m1Table[n1] * scale
			}
		}
	}
	return nil
}

// RepackQ8_0 repacks standard 34-byte Q8_0 blocks (2B scale + 32B int8) into contiguous
// 128-byte hardware tiles and an aligned scale buffer for sm_121a TMA bulk streaming.
func RepackQ8_0(rawBlocks []byte, rows, cols int) (*PrepackedTensor, error) {
	if cols%32 != 0 {
		return nil, fmt.Errorf("cudaprepack: cols (%d) must be a multiple of 32", cols)
	}
	numBlocksPerRow := cols / 32
	totalBlocks := rows * numBlocksPerRow
	if len(rawBlocks) < totalBlocks*34 {
		return nil, fmt.Errorf("cudaprepack: rawBlocks buffer too short (got %d, want %d)", len(rawBlocks), totalBlocks*34)
	}

	rawRowBytes := numBlocksPerRow * 32
	paddedRowStride := ((rawRowBytes + MinAddressAlignment - 1) / MinAddressAlignment) * MinAddressAlignment

	payload := make([]byte, rows*paddedRowStride)
	scales := make([]byte, totalBlocks*2)

	for r := 0; r < rows; r++ {
		for b := 0; b < numBlocksPerRow; b++ {
			blkIdx := r*numBlocksPerRow + b
			rawOffset := blkIdx * 34

			// Copy 2B float16 scale
			scales[blkIdx*2] = rawBlocks[rawOffset]
			scales[blkIdx*2+1] = rawBlocks[rawOffset+1]

			// Copy 32B int8 payload
			dstOffset := r*paddedRowStride + b*32
			copy(payload[dstOffset:dstOffset+32], rawBlocks[rawOffset+2:rawOffset+34])
		}
	}

	boxWidthBytes := 128
	if paddedRowStride < 128 {
		boxWidthBytes = paddedRowStride
	}
	box0 := uint32(boxWidthBytes) // 1 byte per element
	box1 := uint32(32)
	if box1 > uint32(rows) {
		box1 = uint32(rows)
	}

	desc := BoxDescriptor{
		Rank:           2,
		DataType:       TypeQ8_0,
		GlobalDims:     [3]uint64{uint64(cols), uint64(rows), 1},
		GlobalStrides:  [3]int64{int64(paddedRowStride), 0, 0},
		BoxDims:        [3]uint32{box0, box1, 1},
		ElementStrides: [3]uint32{1, 1, 1},
		Swizzle:        SwizzleNone,
		Interleave:     InterleaveNone,
	}

	return &PrepackedTensor{
		Arch:    ArchSM121a,
		Type:    TypeQ8_0,
		BoxDesc: desc,
		Payload: payload,
		Scales:  scales,
	}, nil
}

// UnpackQ8_0 converts pre-packed Q8_0 tiles back into standard 34-byte interleaved Q8_0 blocks.
func UnpackQ8_0(p *PrepackedTensor) ([]byte, error) {
	if p.Type != TypeQ8_0 {
		return nil, fmt.Errorf("cudaprepack: tensor type is %s, want Q8_0", p.Type)
	}
	cols := int(p.BoxDesc.GlobalDims[0])
	rows := int(p.BoxDesc.GlobalDims[1])
	paddedRowStride := int(p.BoxDesc.GlobalStrides[0])
	numBlocksPerRow := cols / 32
	totalBlocks := rows * numBlocksPerRow

	out := make([]byte, totalBlocks*34)
	for r := 0; r < rows; r++ {
		for b := 0; b < numBlocksPerRow; b++ {
			blkIdx := r*numBlocksPerRow + b
			rawOffset := blkIdx * 34

			// Copy scale
			out[rawOffset] = p.Scales[blkIdx*2]
			out[rawOffset+1] = p.Scales[blkIdx*2+1]

			// Copy quants
			srcOffset := r*paddedRowStride + b*32
			copy(out[rawOffset+2:rawOffset+34], p.Payload[srcOffset:srcOffset+32])
		}
	}
	return out, nil
}

// RepackIQ2_XXS repacks standard 66-byte IQ2_XXS super-blocks (2B scale + 64B grid payload)
// into contiguous 128-byte hardware tiles for sm_121a TMA bulk streaming.
func RepackIQ2_XXS(rawBlocks []byte, rows, cols int) (*PrepackedTensor, error) {
	if cols%256 != 0 {
		return nil, fmt.Errorf("cudaprepack: cols (%d) must be a multiple of 256 for IQ2_XXS", cols)
	}
	numBlocksPerRow := cols / 256
	totalBlocks := rows * numBlocksPerRow
	if len(rawBlocks) < totalBlocks*66 {
		return nil, fmt.Errorf("cudaprepack: rawBlocks too short (got %d, want %d)", len(rawBlocks), totalBlocks*66)
	}

	rawRowBytes := numBlocksPerRow * 64
	paddedRowStride := ((rawRowBytes + MinAddressAlignment - 1) / MinAddressAlignment) * MinAddressAlignment

	payload := make([]byte, rows*paddedRowStride)
	scales := make([]byte, totalBlocks*2)

	for r := 0; r < rows; r++ {
		for b := 0; b < numBlocksPerRow; b++ {
			blkIdx := r*numBlocksPerRow + b
			rawOffset := blkIdx * 66

			// Extract 2B scale d
			scales[blkIdx*2] = rawBlocks[rawOffset]
			scales[blkIdx*2+1] = rawBlocks[rawOffset+1]

			// Extract 64B qs
			dstOffset := r*paddedRowStride + b*64
			copy(payload[dstOffset:dstOffset+64], rawBlocks[rawOffset+2:rawOffset+66])
		}
	}

	boxWidthBytes := 128
	if paddedRowStride < 128 {
		boxWidthBytes = paddedRowStride
	}
	box0 := uint32(boxWidthBytes * 4) // 4 2-bit quants per byte = 512 elements in 128B
	box1 := uint32(16)
	if box1 > uint32(rows) {
		box1 = uint32(rows)
	}

	desc := BoxDescriptor{
		Rank:           2,
		DataType:       TypeIQ2_XXS,
		GlobalDims:     [3]uint64{uint64(cols), uint64(rows), 1},
		GlobalStrides:  [3]int64{int64(paddedRowStride), 0, 0},
		BoxDims:        [3]uint32{box0, box1, 1},
		ElementStrides: [3]uint32{1, 1, 1},
		Swizzle:        SwizzleNone,
		Interleave:     InterleaveNone,
	}

	return &PrepackedTensor{
		Arch:    ArchSM121a,
		Type:    TypeIQ2_XXS,
		BoxDesc: desc,
		Payload: payload,
		Scales:  scales,
	}, nil
}

// UnpackIQ2_XXS unpacks pre-packed IQ2_XXS back to standard 66-byte blocks.
func UnpackIQ2_XXS(p *PrepackedTensor) ([]byte, error) {
	if p.Type != TypeIQ2_XXS {
		return nil, fmt.Errorf("cudaprepack: tensor type is %s, want IQ2_XXS", p.Type)
	}
	cols := int(p.BoxDesc.GlobalDims[0])
	rows := int(p.BoxDesc.GlobalDims[1])
	paddedRowStride := int(p.BoxDesc.GlobalStrides[0])
	numBlocksPerRow := cols / 256
	totalBlocks := rows * numBlocksPerRow

	out := make([]byte, totalBlocks*66)
	for r := 0; r < rows; r++ {
		for b := 0; b < numBlocksPerRow; b++ {
			blkIdx := r*numBlocksPerRow + b
			rawOffset := blkIdx * 66

			// Scale d
			out[rawOffset] = p.Scales[blkIdx*2]
			out[rawOffset+1] = p.Scales[blkIdx*2+1]

			// Payload qs
			srcOffset := r*paddedRowStride + b*64
			copy(out[rawOffset+2:rawOffset+66], p.Payload[srcOffset:srcOffset+64])
		}
	}
	return out, nil
}

// QuantizeFP8 converts float32 matrix into FP8 E4M3 or E5M2 and formats into contiguous TMA tiles.
func QuantizeFP8(src []float32, rows, cols int, dt DataType) (*PrepackedTensor, error) {
	if dt != TypeFP8_E4M3 && dt != TypeFP8_E5M2 {
		return nil, fmt.Errorf("cudaprepack: unsupported FP8 type %s", dt)
	}
	if len(src) < rows*cols {
		return nil, fmt.Errorf("cudaprepack: source buffer too short")
	}

	rawRowBytes := cols // 1 byte per element
	paddedRowStride := ((rawRowBytes + MinAddressAlignment - 1) / MinAddressAlignment) * MinAddressAlignment
	payload := make([]byte, rows*paddedRowStride)

	for r := 0; r < rows; r++ {
		dstOffset := r * paddedRowStride
		srcOffset := r * cols
		for c := 0; c < cols; c++ {
			if dt == TypeFP8_E4M3 {
				payload[dstOffset+c] = Float32ToE4M3(src[srcOffset+c])
			} else {
				payload[dstOffset+c] = Float32ToE5M2(src[srcOffset+c])
			}
		}
	}

	boxWidthBytes := 128
	if paddedRowStride < 128 {
		boxWidthBytes = paddedRowStride
	}
	box0 := uint32(boxWidthBytes)
	box1 := uint32(32)
	if box1 > uint32(rows) {
		box1 = uint32(rows)
	}

	desc := BoxDescriptor{
		Rank:           2,
		DataType:       dt,
		GlobalDims:     [3]uint64{uint64(cols), uint64(rows), 1},
		GlobalStrides:  [3]int64{int64(paddedRowStride), 0, 0},
		BoxDims:        [3]uint32{box0, box1, 1},
		ElementStrides: [3]uint32{1, 1, 1},
		Swizzle:        SwizzleNone,
		Interleave:     InterleaveNone,
	}

	return &PrepackedTensor{
		Arch:    ArchSM121a,
		Type:    dt,
		BoxDesc: desc,
		Payload: payload,
	}, nil
}

// DequantizeFP8 converts prepacked FP8 matrix back to float32.
func DequantizeFP8(p *PrepackedTensor, dst []float32) error {
	if p.Type != TypeFP8_E4M3 && p.Type != TypeFP8_E5M2 {
		return fmt.Errorf("cudaprepack: tensor type is %s, want FP8", p.Type)
	}
	cols := int(p.BoxDesc.GlobalDims[0])
	rows := int(p.BoxDesc.GlobalDims[1])
	paddedRowStride := int(p.BoxDesc.GlobalStrides[0])

	if len(dst) < rows*cols {
		return fmt.Errorf("cudaprepack: destination slice too short")
	}

	for r := 0; r < rows; r++ {
		srcOffset := r * paddedRowStride
		dstOffset := r * cols
		for c := 0; c < cols; c++ {
			b := p.Payload[srcOffset+c]
			if p.Type == TypeFP8_E4M3 {
				dst[dstOffset+c] = E4M3ToFloat32(b)
			} else {
				dst[dstOffset+c] = E5M2ToFloat32(b)
			}
		}
	}
	return nil
}
