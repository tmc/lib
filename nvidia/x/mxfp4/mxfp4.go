package mxfp4

import (
	"fmt"
	"math"
)

// DefaultGroupSize is the standard OCP MXFP4 block size.
const DefaultGroupSize = 32

// E2M1Table maps each 4-bit nibble (0..15) to its exact floating-point value.
// Format: 1 sign bit, 2 exponent bits, 1 mantissa bit.
var E2M1Table = [16]float32{
	0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
	-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
}

// QuantizedMatrix holds packed 4-bit weights and per-group scale factors.
type QuantizedMatrix struct {
	Rows      int
	Cols      int
	GroupSize int
	Packed    []byte
	Scales    []float32
}

// Quantize compresses a float32 slice into packed MXFP4 nibbles and scales.
func Quantize(src []float32, groupSize int) ([]byte, []float32, error) {
	if groupSize <= 0 || groupSize%2 != 0 {
		return nil, nil, fmt.Errorf("mxfp4: groupSize must be positive even integer, got %d", groupSize)
	}
	if len(src)%groupSize != 0 {
		return nil, nil, fmt.Errorf("mxfp4: src length (%d) must be multiple of groupSize (%d)", len(src), groupSize)
	}

	numGroups := len(src) / groupSize
	packed := make([]byte, len(src)/2)
	scales := make([]float32, numGroups)

	for g := 0; g < numGroups; g++ {
		groupStart := g * groupSize
		group := src[groupStart : groupStart+groupSize]

		// Find max absolute value in group
		maxAbs := float32(0.0)
		for _, v := range group {
			abs := float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}

		// Scale factor maps maxAbs to 6.0 (maximum representable E2M1 magnitude)
		scale := maxAbs / 6.0
		if scale == 0 {
			scale = 1.0
		}
		scales[g] = scale
		invScale := 1.0 / scale

		// Quantize pairs into single bytes
		packedStart := g * (groupSize / 2)
		for i := 0; i < groupSize; i += 2 {
			v0 := group[i] * invScale
			v1 := group[i+1] * invScale

			n0 := quantizeE2M1(v0)
			n1 := quantizeE2M1(v1)

			packed[packedStart+i/2] = byte(n0 | (n1 << 4))
		}
	}

	return packed, scales, nil
}

// Dequantize unpacks MXFP4 bytes and scales into float32 destination slice.
func Dequantize(packed []byte, scales []float32, groupSize int, dst []float32) error {
	if groupSize <= 0 || groupSize%2 != 0 {
		return fmt.Errorf("mxfp4: groupSize must be positive even integer, got %d", groupSize)
	}
	expectedLen := len(scales) * groupSize
	if len(dst) < expectedLen {
		return fmt.Errorf("mxfp4: dst slice too short (%d < %d)", len(dst), expectedLen)
	}
	if len(packed) < expectedLen/2 {
		return fmt.Errorf("mxfp4: packed slice too short (%d < %d)", len(packed), expectedLen/2)
	}

	for g := 0; g < len(scales); g++ {
		scale := scales[g]
		groupStart := g * groupSize
		packedStart := g * (groupSize / 2)

		for i := 0; i < groupSize/2; i++ {
			b := packed[packedStart+i]
			n0 := b & 0x0F
			n1 := (b >> 4) & 0x0F

			dst[groupStart+i*2] = E2M1Table[n0] * scale
			dst[groupStart+i*2+1] = E2M1Table[n1] * scale
		}
	}

	return nil
}

// quantizeE2M1 finds the nearest 4-bit nibble code for a normalized value.
func quantizeE2M1(val float32) uint8 {
	isNeg := false
	if math.Signbit(float64(val)) {
		isNeg = true
		val = -val
	}

	// Find closest value in positive half (indices 0..7)
	bestIdx := 0
	bestDist := float32(math.Abs(float64(val - E2M1Table[0])))
	for i := 1; i < 8; i++ {
		dist := float32(math.Abs(float64(val - E2M1Table[i])))
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}

	if isNeg {
		bestIdx += 8
	}
	return uint8(bestIdx)
}

// QuantizeMatrix quantizes a 2D row-major float32 matrix.
func QuantizeMatrix(rows, cols int, src []float32, groupSize int) (*QuantizedMatrix, error) {
	if len(src) != rows*cols {
		return nil, fmt.Errorf("mxfp4: src length (%d) != rows*cols (%d*%d)", len(src), rows, cols)
	}
	packed, scales, err := Quantize(src, groupSize)
	if err != nil {
		return nil, err
	}
	return &QuantizedMatrix{
		Rows:      rows,
		Cols:      cols,
		GroupSize: groupSize,
		Packed:    packed,
		Scales:    scales,
	}, nil
}
