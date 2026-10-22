// SPDX-License-Identifier: Apache-2.0

package mxfp4

import (
	"fmt"
	"math"
	"sort"
)

const (
	// NVFP4BlockElements is the micro-block size (16 E2M1 elements) used by
	// Blackwell NVFP4 and DeepInfer's V41_FP4_FORMAT KV cache.
	NVFP4BlockElements = 16
	// NVFP4PackedBytes is the packed E2M1 byte count per 16-element micro-block.
	NVFP4PackedBytes = 8
	// NVFP4ScaleBytes is the FP8 E4M3 scale byte count per 16-element micro-block.
	NVFP4ScaleBytes = 1
)

const nvfp4E4M3Max = 448.0

var nvfp4E4M3PosTable = func() [127]float64 {
	var t [127]float64
	for code := 0; code < 127; code++ {
		exp := (code >> 3) & 0x0F
		mant := code & 0x07
		if exp == 0 {
			t[code] = float64(mant) * (1.0 / 512.0)
		} else {
			t[code] = math.Ldexp(1.0+float64(mant)/8.0, exp-7)
		}
	}
	return t
}()

func encodeE4M3Scale(scale float32) uint8 {
	v := float64(scale)
	if v <= 0 {
		return 0
	}
	if v >= nvfp4E4M3Max {
		return 0x7E
	}
	hi := sort.Search(len(nvfp4E4M3PosTable), func(i int) bool {
		return nvfp4E4M3PosTable[i] >= v
	})
	if hi <= 1 {
		// Clamp tiny positive scales to the minimum positive E4M3 subnormal (2^-9)
		// matching DeepInfer's clamp_min(FLOAT8E4M3_MIN_POSITIVE * FLOAT4_E2M1_MAX).
		return 0x01
	}
	lo := hi - 1
	dl := v - nvfp4E4M3PosTable[lo]
	du := nvfp4E4M3PosTable[hi] - v
	if dl < du || (dl == du && lo%2 == 0) {
		return uint8(lo)
	}
	return uint8(hi)
}

func decodeE4M3Scale(code uint8) (float32, error) {
	mag := code & 0x7F
	if mag == 0x7F {
		return 0, fmt.Errorf("mxfp4: NaN E4M3 scale code %#02x", code)
	}
	if code&0x80 != 0 && mag != 0 {
		return 0, fmt.Errorf("mxfp4: negative E4M3 scale code %#02x", code)
	}
	return float32(nvfp4E4M3PosTable[mag]), nil
}

// PackE2M1WithE4M3Scale quantizes src (whose length must be a positive multiple
// of 16) into 8 packed E2M1 bytes and 1 unsigned FP8 E4M3 scale byte per
// 16-element micro-block (scale = amax / 6.0 encoded in FP8 E4M3).
func PackE2M1WithE4M3Scale(src []float32) (packedData []byte, e4m3Scales []uint8, err error) {
	if len(src) == 0 || len(src)%NVFP4BlockElements != 0 {
		return nil, nil, fmt.Errorf("mxfp4: src length (%d) must be a positive multiple of %d", len(src), NVFP4BlockElements)
	}
	numBlocks := len(src) / NVFP4BlockElements
	packedData = make([]byte, numBlocks*NVFP4PackedBytes)
	e4m3Scales = make([]uint8, numBlocks*NVFP4ScaleBytes)

	for b := 0; b < numBlocks; b++ {
		blockStart := b * NVFP4BlockElements
		block := src[blockStart : blockStart+NVFP4BlockElements]

		amax := float32(0)
		for i, v := range block {
			abs := float32(math.Abs(float64(v)))
			if math.IsNaN(float64(abs)) || math.IsInf(float64(abs), 0) {
				return nil, nil, fmt.Errorf("mxfp4: src[%d] is %v: NVFP4 cannot represent non-finite values", blockStart+i, v)
			}
			if abs > amax {
				amax = abs
			}
		}

		if amax == 0 {
			e4m3Scales[b] = 0
			continue
		}

		scaleCode := encodeE4M3Scale(amax / e2m1MaxMagnitude)
		e4m3Scales[b] = scaleCode
		scaleVal := float32(nvfp4E4M3PosTable[scaleCode])
		invScale := 1.0 / scaleVal

		packedStart := b * NVFP4PackedBytes
		for i := 0; i < NVFP4BlockElements; i += 2 {
			n0 := quantizeE2M1(block[i] * invScale)
			n1 := quantizeE2M1(block[i+1] * invScale)
			packedData[packedStart+i/2] = byte(n0 | (n1 << 4))
		}
	}
	return packedData, e4m3Scales, nil
}

// UnpackE2M1WithE4M3Scale unpacks count float32 elements (a positive multiple
// of 16) from packed E2M1 nibbles and per-16 FP8 E4M3 scale factors.
func UnpackE2M1WithE4M3Scale(packedData []byte, e4m3Scales []uint8, count int) ([]float32, error) {
	if count <= 0 || count%NVFP4BlockElements != 0 {
		return nil, fmt.Errorf("mxfp4: count (%d) must be a positive multiple of %d", count, NVFP4BlockElements)
	}
	numBlocks := count / NVFP4BlockElements
	if len(packedData) != numBlocks*NVFP4PackedBytes {
		return nil, fmt.Errorf("mxfp4: packedData length (%d), want %d", len(packedData), numBlocks*NVFP4PackedBytes)
	}
	if len(e4m3Scales) != numBlocks*NVFP4ScaleBytes {
		return nil, fmt.Errorf("mxfp4: e4m3Scales length (%d), want %d", len(e4m3Scales), numBlocks*NVFP4ScaleBytes)
	}

	out := make([]float32, count)
	for b := 0; b < numBlocks; b++ {
		scale, err := decodeE4M3Scale(e4m3Scales[b])
		if err != nil {
			return nil, fmt.Errorf("mxfp4: block %d: %w", b, err)
		}
		packedStart := b * NVFP4PackedBytes
		dstStart := b * NVFP4BlockElements
		for i := 0; i < NVFP4PackedBytes; i++ {
			packedByte := packedData[packedStart+i]
			n0 := packedByte & 0x0F
			n1 := (packedByte >> 4) & 0x0F
			out[dstStart+i*2] = E2M1Table[n0] * scale
			out[dstStart+i*2+1] = E2M1Table[n1] * scale
		}
	}
	return out, nil
}
