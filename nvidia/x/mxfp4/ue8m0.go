// SPDX-License-Identifier: Apache-2.0

package mxfp4

import (
	"fmt"
	"math"
)

// UE8M0 represents an OCP Microscaling unsigned 8-bit biased power-of-two
// scale exponent (2^(int(u)-127)). Code 0xFF (255) encodes NaN.
type UE8M0 uint8

// OCPBlockBytes is the byte width of one conformant 32-element OCP MXFP4 block:
// 16 bytes of packed E2M1 nibbles followed by 1 byte of UE8M0 scale.
const OCPBlockBytes = 17

const e2m1MaxMagnitude = 6.0

// Float32ToUE8M0 converts a float32 scale factor v into a UE8M0 power-of-two
// exponent by rounding up (ceil(log2(v)) + 127). Non-finite values (NaN or Inf)
// map to 0xFF (NaN); non-positive values clamp to 0 (2^-127).
func Float32ToUE8M0(v float32) UE8M0 {
	f := float64(v)
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return UE8M0(0xFF)
	}
	if f <= 0 {
		return UE8M0(0)
	}
	exp := int(math.Ceil(math.Log2(f))) + 127
	if exp < 0 {
		return UE8M0(0)
	}
	if exp > 254 {
		return UE8M0(254)
	}
	return UE8M0(exp)
}

// Float32 returns the float32 power-of-two value 2^(int(u)-127) represented by u,
// or NaN when u is 0xFF.
func (u UE8M0) Float32() float32 {
	if u.IsNaN() {
		return float32(math.NaN())
	}
	return float32(math.Ldexp(1.0, int(u)-127))
}

// IsNaN reports whether u encodes the OCP UE8M0 NaN sentinel (0xFF).
func (u UE8M0) IsNaN() bool {
	return u == 0xFF
}

// PackOCPBlocks quantizes src (whose length must be a positive multiple of 32)
// into conformant 17-byte OCP MXFP4 blocks: 16 bytes of 32 packed E2M1 nibbles
// followed by 1 byte of UE8M0 scale (ceil(log2(amax / 6.0))).
func PackOCPBlocks(src []float32) ([]byte, error) {
	if len(src) == 0 || len(src)%DefaultGroupSize != 0 {
		return nil, fmt.Errorf("mxfp4: src length (%d) must be a positive multiple of %d", len(src), DefaultGroupSize)
	}
	numBlocks := len(src) / DefaultGroupSize
	out := make([]byte, numBlocks*OCPBlockBytes)

	for b := 0; b < numBlocks; b++ {
		groupStart := b * DefaultGroupSize
		group := src[groupStart : groupStart+DefaultGroupSize]

		amax := float32(0)
		for i, v := range group {
			abs := float32(math.Abs(float64(v)))
			if math.IsNaN(float64(abs)) || math.IsInf(float64(abs), 0) {
				return nil, fmt.Errorf("mxfp4: src[%d] is %v: E2M1 cannot represent non-finite values", groupStart+i, v)
			}
			if abs > amax {
				amax = abs
			}
		}

		var scaleExp UE8M0
		if amax == 0 {
			scaleExp = UE8M0(127)
		} else {
			scaleExp = Float32ToUE8M0(amax / e2m1MaxMagnitude)
		}
		scaleVal := scaleExp.Float32()
		invScale := 1.0 / scaleVal

		blockOffset := b * OCPBlockBytes
		for i := 0; i < DefaultGroupSize; i += 2 {
			n0 := quantizeE2M1(group[i] * invScale)
			n1 := quantizeE2M1(group[i+1] * invScale)
			out[blockOffset+i/2] = byte(n0 | (n1 << 4))
		}
		out[blockOffset+16] = byte(scaleExp)
	}
	return out, nil
}

// UnpackOCPBlocks unpacks count float32 elements (a positive multiple of 32)
// from 17-byte OCP MXFP4 blocks.
func UnpackOCPBlocks(data []byte, count int) ([]float32, error) {
	if count <= 0 || count%DefaultGroupSize != 0 {
		return nil, fmt.Errorf("mxfp4: count (%d) must be a positive multiple of %d", count, DefaultGroupSize)
	}
	numBlocks := count / DefaultGroupSize
	if len(data) != numBlocks*OCPBlockBytes {
		return nil, fmt.Errorf("mxfp4: data length (%d) does not match %d OCP blocks (%d bytes)", len(data), numBlocks, numBlocks*OCPBlockBytes)
	}

	out := make([]float32, count)
	for b := 0; b < numBlocks; b++ {
		blockOffset := b * OCPBlockBytes
		scaleExp := UE8M0(data[blockOffset+16])
		if scaleExp.IsNaN() {
			return nil, fmt.Errorf("mxfp4: block %d has NaN UE8M0 scale (0xff)", b)
		}
		scale := scaleExp.Float32()
		dstOffset := b * DefaultGroupSize
		for i := 0; i < 16; i++ {
			packedByte := data[blockOffset+i]
			n0 := packedByte & 0x0F
			n1 := (packedByte >> 4) & 0x0F
			out[dstOffset+i*2] = E2M1Table[n0] * scale
			out[dstOffset+i*2+1] = E2M1Table[n1] * scale
		}
	}
	return out, nil
}
