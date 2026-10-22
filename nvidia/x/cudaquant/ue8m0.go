// SPDX-License-Identifier: Apache-2.0

package cudaquant

import (
	"fmt"
	"math"
	"sort"
)

// ScaleRecipe selects the block scale representation used by per-token FP8
// quantization kernels.
type ScaleRecipe uint8

const (
	// ScaleRecipeFloat32 stores one IEEE-754 binary32 scale per quantization block.
	ScaleRecipeFloat32 ScaleRecipe = iota
	// ScaleRecipeUE8M0 stores one unsigned 8-bit biased power-of-two exponent
	// (2^(exp-127)) per quantization block, packed four per uint32 word.
	ScaleRecipeUE8M0
)

// String returns the canonical name of r.
func (r ScaleRecipe) String() string {
	switch r {
	case ScaleRecipeFloat32:
		return "float32"
	case ScaleRecipeUE8M0:
		return "ue8m0"
	default:
		return fmt.Sprintf("ScaleRecipe(%d)", uint8(r))
	}
}

const fp8E4M3Max = 448.0

var fp8E4M3PosTable = func() [127]float64 {
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

func float32ToFP8E4M3(x float32) uint8 {
	sign := uint8(0)
	v := float64(x)
	if math.Signbit(v) {
		sign = 0x80
		v = -v
	}
	if math.IsNaN(v) {
		return sign | 0x7F
	}
	if v >= fp8E4M3Max {
		return sign | 0x7E
	}
	if v == 0 {
		return sign
	}
	hi := sort.Search(len(fp8E4M3PosTable), func(i int) bool {
		return fp8E4M3PosTable[i] >= v
	})
	if hi == 0 {
		return sign
	}
	lo := hi - 1
	dl := v - fp8E4M3PosTable[lo]
	du := fp8E4M3PosTable[hi] - v
	if dl < du || (dl == du && lo%2 == 0) {
		return sign | uint8(lo)
	}
	return sign | uint8(hi)
}

func fp8E4M3ToFloat32(b uint8) (float32, bool) {
	mag := b & 0x7F
	if mag == 0x7F {
		return float32(math.NaN()), false
	}
	val := float32(fp8E4M3PosTable[mag])
	if b&0x80 != 0 {
		val = -val
	}
	return val, true
}

// QuantizeScaleUE8M0 computes the unsigned 8-bit biased power-of-two scale
// exponent for a block maximum magnitude amax and target format maximum fp8Max:
//
//	ceil(log2(max(amax, 1e-12) / fp8Max)) + 127
//
// clamped to [0, 254]. When fp8Max is non-positive or non-finite, 448.0 is used.
func QuantizeScaleUE8M0(amax float32, fp8Max float32) uint8 {
	maxVal := float64(fp8Max)
	if maxVal <= 0 || math.IsNaN(maxVal) || math.IsInf(maxVal, 0) {
		maxVal = fp8E4M3Max
	}
	a := math.Abs(float64(amax))
	if math.IsNaN(a) {
		a = 1e-12
	} else if math.IsInf(a, 1) {
		return 254
	}
	if a < 1e-12 {
		a = 1e-12
	}
	ratio := float32(a / maxVal)
	if ratio <= 0 {
		return 0
	}
	bits := math.Float32bits(ratio)
	exp := int((bits >> 23) & 0xff)
	if exp == 0 {
		exp = int(math.Ceil(math.Log2(a/maxVal))) + 127
	} else {
		exp = int(((bits - 1) >> 23) + 1)
	}
	if exp < 0 {
		return 0
	}
	if exp > 254 {
		return 254
	}
	return uint8(exp)
}

// DequantizeScaleUE8M0 reconstructs the float32 power-of-two scale factor
// 2^(int(exp)-127) from its biased exponent bits via math.Float32frombits(uint32(exp) << 23).
func DequantizeScaleUE8M0(exp uint8) float32 {
	return math.Float32frombits(uint32(exp) << 23)
}

// PackUE8M0 packs unsigned 8-bit UE8M0 biased exponents into uint32 words in
// little-endian byte order (e0 | e1<<8 | e2<<16 | e3<<24), padding trailing
// bytes with zero when len(exponents) is not a multiple of 4.
func PackUE8M0(exponents []uint8) ([]uint32, error) {
	if len(exponents) == 0 {
		return nil, fmt.Errorf("cudaquant: pack ue8m0: empty exponents")
	}
	words := (len(exponents) + 3) / 4
	out := make([]uint32, words)
	for i, e := range exponents {
		out[i/4] |= uint32(e) << ((i % 4) * 8)
	}
	return out, nil
}

// UnpackUE8M0 unpacks count unsigned 8-bit UE8M0 biased exponents from
// little-endian packed uint32 words.
func UnpackUE8M0(packed []uint32, count int) ([]uint8, error) {
	if count <= 0 {
		return nil, fmt.Errorf("cudaquant: unpack ue8m0: count %d must be positive", count)
	}
	wantWords := (count + 3) / 4
	if len(packed) != wantWords {
		return nil, fmt.Errorf("cudaquant: unpack ue8m0: packed length %d, want %d for count %d", len(packed), wantWords, count)
	}
	out := make([]uint8, count)
	for i := 0; i < count; i++ {
		out[i] = uint8((packed[i/4] >> ((i % 4) * 8)) & 0xff)
	}
	return out, nil
}

// QuantizeBlockFP8UE8M0CPU quantizes a [rows, cols] row-major float32 matrix into
// FP8 E4M3 values using per-block UE8M0 power-of-two scales over contiguous
// horizontal blocks of blockSize elements.
func QuantizeBlockFP8UE8M0CPU(input []float32, rows, cols, blockSize int) (q []uint8, packedScales []uint32, err error) {
	if rows <= 0 || cols <= 0 || blockSize <= 0 || rows > math.MaxInt/cols {
		return nil, nil, fmt.Errorf("cudaquant: invalid dimensions rows=%d cols=%d blockSize=%d", rows, cols, blockSize)
	}
	if cols%blockSize != 0 {
		return nil, nil, fmt.Errorf("cudaquant: cols (%d) must be a multiple of blockSize (%d)", cols, blockSize)
	}
	if len(input) != rows*cols {
		return nil, nil, fmt.Errorf("cudaquant: input length %d does not match rows*cols (%d)", len(input), rows*cols)
	}
	for i, v := range input {
		if !finite(v) {
			return nil, nil, fmt.Errorf("cudaquant: non-finite input at index %d: %v", i, v)
		}
	}

	blocksPerRow := cols / blockSize
	totalBlocks := rows * blocksPerRow
	q = make([]uint8, rows*cols)
	exponents := make([]uint8, totalBlocks)

	for r := 0; r < rows; r++ {
		for b := 0; b < blocksPerRow; b++ {
			base := r*cols + b*blockSize
			amax := float32(0)
			for i := 0; i < blockSize; i++ {
				av := float32(math.Abs(float64(input[base+i])))
				if av > amax {
					amax = av
				}
			}
			exp := QuantizeScaleUE8M0(amax, fp8E4M3Max)
			exponents[r*blocksPerRow+b] = exp
			scale := DequantizeScaleUE8M0(exp)
			var invScale float32
			if scale > 0 {
				invScale = 1.0 / scale
			}
			for i := 0; i < blockSize; i++ {
				q[base+i] = float32ToFP8E4M3(input[base+i] * invScale)
			}
		}
	}

	packedScales, err = PackUE8M0(exponents)
	if err != nil {
		return nil, nil, err
	}
	return q, packedScales, nil
}

// DequantizeBlockFP8UE8M0CPU dequantizes a [rows, cols] FP8 E4M3 matrix and its
// packed UE8M0 block scales back to float32.
func DequantizeBlockFP8UE8M0CPU(q []uint8, packedScales []uint32, rows, cols, blockSize int) ([]float32, error) {
	if rows <= 0 || cols <= 0 || blockSize <= 0 || rows > math.MaxInt/cols {
		return nil, fmt.Errorf("cudaquant: invalid dimensions rows=%d cols=%d blockSize=%d", rows, cols, blockSize)
	}
	if cols%blockSize != 0 {
		return nil, fmt.Errorf("cudaquant: cols (%d) must be a multiple of blockSize (%d)", cols, blockSize)
	}
	if len(q) != rows*cols {
		return nil, fmt.Errorf("cudaquant: quantized length %d does not match rows*cols (%d)", len(q), rows*cols)
	}
	blocksPerRow := cols / blockSize
	totalBlocks := rows * blocksPerRow
	exponents, err := UnpackUE8M0(packedScales, totalBlocks)
	if err != nil {
		return nil, err
	}

	out := make([]float32, rows*cols)
	for r := 0; r < rows; r++ {
		for b := 0; b < blocksPerRow; b++ {
			exp := exponents[r*blocksPerRow+b]
			if exp == 0xFF {
				return nil, fmt.Errorf("cudaquant: invalid NaN UE8M0 exponent 0xff at block %d", r*blocksPerRow+b)
			}
			scale := DequantizeScaleUE8M0(exp)
			base := r*cols + b*blockSize
			for i := 0; i < blockSize; i++ {
				val, ok := fp8E4M3ToFloat32(q[base+i])
				if !ok {
					return nil, fmt.Errorf("cudaquant: NaN FP8 E4M3 code %#02x at index %d", q[base+i], base+i)
				}
				out[base+i] = val * scale
			}
		}
	}
	return out, nil
}
