package hopper

import (
	"fmt"
	"math"
)

// A DType is the element type of a tensor in memory.
// Values in registers are always FP32; a DType says how they are stored.
type DType uint8

const (
	F32  DType = iota + 1
	BF16       // bfloat16, round to nearest even
	E4M3       // float8_e4m3fn, saturating to ±448
	E2M1       // float4_e2m1fn, two codes per byte, element 2i in the low nibble
	U8
	I32
	I64
	F16 // IEEE binary16 (half precision), round to nearest even

	FP16 = F16
	FP32 = F32
)

var dtypeNames = [...]string{F32: "f32", BF16: "bf16", E4M3: "e4m3", E2M1: "e2m1", U8: "u8", I32: "i32", I64: "i64"}

func (d DType) String() string {
	if int(d) < len(dtypeNames) && dtypeNames[d] != "" {
		return dtypeNames[d]
	}
	return fmt.Sprintf("DType(%d)", uint8(d))
}

// ParseDType returns the DType named s, as printed by [DType.String].
func ParseDType(s string) (DType, error) {
	for d, name := range dtypeNames {
		if name != "" && name == s {
			return DType(d), nil
		}
	}
	return 0, fmt.Errorf("unknown dtype %q", s)
}

// Bits reports the storage size of one element in bits.
func (d DType) Bits() int {
	switch d {
	case F32, I32:
		return 32
	case BF16:
		return 16
	case E4M3, U8:
		return 8
	case E2M1:
		return 4
	case I64:
		return 64
	}
	return 0
}

// cType is the C type a kernel uses to address d in memory. Low-precision
// floats are carried as raw bits so emitted code needs no CUDA headers.
func (d DType) cType() string {
	switch d {
	case F32:
		return "float"
	case BF16:
		return "uint16_t"
	case E4M3, E2M1, U8:
		return "uint8_t"
	case I32:
		return "int32_t"
	case I64:
		return "int64_t"
	}
	return ""
}

// Round returns f rounded to d, as the matching CUDA conversion does.
// Integer and packed types return f unchanged.
func (d DType) Round(f float32) float32 {
	switch d {
	case BF16:
		return FromBF16Bits(BF16Bits(f))
	case E4M3:
		return FromE4M3Bits(E4M3Bits(f))
	}
	return f
}

// BF16Bits returns the bfloat16 encoding of f, rounded to nearest even, as
// cvt.rn.bf16.f32 and torch's .to(torch.bfloat16) do. NaN becomes 0x7fc0.
func BF16Bits(f float32) uint16 {
	u := math.Float32bits(f)
	if u&0x7fffffff > 0x7f800000 {
		return 0x7fc0 | uint16(u>>16)&0x8000
	}
	u += 0x7fff + (u>>16)&1
	return uint16(u >> 16)
}

// FromBF16Bits decodes a bfloat16 encoding.
func FromBF16Bits(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// SplitBF16 splits f into hi + lo, both bfloat16, with hi = bf16(f) and
// lo = bf16(f - hi). The pair carries 16 significant bits, so
// x·hi + x·lo with FP32 accumulation reproduces an FP32 weight to about
// 2^-17 relative error (moe8_bf16_proj/reference.py:28-29).
func SplitBF16(f float32) (hi, lo uint16) {
	hi = BF16Bits(f)
	lo = BF16Bits(f - FromBF16Bits(hi))
	return hi, lo
}

// E4M3Bits encodes f as float8_e4m3fn, rounding to nearest even and
// saturating finite overflow to ±448, as cvt.rn.satfinite.e4m3x2.f32 does.
// NaN encodes as 0x7f with f's sign. torch's .to(float8_e4m3fn) differs only
// above 464 in magnitude, where it returns NaN.
func E4M3Bits(f float32) uint8 {
	bits := math.Float32bits(f)
	sign := uint8(bits>>24) & 0x80
	a := bits & 0x7fffffff
	if a > 0x7f800000 {
		return sign | 0x7f
	}
	if a >= 0x43e00000 { // 448
		return sign | 0x7e
	}
	if a < 0x3c800000 { // below 2^-6: subnormal, step 2^-9
		m := uint8(math.RoundToEven(float64(math.Float32frombits(a)) * 512))
		return sign | m // m == 8 is 2^-6, the smallest normal
	}
	e := int(a>>23) - 127 + 7
	m := (a >> 20) & 7
	rem := a & 0xfffff
	if rem > 0x80000 || rem == 0x80000 && m&1 == 1 {
		m++
		if m == 8 {
			m = 0
			e++
		}
	}
	if e > 15 || e == 15 && m == 7 {
		return sign | 0x7e
	}
	return sign | uint8(e<<3) | uint8(m)
}

// FromE4M3Bits decodes a float8_e4m3fn encoding. 0x7f and 0xff are NaN.
func FromE4M3Bits(b uint8) float32 {
	e, m := int(b>>3)&15, int(b&7)
	var v float32
	switch {
	case e == 15 && m == 7:
		v = float32(math.NaN())
	case e == 0:
		v = float32(m) / 512
	default:
		v = float32(math.Ldexp(1+float64(m)/8, e-7))
	}
	if b&0x80 != 0 {
		return -v
	}
	return v
}

// e2m1 magnitudes by 3-bit code (moe8_moe_w4a16/nvfp4.py:24).
var e2m1Grid = [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// FromE2M1 decodes a 4-bit E2M1 code; bit 3 is the sign
// (moe8_moe_w4a16/nvfp4.py:40-47).
func FromE2M1(code uint8) float32 {
	v := e2m1Grid[code&7]
	if code&8 != 0 {
		return -v
	}
	return v
}

// NVFP4Block is the number of values sharing one E4M3 block scale.
const NVFP4Block = 16

// DequantNVFP4 decodes one NVFP4 row of k values: packed holds k/2 bytes
// (element 2i in the low nibble), scales k/16 E4M3 block scales, and global
// the FP32 tensor scale. Each value is e2m1 * (scale * global), computed in
// FP32 as moe8_moe_w4a16/nvfp4.py:90-97 does.
func DequantNVFP4(dst []float32, packed, scales []uint8, global float32) error {
	k := len(dst)
	if k%NVFP4Block != 0 || len(packed) != k/2 || len(scales) != k/NVFP4Block {
		return fmt.Errorf("dequant nvfp4: %d values need %d packed bytes and %d scales, have %d and %d",
			k, k/2, k/NVFP4Block, len(packed), len(scales))
	}
	for i := range dst {
		code := packed[i/2] >> (4 * (i & 1)) & 15
		dst[i] = FromE2M1(code) * (FromE4M3Bits(scales[i/NVFP4Block]) * global)
	}
	return nil
}
