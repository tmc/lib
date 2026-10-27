package hopper

import (
	"fmt"
	"math"
	"strings"
)

// SubBytePackType identifies a PTX ISA 9.4 (§9.7.6) 4-wide quad-packed
// sub-byte floating-point SIMD register format on Rubin (sm_107f).
type SubBytePackType string

const (
	SubBytePackE4M3x4   SubBytePackType = "e4m3x4"
	SubBytePackE5M2x4   SubBytePackType = "e5m2x4"
	SubBytePackE3M2x4   SubBytePackType = "e3m2x4"
	SubBytePackE2M3x4   SubBytePackType = "e2m3x4"
	SubBytePackE2M1x4   SubBytePackType = "e2m1x4"
	SubBytePackE2M1P4x4 SubBytePackType = "e2m1p4x4"
)

// Valid reports whether t is a supported PTX ISA 9.4 quad-packed sub-byte type.
func (t SubBytePackType) Valid() bool {
	switch t {
	case SubBytePackE4M3x4, SubBytePackE5M2x4, SubBytePackE3M2x4,
		SubBytePackE2M3x4, SubBytePackE2M1x4, SubBytePackE2M1P4x4:
		return true
	default:
		return false
	}
}

// encodeE4M3SatFinite encodes f into OCP E4M3 with round-to-nearest-even (.rn)
// and .satfinite semantics (clamping overflow and ±Inf to ±448.0 = sign|0x7e).
func encodeE4M3SatFinite(f float32) uint8 {
	bits := math.Float32bits(f)
	if bits&0x7fffffff > 0x7f800000 {
		return 0x7f
	}
	return E4M3Bits(f)
}

// encodeE2M1SatFinite encodes f into a 4-bit E2M1 code (bit 3 sign, bits 2..0
// magnitude in {0, 0.5, 1, 1.5, 2, 3, 4, 6}) using round-to-nearest-even (.rn)
// and .satfinite saturation to ±6.0.
func encodeE2M1SatFinite(f float32) uint8 {
	bits := math.Float32bits(f)
	if bits&0x7fffffff > 0x7f800000 {
		return 0x00
	}
	var sign uint8
	if bits&0x80000000 != 0 {
		sign = 0x08
		f = -f
	}
	var mag uint8
	switch {
	case f > 5.0:
		mag = 7
	case f >= 3.5:
		mag = 6
	case f > 2.5:
		mag = 5
	case f >= 1.75:
		mag = 4
	case f > 1.25:
		mag = 3
	case f >= 0.75:
		mag = 2
	case f > 0.25:
		mag = 1
	default:
		mag = 0
	}
	if mag == 0 {
		return 0x00
	}
	return sign | mag
}

// encodeE5M2SatFinite encodes f into FP8 E5M2 (1 sign, 5 exp bias 15, 2 mantissa)
// with .rn.satfinite saturation to ±57344.0 (sign | 0x7b).
func encodeE5M2SatFinite(f float32) uint8 {
	bits := math.Float32bits(f)
	if bits&0x7fffffff > 0x7f800000 {
		return 0x7f
	}
	sign := uint8(bits>>24) & 0x80
	a := bits & 0x7fffffff
	if a >= 0x47600000 { // >= 57344.0
		return sign | 0x7b
	}
	if a < 0x38800000 { // subnormal < 2^-14, step 2^-16
		m := uint8(math.RoundToEven(float64(math.Float32frombits(a)) * 65536.0))
		return sign | m
	}
	e := int(a>>23) - 127 + 15
	m := (a >> 21) & 0x3
	rem := a & 0x1fffff
	if rem > 0x100000 || (rem == 0x100000 && (m&1) == 1) {
		m++
		if m == 4 {
			m = 0
			e++
		}
	}
	if e > 30 {
		return sign | 0x7b
	}
	return sign | uint8(e<<2) | uint8(m)
}

// encodeE3M2SatFinite encodes f into 6-bit MXFP6 E3M2 (1 sign at bit 5, 3 exp
// bias 3, 2 mantissa, max finite ±28.0 = sign | 0x1f) with .rn.satfinite.
func encodeE3M2SatFinite(f float32) uint8 {
	bits := math.Float32bits(f)
	if bits&0x7fffffff > 0x7f800000 {
		return 0x00
	}
	var sign uint8
	if bits&0x80000000 != 0 {
		sign = 0x20
	}
	a := bits & 0x7fffffff
	if a >= 0x41e00000 { // >= 28.0
		return sign | 0x1f
	}
	if a < 0x3e800000 { // < 2^-2 (0.25): subnormal with step 2^-4 (0.0625)
		m := uint8(math.RoundToEven(float64(math.Float32frombits(a)) * 16.0))
		if m == 0 {
			return 0x00
		}
		return sign | m
	}
	e := int(a>>23) - 127 + 3
	m := (a >> 21) & 0x3
	rem := a & 0x1fffff
	if rem > 0x100000 || (rem == 0x100000 && (m&1) == 1) {
		m++
		if m == 4 {
			m = 0
			e++
		}
	}
	if e > 7 {
		return sign | 0x1f
	}
	return sign | uint8(e<<2) | uint8(m)
}

// encodeE2M3SatFinite encodes f into 6-bit MXFP6 E2M3 (1 sign at bit 5, 2 exp
// bias 1, 3 mantissa, max finite ±7.5 = sign | 0x1f) with .rn.satfinite.
func encodeE2M3SatFinite(f float32) uint8 {
	bits := math.Float32bits(f)
	if bits&0x7fffffff > 0x7f800000 {
		return 0x00
	}
	var sign uint8
	if bits&0x80000000 != 0 {
		sign = 0x20
	}
	a := bits & 0x7fffffff
	if a >= 0x40f00000 { // >= 7.5
		return sign | 0x1f
	}
	if a < 0x3f800000 { // < 1.0: subnormal with step 0.125 (2^-3)
		m := uint8(math.RoundToEven(float64(math.Float32frombits(a)) * 8.0))
		if m == 0 {
			return 0x00
		}
		return sign | m
	}
	e := int(a>>23) - 127 + 1
	m := (a >> 20) & 0x7
	rem := a & 0xfffff
	if rem > 0x80000 || (rem == 0x80000 && (m&1) == 1) {
		m++
		if m == 8 {
			m = 0
			e++
		}
	}
	if e > 3 {
		return sign | 0x1f
	}
	return sign | uint8(e<<3) | uint8(m)
}

// PackE4M3x4 quantizes 4 float32 values into a 32-bit .e4m3x4 word with
// round-to-nearest-even (.rn) and .satfinite saturation to ±448.0. Byte i
// (bits [8*i+7 : 8*i]) holds lane i.
func PackE4M3x4(v [4]float32) uint32 {
	return uint32(encodeE4M3SatFinite(v[0])) |
		(uint32(encodeE4M3SatFinite(v[1])) << 8) |
		(uint32(encodeE4M3SatFinite(v[2])) << 16) |
		(uint32(encodeE4M3SatFinite(v[3])) << 24)
}

// UnpackE4M3x4 decodes a 32-bit .e4m3x4 word into 4 float32 values.
func UnpackE4M3x4(u uint32) [4]float32 {
	return [4]float32{
		FromE4M3Bits(uint8(u & 0xff)),
		FromE4M3Bits(uint8((u >> 8) & 0xff)),
		FromE4M3Bits(uint8((u >> 16) & 0xff)),
		FromE4M3Bits(uint8((u >> 24) & 0xff)),
	}
}

// PackE2M1x4 quantizes 4 float32 values into a 32-bit .e2m1p4x4 / .e2m1x4
// container (1 byte per lane with 4-bit padding in bits 4..7: byte i holds the
// 4-bit E2M1 code in bits 0..3, saturating to ±6.0).
func PackE2M1x4(v [4]float32) uint32 {
	return uint32(encodeE2M1SatFinite(v[0])) |
		(uint32(encodeE2M1SatFinite(v[1])) << 8) |
		(uint32(encodeE2M1SatFinite(v[2])) << 16) |
		(uint32(encodeE2M1SatFinite(v[3])) << 24)
}

// UnpackE2M1x4 decodes a 32-bit packed E2M1x4 / E2M1P4x4 word into 4 float32
// values. It supports the byte-strided .e2m1p4x4 container (byte i holds lane i
// in bits 0..3) as well as compact low-16-bit nibble packing when the upper 16
// bits are zero and high nibbles of the low 16 bits are populated.
func UnpackE2M1x4(u uint32) [4]float32 {
	if (u&0xffff0000) == 0 && (u&0x0000f0f0) != 0 {
		return [4]float32{
			FromE2M1(uint8(u & 0x0f)),
			FromE2M1(uint8((u >> 4) & 0x0f)),
			FromE2M1(uint8((u >> 8) & 0x0f)),
			FromE2M1(uint8((u >> 12) & 0x0f)),
		}
	}
	return [4]float32{
		FromE2M1(uint8(u & 0x0f)),
		FromE2M1(uint8((u >> 8) & 0x0f)),
		FromE2M1(uint8((u >> 16) & 0x0f)),
		FromE2M1(uint8((u >> 24) & 0x0f)),
	}
}

// EvalQuadSubByteFMA evaluates the 4-wide quad-packed sub-byte SIMD FMA
// d[i] = a[i]*b[i] + c[i] for i = 0..3 and quantizes the 4 results into
// outType using .rn.satfinite semantics.
func EvalQuadSubByteFMA(a, b, c [4]float32, outType SubBytePackType) uint32 {
	var d [4]float32
	for i := range 4 {
		d[i] = a[i]*b[i] + c[i]
	}
	switch outType {
	case SubBytePackE4M3x4:
		return PackE4M3x4(d)
	case SubBytePackE2M1x4, SubBytePackE2M1P4x4:
		return PackE2M1x4(d)
	case SubBytePackE5M2x4:
		return uint32(encodeE5M2SatFinite(d[0])) |
			(uint32(encodeE5M2SatFinite(d[1])) << 8) |
			(uint32(encodeE5M2SatFinite(d[2])) << 16) |
			(uint32(encodeE5M2SatFinite(d[3])) << 24)
	case SubBytePackE3M2x4:
		return uint32(encodeE3M2SatFinite(d[0])) |
			(uint32(encodeE3M2SatFinite(d[1])) << 8) |
			(uint32(encodeE3M2SatFinite(d[2])) << 16) |
			(uint32(encodeE3M2SatFinite(d[3])) << 24)
	case SubBytePackE2M3x4:
		return uint32(encodeE2M3SatFinite(d[0])) |
			(uint32(encodeE2M3SatFinite(d[1])) << 8) |
			(uint32(encodeE2M3SatFinite(d[2])) << 16) |
			(uint32(encodeE2M3SatFinite(d[3])) << 24)
	default:
		return PackE4M3x4(d)
	}
}

// EvalQuadSubByteAdd evaluates the 4-wide quad-packed sub-byte SIMD ADD
// d[i] = a[i] + b[i] for i = 0..3 and quantizes the 4 results into E4M3x4
// using .rn.satfinite semantics.
func EvalQuadSubByteAdd(a, b [4]float32) uint32 {
	var d [4]float32
	for i := range 4 {
		d[i] = a[i] + b[i]
	}
	return PackE4M3x4(d)
}

// f32ToF16FTZRZ implements a single lane of PTX ISA 9.4
// mul.ftz.rz.f16x2.f32x2.f32x2: flushes subnormals to zero and rounds the
// product toward zero (.rz) into IEEE binary16 bits.
func f32ToF16FTZRZ(f float32) uint16 {
	u := math.Float32bits(f)
	s := uint16((u >> 16) & 0x8000)
	a := u & 0x7fffffff
	if a > 0x7f800000 {
		return s | 0x7e00
	}
	if a == 0x7f800000 {
		return s | 0x7c00
	}
	if a >= 0x47800000 { // finite overflow >= 65536.0 truncates under .rz to max normal 65504.0 (0x7bff)
		return s | 0x7bff
	}
	if a < 0x38800000 { // < 2^-14 flushes to ±0 under .ftz.rz
		return s
	}
	e := ((a >> 23) - 112) << 10
	m := (a >> 13) & 0x3ff
	return s | uint16(e|m)
}

func f16BitsToF32(h uint16) float32 {
	s := uint32(h&0x8000) << 16
	e := uint32((h >> 10) & 0x1f)
	m := uint32(h & 0x3ff)
	if e == 0 {
		if m == 0 {
			return math.Float32frombits(s)
		}
		v := float32(math.Ldexp(float64(m), -24))
		if s != 0 {
			return -v
		}
		return v
	}
	if e == 31 {
		if m != 0 {
			return math.Float32frombits(s | 0x7fc00000)
		}
		return math.Float32frombits(s | 0x7f800000)
	}
	return math.Float32frombits(s | ((e + 112) << 23) | (m << 13))
}

func evalMulFTZRZF16x2F32x2(a0, a1, b0, b1 float32) (out0, out1 float32) {
	p0 := flushSubnormalF32(a0) * flushSubnormalF32(b0)
	p1 := flushSubnormalF32(a1) * flushSubnormalF32(b1)
	h0 := f32ToF16FTZRZ(p0)
	h1 := f32ToF16FTZRZ(p1)
	return f16BitsToF32(h0), f16BitsToF32(h1)
}

const rubinPTX94QuadSIMDHelpers = `#ifndef EG_RUBIN_PTX94_QUAD_SIMD_HELPERS
#define EG_RUBIN_PTX94_QUAD_SIMD_HELPERS

EG_DEV float eg_rubin_simd_flush_subnormal_f32(float x) {
    const uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u) {
        return 0.0f;
    }
    return x;
}

EG_DEV float eg_rubin_simd_e2m1_to_f32(uint32_t code) {
    static constexpr float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = grid[code & 7u];
    return (code & 8u) ? -v : v;
}

EG_DEV uint8_t eg_rubin_simd_f32_to_e4m3_rn_satfinite(float f) {
    const uint32_t bits = __float_as_uint(f);
    const uint32_t a = bits & 0x7fffffffu;
    if (a > 0x7f800000u) return 0x7fu;
    const uint8_t sign = (uint8_t)((bits >> 24) & 0x80u);
    if (a >= 0x43e00000u) return sign | 0x7eu;
    if (a < 0x3c800000u) {
        const float scaled = __uint_as_float(a) * 512.0f;
        int m = (int)floorf(scaled);
        const float frac = scaled - (float)m;
        if (frac > 0.5f || (frac == 0.5f && (m & 1) != 0)) {
            m++;
        }
        return sign | (uint8_t)m;
    }
    int e = (int)(a >> 23) - 127 + 7;
    uint32_t m = (a >> 20) & 7u;
    const uint32_t rem = a & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u) != 0u)) {
        if (++m == 8u) {
            m = 0u;
            ++e;
        }
    }
    if (e > 15 || (e == 15 && m == 7u)) return sign | 0x7eu;
    return sign | (uint8_t)(e << 3) | (uint8_t)m;
}

EG_DEV uint16_t eg_rubin_simd_f32_to_f16_ftz_rz(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint16_t s = (uint16_t)((u >> 16) & 0x8000u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7e00u;
    if (a == 0x7f800000u) return s | 0x7c00u;
    if (a >= 0x47800000u) return s | 0x7bffu;
    if (a < 0x38800000u) return s;
    const uint32_t e = ((a >> 23) - 112u) << 10;
    const uint32_t m = (a >> 13) & 0x3ffu;
    return s | (uint16_t)(e | m);
}

EG_DEV float eg_rubin_simd_f16_to_f32(uint16_t h) {
    const uint32_t s = ((uint32_t)(h & 0x8000u)) << 16;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    if (e == 0u) {
        if (m == 0u) return __uint_as_float(s);
        const float v = (float)m * (1.0f / 16777216.0f);
        return s ? -v : v;
    }
    if (e == 31u) {
        return __uint_as_float(s | (m ? 0x7fc00000u : 0x7f800000u));
    }
    return __uint_as_float(s | ((e + 112u) << 23) | (m << 13));
}

// eg_rubin_fma_e4m3x4_e2m1p4x4 implements PTX ISA 9.4 (§9.7.6)
// fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4 d, a, b, c;
EG_DEV uint32_t eg_rubin_fma_e4m3x4_e2m1p4x4(uint32_t a_e2m1p4x4, uint32_t b_e4m3x4, uint32_t c_e4m3x4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070
    uint32_t d;
    asm("fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4 %0, %1, %2, %3;"
        : "=r"(d) : "r"(a_e2m1p4x4), "r"(b_e4m3x4), "r"(c_e4m3x4));
    return d;
#else
    uint32_t out = 0u;
    for (int i = 0; i < 4; ++i) {
        const float av = eg_rubin_simd_e2m1_to_f32((a_e2m1p4x4 >> (8 * i)) & 0x0fu);
        const float bv = eg_e4m3_to_f32((b_e4m3x4 >> (8 * i)) & 0xffu);
        const float cv = eg_e4m3_to_f32((c_e4m3x4 >> (8 * i)) & 0xffu);
        const uint8_t dv = eg_rubin_simd_f32_to_e4m3_rn_satfinite(av * bv + cv);
        out |= ((uint32_t)dv) << (8 * i);
    }
    return out;
#endif
}

// eg_rubin_add_e4m3x4_e2m1p4x4 implements PTX ISA 9.4 (§9.7.6)
// add.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4 d, a, b;
EG_DEV uint32_t eg_rubin_add_e4m3x4_e2m1p4x4(uint32_t a_e2m1p4x4, uint32_t b_e4m3x4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070
    uint32_t d;
    asm("add.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4 %0, %1, %2;"
        : "=r"(d) : "r"(a_e2m1p4x4), "r"(b_e4m3x4));
    return d;
#else
    uint32_t out = 0u;
    for (int i = 0; i < 4; ++i) {
        const float av = eg_rubin_simd_e2m1_to_f32((a_e2m1p4x4 >> (8 * i)) & 0x0fu);
        const float bv = eg_e4m3_to_f32((b_e4m3x4 >> (8 * i)) & 0xffu);
        const uint8_t dv = eg_rubin_simd_f32_to_e4m3_rn_satfinite(av + bv);
        out |= ((uint32_t)dv) << (8 * i);
    }
    return out;
#endif
}

// eg_rubin_mul_ftz_rz_f16x2_f32x2 implements PTX ISA 9.4 (§9.7.6)
// mul.ftz.rz.f16x2.f32x2.f32x2 d, a, b;
EG_DEV uint32_t eg_rubin_mul_ftz_rz_f16x2_f32x2(float a0, float a1, float b0, float b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1070
    uint32_t d;
    const uint64_t pa = ((uint64_t)__float_as_uint(a0)) | (((uint64_t)__float_as_uint(a1)) << 32);
    const uint64_t pb = ((uint64_t)__float_as_uint(b0)) | (((uint64_t)__float_as_uint(b1)) << 32);
    asm("mul.ftz.rz.f16x2.f32x2.f32x2 %0, %1, %2;" : "=r"(d) : "l"(pa), "l"(pb));
    return d;
#else
    const float p0 = eg_rubin_simd_flush_subnormal_f32(a0) * eg_rubin_simd_flush_subnormal_f32(b0);
    const float p1 = eg_rubin_simd_flush_subnormal_f32(a1) * eg_rubin_simd_flush_subnormal_f32(b1);
    const uint16_t h0 = eg_rubin_simd_f32_to_f16_ftz_rz(p0);
    const uint16_t h1 = eg_rubin_simd_f32_to_f16_ftz_rz(p1);
    return (uint32_t)h0 | (((uint32_t)h1) << 16);
#endif
}

#endif // EG_RUBIN_PTX94_QUAD_SIMD_HELPERS
`

// RubinQuadSubByteSIMDEpilogue models a fused Rubin (sm_107f) PTX ISA 9.4
// (§9.7.6) sub-byte epilogue kernel operating on quad-packed .e4m3x4 and
// .e2m1p4x4 vectors plus down-converting mul.ftz.rz.f16x2.f32x2.f32x2.
type RubinQuadSubByteSIMDEpilogue struct {
	Name     string
	NumQuads int    // default 64 (256 elements)
	Target   string // default "sm_107f" (valid: "sm_107a", "sm_107f")
}

func (k RubinQuadSubByteSIMDEpilogue) norm() (RubinQuadSubByteSIMDEpilogue, error) {
	if k.Name == "" {
		k.Name = "rubin_quad_subbyte_simd_epilogue"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("rubin quad subbyte simd epilogue: bad name %q", k.Name)
	}
	if k.NumQuads == 0 {
		k.NumQuads = 64
	}
	if k.NumQuads < 1 || k.NumQuads > 65536 {
		return k, fmt.Errorf("rubin quad subbyte simd epilogue %s: NumQuads %d must be in 1..65536", k.Name, k.NumQuads)
	}
	if k.Target == "" {
		k.Target = "sm_107f"
	}
	if !validRubinPTX94Target(k.Target) {
		return k, fmt.Errorf("rubin quad subbyte simd epilogue %s: unsupported target %q (want sm_107a or sm_107f)", k.Name, k.Target)
	}
	return k, nil
}

// Validate checks that k has a valid kernel identifier, positive NumQuads, and
// a Rubin target ("sm_107a" or "sm_107f").
func (k RubinQuadSubByteSIMDEpilogue) Validate() error {
	_, err := k.norm()
	return err
}

func (k RubinQuadSubByteSIMDEpilogue) params() []string {
	return []string{
		"const unsigned int* __restrict__ a_e2m1x4",
		"const unsigned int* __restrict__ b_e4m3x4",
		"const unsigned int* __restrict__ c_e4m3x4",
		"unsigned int* __restrict__ out_fma_e4m3x4",
		"unsigned int* __restrict__ out_add_e4m3x4",
		"float* __restrict__ out_f32",
		"int num_quads",
	}
}

// CUDA returns the Rubin (sm_107f) PTX ISA 9.4 quad-packed sub-byte SIMD
// epilogue device kernel source (without [Prelude]).
func (k RubinQuadSubByteSIMDEpilogue) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(rubinPTX94QuadSIMDHelpers)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: Rubin (%s) PTX ISA 9.4 quad-packed sub-byte SIMD epilogue (NumQuads=%d).\n",
		k.Name, k.Target, k.NumQuads)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(64) %s(\n    %s) {\n",
		k.Name, strings.Join(k.params(), ",\n    "))
	b.WriteString(`    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(gridDim.x * blockDim.x);
    for (int q = tid; q < num_quads; q += stride) {
        const uint32_t a_q = a_e2m1x4[q];
        const uint32_t b_q = b_e4m3x4[q];
        const uint32_t c_q = c_e4m3x4[q];

        const uint32_t d_fma = eg_rubin_fma_e4m3x4_e2m1p4x4(a_q, b_q, c_q);
        const uint32_t d_add = eg_rubin_add_e4m3x4_e2m1p4x4(a_q, b_q);

        out_fma_e4m3x4[q] = d_fma;
        out_add_e4m3x4[q] = d_add;

        const float f0 = eg_e4m3_to_f32(d_fma & 0xffu);
        const float f1 = eg_e4m3_to_f32((d_fma >> 8) & 0xffu);
        const float f2 = eg_e4m3_to_f32((d_fma >> 16) & 0xffu);
        const float f3 = eg_e4m3_to_f32((d_fma >> 24) & 0xffu);

        const float g0 = eg_e4m3_to_f32(d_add & 0xffu);
        const float g1 = eg_e4m3_to_f32((d_add >> 8) & 0xffu);
        const float g2 = eg_e4m3_to_f32((d_add >> 16) & 0xffu);
        const float g3 = eg_e4m3_to_f32((d_add >> 24) & 0xffu);

        const uint32_t p01 = eg_rubin_mul_ftz_rz_f16x2_f32x2(f0, f1, g0, g1);
        const uint32_t p23 = eg_rubin_mul_ftz_rz_f16x2_f32x2(f2, f3, g2, g3);

        const int base = 4 * q;
        out_f32[base + 0] = eg_rubin_simd_f16_to_f32((uint16_t)(p01 & 0xffffu));
        out_f32[base + 1] = eg_rubin_simd_f16_to_f32((uint16_t)(p01 >> 16));
        out_f32[base + 2] = eg_rubin_simd_f16_to_f32((uint16_t)(p23 & 0xffffu));
        out_f32[base + 3] = eg_rubin_simd_f16_to_f32((uint16_t)(p23 >> 16));
    }
}
`)
	return b.String(), nil
}

// Launcher returns a host launcher function for [RubinQuadSubByteSIMDEpilogue].
func (k RubinQuadSubByteSIMDEpilogue) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := paramNames(params)
	blocks := (k.NumQuads + 63) / 64
	if blocks < 1 {
		blocks = 1
	}
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream = nullptr) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    const dim3 grid((num_quads + 63) / 64, 1, 1), block(64, 1, 1);\n")
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", k.Name, strings.Join(names, ", "))
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream = nullptr) {\n", k.Name, strings.Join(params[:len(params)-1], ", "))
	fmt.Fprintf(&b, "    const dim3 grid(%d, 1, 1), block(64, 1, 1);\n", blocks)
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s, %d);\n}\n",
		k.Name, strings.Join(names[:len(names)-1], ", "), k.NumQuads)
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude], the kernel, and its launcher.
func (k RubinQuadSubByteSIMDEpilogue) Source() (string, error) {
	src, err := k.CUDA()
	if err != nil {
		return "", err
	}
	l, err := k.Launcher()
	if err != nil {
		return "", err
	}
	return Prelude + "\n" + src + "\n" + l, nil
}

// Reference executes the bit-exact CPU reference for [RubinQuadSubByteSIMDEpilogue]
// over NumQuads packed .e2m1p4x4 and .e4m3x4 input words.
func (k RubinQuadSubByteSIMDEpilogue) Reference(aE2M1x4, bE4M3x4, cE4M3x4 []uint32) (outFMAE4M3x4, outAddE4M3x4 []uint32, outF32 []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	n := k.NumQuads
	if len(aE2M1x4) != n || len(bE4M3x4) != n || len(cE4M3x4) != n {
		return nil, nil, nil, fmt.Errorf("rubin quad subbyte simd epilogue %s reference: len(a)=%d len(b)=%d len(c)=%d, want %d",
			k.Name, len(aE2M1x4), len(bE4M3x4), len(cE4M3x4), n)
	}
	outFMAE4M3x4 = make([]uint32, n)
	outAddE4M3x4 = make([]uint32, n)
	outF32 = make([]float32, 4*n)

	for q := range n {
		av := UnpackE2M1x4(aE2M1x4[q])
		bv := UnpackE4M3x4(bE4M3x4[q])
		cv := UnpackE4M3x4(cE4M3x4[q])

		dFMA := EvalQuadSubByteFMA(av, bv, cv, SubBytePackE4M3x4)
		dAdd := EvalQuadSubByteAdd(av, bv)
		outFMAE4M3x4[q] = dFMA
		outAddE4M3x4[q] = dAdd

		fv := UnpackE4M3x4(dFMA)
		gv := UnpackE4M3x4(dAdd)
		p0, p1 := evalMulFTZRZF16x2F32x2(fv[0], fv[1], gv[0], gv[1])
		p2, p3 := evalMulFTZRZF16x2F32x2(fv[2], fv[3], gv[2], gv[3])
		base := 4 * q
		outF32[base+0] = p0
		outF32[base+1] = p1
		outF32[base+2] = p2
		outF32[base+3] = p3
	}
	return outFMAE4M3x4, outAddE4M3x4, outF32, nil
}
