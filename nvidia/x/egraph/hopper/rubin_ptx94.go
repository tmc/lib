package hopper

import (
	"fmt"
	"math"
	"strings"
)

// MaxNormalUE5M3 is the maximum finite value representable in PTX ISA 9.4's
// unsigned 8-bit UE5M3 floating-point format: (1 + 7/8) * 2^(30-15) = 61440.0
// (exponent 30 = 0x1e, mantissa 7 = 0x7, byte 0xf7).
const MaxNormalUE5M3 = 61440.0

// ToUE5M3Bits encodes v into PTX ISA 9.4's unsigned 8-bit UE5M3 format
// (0 sign bits, 5 exponent bits bits[7:3] with bias 15, 3 mantissa bits bits[2:0])
// using round-to-nearest-even (.rn) and saturating finite semantics (.satfinite):
//   - Negative values and -0.0 clamp to 0x00 (+0.0).
//   - NaN, +Inf, and finite values above 61440.0 clamp to 0xf7 (61440.0).
//   - Subnormals (exp == 0) encode (m / 8.0) * 2^-14 with step 2^-17.
//   - Normals (1 <= exp <= 30) encode (1.0 + m / 8.0) * 2^(exp - 15).
func ToUE5M3Bits(v float32) uint8 {
	bits := math.Float32bits(v)
	// NaN clamps to 0xf7 under .satfinite.
	if bits&0x7fffffff > 0x7f800000 {
		return 0xf7
	}
	// Negative values (including -Inf and -0.0) clamp to +0.0 (0x00).
	if bits&0x80000000 != 0 || bits == 0 {
		return 0x00
	}
	// +Inf or >= 61440.0 (0x47700000) clamps to max normal 0xf7.
	if bits >= 0x47700000 {
		return 0xf7
	}
	// Below 2^-14 (0x38800000): subnormal with step 2^-17 = 1/131072.
	if bits < 0x38800000 {
		m := uint8(math.RoundToEven(float64(v) * 131072.0))
		return m // m == 8 is 0x08 (exp=1, m=0 = 2^-14)
	}
	e := int((bits>>23)&0xff) - 127 + 15
	m := (bits >> 20) & 0x7
	rem := bits & 0xfffff
	if rem > 0x80000 || (rem == 0x80000 && (m&1) == 1) {
		m++
		if m == 8 {
			m = 0
			e++
		}
	}
	if e > 30 {
		return 0xf7
	}
	return uint8((e << 3) | int(m))
}

// FromUE5M3Bits decodes a PTX ISA 9.4 UE5M3 byte into float32.
// Exponent 31 (0x1f, codes 0xf8..0xff) decodes as NaN.
func FromUE5M3Bits(b uint8) float32 {
	e := int((b >> 3) & 0x1f)
	m := int(b & 0x7)
	switch {
	case e == 31:
		return float32(math.NaN())
	case e == 0:
		return float32(m) * (1.0 / 131072.0)
	default:
		return float32(math.Ldexp(1.0+float64(m)/8.0, e-15))
	}
}

// PackUE5M3x2 quantizes lo and hi to UE5M3 and packs lo into bits[7:0] and hi
// into bits[15:8].
func PackUE5M3x2(lo, hi float32) uint16 {
	return uint16(ToUE5M3Bits(lo)) | (uint16(ToUE5M3Bits(hi)) << 8)
}

// UnpackUE5M3x2 decodes a packed UE5M3x2 uint16 into (lo, hi) float32 values.
func UnpackUE5M3x2(packed uint16) (lo, hi float32) {
	return FromUE5M3Bits(uint8(packed & 0xff)), FromUE5M3Bits(uint8(packed >> 8))
}

// ToUE8M0 encodes a positive scale factor as an unsigned 8-bit exponent-only
// UE8M0 byte (bias 127) by rounding up to the next power of two (ceil_pow2).
// Non-positive inputs return 0x00 (2^-127); NaN returns 0xff.
func ToUE8M0(s float32) uint8 {
	bits := math.Float32bits(s)
	if bits&0x7fffffff > 0x7f800000 {
		return 0xff
	}
	if bits&0x80000000 != 0 || bits == 0 {
		return 0x00
	}
	if s <= float32(math.Ldexp(1.0, -127)) {
		return 0x00
	}
	exp := int((bits >> 23) & 0xff)
	mant := bits & 0x7fffff
	if exp == 0 {
		return 0x00
	}
	if exp >= 0xff {
		return 0xfe
	}
	e := exp
	if mant != 0 {
		e++
	}
	if e > 254 {
		e = 254
	}
	return uint8(e)
}

// FromUE8M0 decodes an unsigned 8-bit UE8M0 power-of-two scale byte (bias 127).
// Code 0xff decodes as NaN; codes 0..254 decode as 2^(code - 127).
func FromUE8M0(code uint8) float32 {
	if code == 0xff {
		return float32(math.NaN())
	}
	if code == 0 {
		return float32(math.Ldexp(1.0, -127))
	}
	return math.Float32frombits(uint32(code) << 23)
}

// ApplyPZO8 implements PTX ISA 9.4 §9.7.8.8 (.pzo, Positive-Zero-Only) on an
// 8-bit or 4-bit narrowing float conversion code: when code equals signBitMask
// (negative zero, e.g. 0x80 for E4M3/E5M2 or 0x08 for E2M1), it is forced to
// positive zero (0x00) while all non-zero negative encodings are preserved.
func ApplyPZO8(code uint8, signBitMask uint8) uint8 {
	if signBitMask == 0x88 {
		lo := code & 0x0f
		hi := code & 0xf0
		if lo == 0x08 {
			lo = 0x00
		}
		if hi == 0x80 {
			hi = 0x00
		}
		return lo | hi
	}
	if code == signBitMask {
		return 0x00
	}
	return code
}

func flushSubnormalF32(x float32) float32 {
	bits := math.Float32bits(x)
	if (bits&0x7f800000) == 0 && (bits&0x007fffff) != 0 {
		return 0
	}
	return x
}

// CvtScaledN1UE8M0ToUE5M3x2 implements PTX ISA 9.4:
//
//	cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32 d, a, b, scale-factor;
//
// It flushes subnormal FP32 inputs a and b to zero, divides both by
// FromUE8M0(ue8m0Scale), and converts b into d[7:0] and a into d[15:8].
func CvtScaledN1UE8M0ToUE5M3x2(a, b float32, ue8m0Scale uint8) uint16 {
	af := flushSubnormalF32(a)
	bf := flushSubnormalF32(b)
	scale := FromUE8M0(ue8m0Scale)
	lo := ToUE5M3Bits(bf / scale)
	hi := ToUE5M3Bits(af / scale)
	return uint16(lo) | (uint16(hi) << 8)
}

// CvtScaledN2UE8M0FromUE5M3x2 implements PTX ISA 9.4:
//
//	cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2 d, a, scale-factor;
//
// It decodes a[7:0] and a[15:8] from UE5M3, multiplies them by the respective
// UE8M0 scales in ue8m0Pair[7:0] and ue8m0Pair[15:8], rounds each product to
// BF16 (.rn), and returns the resulting (lo, hi) values as float32.
func CvtScaledN2UE8M0FromUE5M3x2(packed uint16, ue8m0Pair uint16) (lo, hi float32) {
	vLo := FromUE5M3Bits(uint8(packed&0xff)) * FromUE8M0(uint8(ue8m0Pair&0xff))
	vHi := FromUE5M3Bits(uint8(packed>>8)) * FromUE8M0(uint8(ue8m0Pair>>8))
	lo = FromBF16Bits(BF16Bits(vLo))
	hi = FromBF16Bits(BF16Bits(vHi))
	return lo, hi
}

const rubinPTX94Prelude = `#ifndef EG_RUBIN_PTX94_PRELUDE
#define EG_RUBIN_PTX94_PRELUDE

EG_DEV float eg_rubin_flush_subnormal_f32(float x) {
    const uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u) {
        return 0.0f;
    }
    return x;
}

EG_DEV uint8_t eg_rubin_to_ue8m0(float s) {
    const uint32_t bits = __float_as_uint(s);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0xffu;
    if ((bits & 0x80000000u) != 0u || bits == 0u) return 0x00u;
    if (s <= __uint_as_float(0x00400000u)) return 0x00u;
    const uint32_t exp = (bits >> 23) & 0xffu;
    const uint32_t mant = bits & 0x7fffffu;
    if (exp == 0u) return 0x00u;
    if (exp >= 0xffu) return 0xfeu;
    uint32_t e = exp + (mant != 0u ? 1u : 0u);
    if (e > 254u) e = 254u;
    return (uint8_t)e;
}

EG_DEV float eg_rubin_from_ue8m0(uint8_t code) {
    if (code == 0xffu) return __uint_as_float(0x7fc00000u);
    if (code == 0x00u) return __uint_as_float(0x00400000u);
    return __uint_as_float((uint32_t)code << 23);
}

EG_DEV uint8_t eg_rubin_f32_to_ue5m3(float v) {
    const uint32_t bits = __float_as_uint(v);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0xf7u;
    if ((bits & 0x80000000u) != 0u || bits == 0u) return 0x00u;
    if (bits >= 0x47700000u) return 0xf7u;
    if (bits < 0x38800000u) {
        const float scaled = v * 131072.0f;
        int m = (int)floorf(scaled);
        const float frac = scaled - (float)m;
        if (frac > 0.5f || (frac == 0.5f && (m & 1) != 0)) {
            m++;
        }
        return (uint8_t)m;
    }
    int e = (int)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t m = (bits >> 20) & 0x7u;
    const uint32_t rem = bits & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u) == 1u)) {
        m++;
        if (m == 8u) {
            m = 0u;
            e++;
        }
    }
    if (e > 30) return 0xf7u;
    return (uint8_t)((e << 3) | (int)m);
}

EG_DEV float eg_rubin_ue5m3_to_f32(uint8_t b) {
    const uint32_t e = (b >> 3) & 0x1fu;
    const uint32_t m = b & 0x7u;
    if (e == 31u) return __uint_as_float(0x7fc00000u);
    if (e == 0u) return (float)m * (1.0f / 131072.0f);
    return __uint_as_float(((e + 112u) << 23) | (m << 20));
}

// eg_rubin_cvt_pzo_e2m1x2_f32 converts (a, b) to packed e2m1x2 with PTX ISA 9.4
// .pzo (Positive-Zero-Only) canonicalization.
EG_DEV uint8_t eg_rubin_cvt_pzo_e2m1x2_f32(float a, float b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint16_t d16;
    asm("{\n\t"
        ".reg .b8 d8;\n\t"
        "cvt.rn.satfinite.pzo.e2m1x2.f32 d8, %1, %2;\n\t"
        "cvt.u16.u8 %0, d8;\n\t"
        "}" : "=h"(d16) : "f"(a), "f"(b));
    return (uint8_t)(d16 & 0xffu);
#else
    auto enc = [](float x) -> uint8_t {
        uint8_t sign = 0u;
        if (__float_as_uint(x) & 0x80000000u) {
            sign = 8u;
            x = -x;
        }
        uint8_t mag = 0u;
        if (x > 5.0f) mag = 7u;
        else if (x > 3.5f) mag = 6u;
        else if (x > 2.5f) mag = 5u;
        else if (x > 1.75f) mag = 4u;
        else if (x > 1.25f) mag = 3u;
        else if (x > 0.75f) mag = 2u;
        else if (x > 0.25f) mag = 1u;
        uint8_t code = sign | mag;
        return (code == 8u) ? 0u : code;
    };
    const uint8_t lo = enc(b);
    const uint8_t hi = enc(a);
    return (uint8_t)(lo | (hi << 4));
#endif
}

// eg_rubin_cvt_scaled_n1_ue8m0_ue5m3x2_f32 implements PTX ISA 9.4
// cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32 d, a, b, scale-factor;
// converting b into d[7:0] and a into d[15:8].
EG_DEV uint16_t eg_rubin_cvt_scaled_n1_ue8m0_ue5m3x2_f32(float a, float b, uint8_t ue8m0_scale) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint16_t d;
    const uint16_t sf = (uint16_t)ue8m0_scale;
    asm("{\n\t"
        ".reg .b8 s8;\n\t"
        "cvt.u8.u16 s8, %3;\n\t"
        "cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32 %0, %1, %2, s8;\n\t"
        "}" : "=h"(d) : "f"(a), "f"(b), "h"(sf));
    return d;
#else
    const float af = eg_rubin_flush_subnormal_f32(a);
    const float bf = eg_rubin_flush_subnormal_f32(b);
    const float scale = eg_rubin_from_ue8m0(ue8m0_scale);
    const uint8_t lo = eg_rubin_f32_to_ue5m3(bf / scale);
    const uint8_t hi = eg_rubin_f32_to_ue5m3(af / scale);
    return (uint16_t)lo | ((uint16_t)hi << 8);
#endif
}

// eg_rubin_cvt_scaled_n2_ue8m0_bf16x2_ue5m3x2 implements PTX ISA 9.4
// cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2 d, a, scale-factor;
EG_DEV uint32_t eg_rubin_cvt_scaled_n2_ue8m0_bf16x2_ue5m3x2(uint16_t packed_ue5m3x2, uint16_t ue8m0_pair) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint32_t d;
    asm("cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2 %0, %1, %2;"
        : "=r"(d) : "h"(packed_ue5m3x2), "h"(ue8m0_pair));
    return d;
#else
    const float v_lo = eg_rubin_ue5m3_to_f32((uint8_t)(packed_ue5m3x2 & 0xffu)) *
                       eg_rubin_from_ue8m0((uint8_t)(ue8m0_pair & 0xffu));
    const float v_hi = eg_rubin_ue5m3_to_f32((uint8_t)(packed_ue5m3x2 >> 8)) *
                       eg_rubin_from_ue8m0((uint8_t)(ue8m0_pair >> 8));
    return eg_pack_bf16x2(v_lo, v_hi);
#endif
}

// eg_rubin_add_rn_f32x2 implements PTX ISA 9.4 64-bit packed SIMD add.rn.f32x2.
EG_DEV float2 eg_rubin_add_rn_f32x2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint64_t da;
    const uint64_t pa = ((uint64_t)__float_as_uint(a.x)) | (((uint64_t)__float_as_uint(a.y)) << 32);
    const uint64_t pb = ((uint64_t)__float_as_uint(b.x)) | (((uint64_t)__float_as_uint(b.y)) << 32);
    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(da) : "l"(pa), "l"(pb));
    return float2{__uint_as_float((uint32_t)(da & 0xffffffffu)), __uint_as_float((uint32_t)(da >> 32))};
#else
    return float2{a.x + b.x, a.y + b.y};
#endif
}

// eg_rubin_fma_rn_f32x2 implements PTX ISA 9.4 64-bit packed SIMD fma.rn.f32x2.
EG_DEV float2 eg_rubin_fma_rn_f32x2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070) && defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && !defined(EG_HOST_EMU) && !defined(EG_EMU)
    uint64_t da;
    const uint64_t pa = ((uint64_t)__float_as_uint(a.x)) | (((uint64_t)__float_as_uint(a.y)) << 32);
    const uint64_t pb = ((uint64_t)__float_as_uint(b.x)) | (((uint64_t)__float_as_uint(b.y)) << 32);
    const uint64_t pc = ((uint64_t)__float_as_uint(c.x)) | (((uint64_t)__float_as_uint(c.y)) << 32);
    asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(da) : "l"(pa), "l"(pb), "l"(pc));
    return float2{__uint_as_float((uint32_t)(da & 0xffffffffu)), __uint_as_float((uint32_t)(da >> 32))};
#else
    return float2{a.x * b.x + c.x, a.y * b.y + c.y};
#endif
}

#endif // EG_RUBIN_PTX94_PRELUDE
`

func validRubinPTX94Target(t string) bool {
	return t == "sm_107f" || t == "sm_107a"
}

// RubinUE5M3ScaledQuant configures a Rubin (sm_107f) PTX ISA 9.4 fused RMSNorm
// and block-scaled UE5M3 activation quantizer/dequantizer using
// cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32,
// cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.ue5m3x2, and .pzo.
type RubinUE5M3ScaledQuant struct {
	Name   string
	Rows   int
	Cols   int // multiple of 32
	Eps    float32
	UsePZO bool
	Target string // "sm_107f"
}

func (k RubinUE5M3ScaledQuant) norm() (RubinUE5M3ScaledQuant, error) {
	if k.Name == "" {
		k.Name = "rubin_ue5m3_scaled_quant"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("rubin ue5m3 scaled quant: bad name %q", k.Name)
	}
	if k.Rows == 0 {
		k.Rows = 4
	}
	if k.Rows < 1 {
		return k, fmt.Errorf("rubin ue5m3 scaled quant %s: Rows %d must be >= 1", k.Name, k.Rows)
	}
	if k.Cols == 0 {
		k.Cols = 64
	}
	if k.Cols < 32 || k.Cols > 65536 || k.Cols%32 != 0 {
		return k, fmt.Errorf("rubin ue5m3 scaled quant %s: Cols %d must be a multiple of 32 in 32..65536", k.Name, k.Cols)
	}
	if k.Eps <= 0 {
		k.Eps = 1e-5
	}
	if k.Target == "" {
		k.Target = "sm_107f"
	}
	if !validRubinPTX94Target(k.Target) {
		return k, fmt.Errorf("rubin ue5m3 scaled quant %s: unsupported target %q (want sm_107f)", k.Name, k.Target)
	}
	return k, nil
}

func (k RubinUE5M3ScaledQuant) params() []string {
	return []string{
		"const float* __restrict__ x",
		"const float* __restrict__ w",
		"uint8_t* __restrict__ out_ue5m3",
		"uint8_t* __restrict__ out_scales",
		"float* __restrict__ out_dequant",
	}
}

// CUDA returns the Rubin (sm_107f) PTX ISA 9.4 device kernel source (without [Prelude]).
func (k RubinUE5M3ScaledQuant) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(rubinPTX94Prelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: Rubin (%s) PTX ISA 9.4 RMSNorm + UE5M3 block-scaled quantizer (Rows=%d, Cols=%d, UsePZO=%v).\n",
		k.Name, k.Target, k.Rows, k.Cols, k.UsePZO)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s(\n    %s) {\n",
		k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int ROWS = %d, COLS = %d, BLOCKS = COLS / 32;\n", k.Rows, k.Cols)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))
	b.WriteString(`    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= ROWS) return;
    __shared__ float s_norm[COLS];

    const size_t row_off = (size_t)r * COLS;
    float sum_sq = 0.0f;
    for (int c = lane; c < COLS; c += 32) {
        const float xv = x[row_off + c];
        sum_sq += xv * xv;
    }
    sum_sq = eg_warp_sum(sum_sq);
    const float inv_rms = rsqrtf(sum_sq * (1.0f / (float)COLS) + EPS);

    for (int c = lane; c < COLS; c += 32) {
        s_norm[c] = (x[row_off + c] * inv_rms) * w[c];
    }
    __syncwarp();

    for (int kb = 0; kb < BLOCKS; ++kb) {
        const int base = kb * 32;
        const float v = s_norm[base + lane];
        const float amax = eg_warp_max(fabsf(v));
        const uint8_t scale_u8 = eg_rubin_to_ue8m0(amax * (1.0f / 61440.0f));
        if (lane == 0) {
            out_scales[(size_t)r * BLOCKS + kb] = scale_u8;
        }
        if (lane < 16) {
            float y0 = s_norm[base + 2 * lane];
            float y1 = s_norm[base + 2 * lane + 1];
`)
	if k.UsePZO {
		b.WriteString(`            const uint8_t pzo_probe = eg_rubin_cvt_pzo_e2m1x2_f32(y1, y0);
            (void)pzo_probe;
            if (y0 == 0.0f) y0 = 0.0f;
            if (y1 == 0.0f) y1 = 0.0f;
`)
	}
	b.WriteString(`            const uint16_t q_pair = eg_rubin_cvt_scaled_n1_ue8m0_ue5m3x2_f32(y1, y0, scale_u8);
            out_ue5m3[row_off + base + 2 * lane] = (uint8_t)(q_pair & 0xffu);
            out_ue5m3[row_off + base + 2 * lane + 1] = (uint8_t)(q_pair >> 8);
            const uint16_t scale_pair = (uint16_t)scale_u8 | ((uint16_t)scale_u8 << 8);
            const uint32_t dq_bf16x2 = eg_rubin_cvt_scaled_n2_ue8m0_bf16x2_ue5m3x2(q_pair, scale_pair);
            out_dequant[row_off + base + 2 * lane] = eg_bf16_to_f32(dq_bf16x2);
            out_dequant[row_off + base + 2 * lane + 1] = eg_bf16_hi_to_f32(dq_bf16x2);
        }
    }
}
`)
	return b.String(), nil
}

// Launcher returns a host launcher function for [RubinUE5M3ScaledQuant].
func (k RubinUE5M3ScaledQuant) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := paramNames(params)
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    const dim3 grid(%d, 1, 1), block(32, 1, 1);\n", k.Rows)
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", k.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude], the kernel, and its launcher.
func (k RubinUE5M3ScaledQuant) Source() (string, error) {
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

// Reference executes the CPU reference for [RubinUE5M3ScaledQuant] over
// [Rows, Cols] inputs x and [Cols] weights w.
func (k RubinUE5M3ScaledQuant) Reference(x, w []float32) (quant []uint8, scales []uint8, dequant []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, nil, err
	}
	if len(x) != k.Rows*k.Cols {
		return nil, nil, nil, fmt.Errorf("rubin ue5m3 scaled quant %s reference: len(x)=%d, want %d", k.Name, len(x), k.Rows*k.Cols)
	}
	if len(w) != k.Cols {
		return nil, nil, nil, fmt.Errorf("rubin ue5m3 scaled quant %s reference: len(w)=%d, want %d", k.Name, len(w), k.Cols)
	}
	rows, cols := k.Rows, k.Cols
	blocks := cols / 32
	quant = make([]uint8, rows*cols)
	scales = make([]uint8, rows*blocks)
	dequant = make([]float32, rows*cols)

	for r := range rows {
		rowOff := r * cols
		var laneSums [32]float32
		for l := range 32 {
			var sq float32
			for c := l; c < cols; c += 32 {
				xv := x[rowOff+c]
				sq += xv * xv
			}
			laneSums[l] = sq
		}
		for mask := 16; mask > 0; mask >>= 1 {
			var next [32]float32
			for l := range 32 {
				next[l] = laneSums[l] + laneSums[l^mask]
			}
			laneSums = next
		}
		invRMS := float32(1.0 / math.Sqrt(float64(laneSums[0]*(1.0/float32(cols))+k.Eps)))

		normRow := make([]float32, cols)
		for c := range cols {
			normRow[c] = (x[rowOff+c] * invRMS) * w[c]
		}

		for kb := range blocks {
			base := kb * 32
			var amax float32
			for t := range 32 {
				if a := float32(math.Abs(float64(normRow[base+t]))); a > amax {
					amax = a
				}
			}
			scaleU8 := ToUE8M0(amax * (1.0 / MaxNormalUE5M3))
			scales[r*blocks+kb] = scaleU8
			scalePair := uint16(scaleU8) | (uint16(scaleU8) << 8)
			for i := range 16 {
				y0 := normRow[base+2*i]
				y1 := normRow[base+2*i+1]
				qPair := CvtScaledN1UE8M0ToUE5M3x2(y1, y0, scaleU8)
				quant[rowOff+base+2*i] = uint8(qPair & 0xff)
				quant[rowOff+base+2*i+1] = uint8(qPair >> 8)
				dq0, dq1 := CvtScaledN2UE8M0FromUE5M3x2(qPair, scalePair)
				dequant[rowOff+base+2*i] = dq0
				dequant[rowOff+base+2*i+1] = dq1
			}
		}
	}
	return quant, scales, dequant, nil
}

// RubinPackedF32x2ResidualRMSNorm configures a Rubin (sm_107f) PTX ISA 9.4
// kernel that fuses 2-wide r = x + residual and y = (r * rsqrt(mean(r^2) + eps)) * weight
// using packed 64-bit add.rn.f32x2 and fma.rn.f32x2 instructions.
type RubinPackedF32x2ResidualRMSNorm struct {
	Name   string
	Rows   int
	Cols   int // even, multiple of 2
	Eps    float32
	Target string // "sm_107f"
}

func (k RubinPackedF32x2ResidualRMSNorm) norm() (RubinPackedF32x2ResidualRMSNorm, error) {
	if k.Name == "" {
		k.Name = "rubin_f32x2_residual_rmsnorm"
	}
	if !cIdent(k.Name) {
		return k, fmt.Errorf("rubin packed f32x2 residual rmsnorm: bad name %q", k.Name)
	}
	if k.Rows == 0 {
		k.Rows = 4
	}
	if k.Rows < 1 {
		return k, fmt.Errorf("rubin packed f32x2 residual rmsnorm %s: Rows %d must be >= 1", k.Name, k.Rows)
	}
	if k.Cols == 0 {
		k.Cols = 64
	}
	if k.Cols < 2 || k.Cols > 65536 || k.Cols%2 != 0 {
		return k, fmt.Errorf("rubin packed f32x2 residual rmsnorm %s: Cols %d must be an even integer in 2..65536", k.Name, k.Cols)
	}
	if k.Eps <= 0 {
		k.Eps = 1e-5
	}
	if k.Target == "" {
		k.Target = "sm_107f"
	}
	if !validRubinPTX94Target(k.Target) {
		return k, fmt.Errorf("rubin packed f32x2 residual rmsnorm %s: unsupported target %q (want sm_107f)", k.Name, k.Target)
	}
	return k, nil
}

func (k RubinPackedF32x2ResidualRMSNorm) params() []string {
	return []string{
		"const float* __restrict__ x",
		"const float* __restrict__ residual",
		"const float* __restrict__ weight",
		"float* __restrict__ r_out",
		"float* __restrict__ y_out",
	}
}

// CUDA returns the Rubin (sm_107f) PTX ISA 9.4 device kernel source (without [Prelude]).
func (k RubinPackedF32x2ResidualRMSNorm) CUDA() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(rubinPTX94Prelude)
	b.WriteString("\n")
	fmt.Fprintf(&b, "// %s: Rubin (%s) PTX ISA 9.4 packed f32x2 residual RMSNorm (Rows=%d, Cols=%d).\n",
		k.Name, k.Target, k.Rows, k.Cols)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s(\n    %s) {\n",
		k.Name, strings.Join(k.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int ROWS = %d, COLS = %d, PAIRS = COLS / 2;\n", k.Rows, k.Cols)
	fmt.Fprintf(&b, "    constexpr float EPS = %s;\n", floatLiteral(k.Eps))
	b.WriteString(`    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= ROWS) return;
    __shared__ float s_r[COLS];

    const size_t row_off = (size_t)r * COLS;
    float2 sq_acc = float2{0.0f, 0.0f};
    for (int p = lane; p < PAIRS; p += 32) {
        const int c = 2 * p;
        const float2 xv = float2{x[row_off + c], x[row_off + c + 1]};
        const float2 rv = float2{residual[row_off + c], residual[row_off + c + 1]};
        const float2 h = eg_rubin_add_rn_f32x2(xv, rv);
        s_r[c] = h.x;
        s_r[c + 1] = h.y;
        r_out[row_off + c] = h.x;
        r_out[row_off + c + 1] = h.y;
        sq_acc = eg_rubin_fma_rn_f32x2(h, h, sq_acc);
    }
    const float sum_sq = eg_warp_sum(sq_acc.x + sq_acc.y);
    const float inv_rms = rsqrtf(sum_sq * (1.0f / (float)COLS) + EPS);
    const float2 inv_rms2 = float2{inv_rms, inv_rms};
    const float2 zero2 = float2{0.0f, 0.0f};

    for (int p = lane; p < PAIRS; p += 32) {
        const int c = 2 * p;
        const float2 h = float2{s_r[c], s_r[c + 1]};
        const float2 wv = float2{weight[c], weight[c + 1]};
        const float2 h_norm = eg_rubin_fma_rn_f32x2(h, inv_rms2, zero2);
        const float2 yv = eg_rubin_fma_rn_f32x2(h_norm, wv, zero2);
        y_out[row_off + c] = yv.x;
        y_out[row_off + c + 1] = yv.y;
    }
}
`)
	return b.String(), nil
}

// Launcher returns a host launcher function for [RubinPackedF32x2ResidualRMSNorm].
func (k RubinPackedF32x2ResidualRMSNorm) Launcher() (string, error) {
	k, err := k.norm()
	if err != nil {
		return "", err
	}
	params := k.params()
	names := paramNames(params)
	var b strings.Builder
	fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.Name, strings.Join(params, ", "))
	fmt.Fprintf(&b, "    const dim3 grid(%d, 1, 1), block(32, 1, 1);\n", k.Rows)
	fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", k.Name, strings.Join(names, ", "))
	return b.String(), nil
}

// Source returns a complete translation unit with [Prelude], the kernel, and its launcher.
func (k RubinPackedF32x2ResidualRMSNorm) Source() (string, error) {
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

// Reference executes the CPU reference for [RubinPackedF32x2ResidualRMSNorm]
// over [Rows, Cols] inputs x and residual and [Cols] weight.
func (k RubinPackedF32x2ResidualRMSNorm) Reference(x, residual, weight []float32) (rOut, yOut []float32, err error) {
	k, err = k.norm()
	if err != nil {
		return nil, nil, err
	}
	if len(x) != k.Rows*k.Cols || len(residual) != k.Rows*k.Cols {
		return nil, nil, fmt.Errorf("rubin packed f32x2 residual rmsnorm %s reference: len(x)=%d len(residual)=%d, want %d",
			k.Name, len(x), len(residual), k.Rows*k.Cols)
	}
	if len(weight) != k.Cols {
		return nil, nil, fmt.Errorf("rubin packed f32x2 residual rmsnorm %s reference: len(weight)=%d, want %d",
			k.Name, len(weight), k.Cols)
	}
	rows, cols := k.Rows, k.Cols
	pairs := cols / 2
	rOut = make([]float32, rows*cols)
	yOut = make([]float32, rows*cols)

	for r := range rows {
		rowOff := r * cols
		var laneSums [32]float32
		for l := range 32 {
			var sqX, sqY float32
			for p := l; p < pairs; p += 32 {
				c := 2 * p
				hx := x[rowOff+c] + residual[rowOff+c]
				hy := x[rowOff+c+1] + residual[rowOff+c+1]
				rOut[rowOff+c] = hx
				rOut[rowOff+c+1] = hy
				sqX += hx * hx
				sqY += hy * hy
			}
			laneSums[l] = sqX + sqY
		}
		for mask := 16; mask > 0; mask >>= 1 {
			var next [32]float32
			for l := range 32 {
				next[l] = laneSums[l] + laneSums[l^mask]
			}
			laneSums = next
		}
		invRMS := float32(1.0 / math.Sqrt(float64(laneSums[0]*(1.0/float32(cols))+k.Eps)))
		for c := range cols {
			yOut[rowOff+c] = (rOut[rowOff+c] * invRMS) * weight[c]
		}
	}
	return rOut, yOut, nil
}
