package main

// Host-side half of the FP8 KV path: the switches that select it and the
// reference E4M3 codec the device kernels mirror bit for bit. Nothing here
// touches a device, so this file has no build tag and its tests run on every
// host. The device half is fp8_kv_cuda.go.

import (
	"fmt"
	"math"
	"os"
	"strconv"
)

// fp8KVEnabled reports whether the FP8 KV path was explicitly requested. Like
// the paged switch it is opt-in through the environment, never a Config field.
func fp8KVEnabled() bool { return os.Getenv("LLAMA_INFER_FP8_KV") == "1" }

// fp8KVScale reads LLAMA_INFER_FP8_KV_SCALE, the factor every K and V element
// is divided by before it is rounded to E4M3 and multiplied by when read back.
// One factor serves every head of every layer; a calibrated per-head factor
// would need a pass over real activations that has not been made. E4M3 holds
// magnitudes up to 448, so the default of 1 saturates anything larger.
func fp8KVScale() (float32, error) {
	s := os.Getenv("LLAMA_INFER_FP8_KV_SCALE")
	if s == "" {
		return 1, nil
	}
	f, err := strconv.ParseFloat(s, 32)
	if err != nil || !(f > 0) || math.IsInf(f, 0) {
		return 0, fmt.Errorf("LLAMA_INFER_FP8_KV_SCALE=%q: want a positive finite number", s)
	}
	return float32(f), nil
}

// fp8E4M3Max is the largest finite E4M3 magnitude.
const fp8E4M3Max = 448

// fp8E4M3 rounds x to the nearest OCP FP8 E4M3 code, ties to even. Finite
// magnitudes of 448 and above, infinities included, saturate to ±448; NaN
// maps to the NaN code. The device kernel in fp8_kv_cuda.go performs the same
// operations on the same bits, so the two agree exactly.
func fp8E4M3(x float32) uint8 {
	b := math.Float32bits(x)
	sign := uint8(b>>31) << 7
	b &= 0x7fffffff
	if b > 0x7f800000 {
		return sign | 0x7f
	}
	if b >= 0x43e00000 {
		return sign | 0x7e
	}
	exp, mant := b>>23, b&0x7fffff
	if exp >= 121 {
		// Normal: E4M3 exponent exp-120 in 1..15, three mantissa bits kept.
		code := (exp-120)<<3 | mant>>20
		rem := mant & 0xfffff
		if rem > 0x80000 || (rem == 0x80000 && code&1 == 1) {
			code++
		}
		return sign | uint8(code)
	}
	// Subnormal: the value in units of 2^-9 is 1.mant * 2^(exp-118), so the
	// 24-bit significand shifts right by 141-exp. A carry into 8 is the
	// smallest normal, which the code 0x08 already denotes.
	shift := 141 - exp
	if shift >= 25 {
		return sign
	}
	full := mant | 0x800000
	m := full >> shift
	rem := full & (1<<shift - 1)
	half := uint32(1) << (shift - 1)
	if rem > half || (rem == half && m&1 == 1) {
		m++
	}
	return sign | uint8(m)
}

// fp8E4M3Value decodes an E4M3 code. Every finite code is exact in float32.
func fp8E4M3Value(c uint8) float32 {
	e, m := int(c>>3)&0xf, int(c&7)
	var v float32
	switch {
	case e == 0:
		v = float32(m) / 512
	case e == 15 && m == 7:
		v = float32(math.NaN())
	default:
		v = float32(math.Ldexp(float64(8+m), e-10))
	}
	if c&0x80 != 0 {
		v = -v
	}
	return v
}

// fp8KVRoundTrip is what one cache element reads back as after it was stored
// with scale: the reference for the device store and dequantize kernels.
func fp8KVRoundTrip(x, scale float32) float32 {
	return fp8E4M3Value(fp8E4M3(x/scale)) * scale
}
