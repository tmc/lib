package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

// A WeightDType names the precision model weights are stored in on the device.
//
// It is a storage choice only. Activations and every accumulation stay
// float32 in all modes, so the difference between WeightF32 and a half mode is
// exactly the precision each weight was rounded to before it was multiplied.
// That separation is what lets quantization error be reported apart from
// implementation error.
type WeightDType int

const (
	// WeightF32 is the reference mode: four-byte weights, unchanged behavior.
	WeightF32 WeightDType = iota
	WeightF16
	WeightBF16
)

// ParseWeightDType returns the mode named by s.
//
// The mode is always selected explicitly. There is no automatic promotion and
// no fallback: a request for a half mode that cannot be satisfied is an error,
// so a run that reports F16 never turns out to have executed in F32.
func ParseWeightDType(s string) (WeightDType, error) {
	switch s {
	case "", "f32", "F32", "float32":
		return WeightF32, nil
	case "f16", "F16", "float16", "half":
		return WeightF16, nil
	case "bf16", "BF16", "bfloat16":
		return WeightBF16, nil
	}
	return WeightF32, fmt.Errorf("unknown weight dtype %q, want f32, f16, or bf16", s)
}

// String returns the mode's name as ParseWeightDType accepts it.
func (d WeightDType) String() string {
	switch d {
	case WeightF32:
		return "f32"
	case WeightF16:
		return "f16"
	case WeightBF16:
		return "bf16"
	}
	return fmt.Sprintf("WeightDType(%d)", int(d))
}

// Bytes returns the device storage size of one weight in this mode.
func (d WeightDType) Bytes() int {
	if d == WeightF32 {
		return 4
	}
	return 2
}

// Suffix returns the kernel name suffix for this mode.
//
// The float32 kernels keep their original unsuffixed names, so nothing about
// the reference path depends on this dispatch existing.
func (d WeightDType) Suffix() string {
	switch d {
	case WeightF16:
		return "F16"
	case WeightBF16:
		return "BF16"
	}
	return ""
}

// CheckpointDType names the safetensors dtype this mode stores, so that a
// checkpoint already in that dtype can be uploaded without any conversion.
func (d WeightDType) CheckpointDType() string {
	switch d {
	case WeightF16:
		return "F16"
	case WeightBF16:
		return "BF16"
	}
	return "F32"
}

// EncodeWeights returns src encoded in d's storage format.
//
// This is the rounding step, and it happens at most once per weight: a
// checkpoint already stored in d's dtype is passed through by the caller
// without reaching this function, so no weight is ever rounded twice. The
// rounding is round-to-nearest-even, which is what the hardware conversion
// instructions do, so the CPU reference and the device see identical weights.
func (d WeightDType) EncodeWeights(src []float32) []byte {
	out := make([]byte, len(src)*d.Bytes())
	switch d {
	case WeightF32:
		for i, v := range src {
			binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
		}
	case WeightF16:
		for i, v := range src {
			binary.LittleEndian.PutUint16(out[i*2:], float32ToFloat16(v))
		}
	case WeightBF16:
		for i, v := range src {
			binary.LittleEndian.PutUint16(out[i*2:], float32ToBfloat16(v))
		}
	}
	return out
}

// DecodeWeights returns the float32 values the device will actually see for
// weights stored as raw.
//
// It is the exact inverse of the device's widening step, so it is what a CPU
// reference must multiply by in order to be comparing implementations rather
// than precisions.
func (d WeightDType) DecodeWeights(raw []byte) []float32 {
	n := len(raw) / d.Bytes()
	out := make([]float32, n)
	switch d {
	case WeightF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case WeightF16:
		for i := range out {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case WeightBF16:
		for i := range out {
			out[i] = bfloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	}
	return out
}

// RoundTrip returns src as the device will see it in mode d.
//
// A CPU reference for a half kernel multiplies by these values, not by src.
// Comparing against src would fold the quantization error into the comparison
// and make a correct kernel look wrong.
func (d WeightDType) RoundTrip(src []float32) []float32 {
	return d.DecodeWeights(d.EncodeWeights(src))
}

// float32ToFloat16 rounds f to IEEE binary16 with round-to-nearest-even.
//
// Overflow becomes an infinity of the right sign rather than the largest
// finite value, underflow becomes a zero of the right sign, and a NaN stays a
// NaN. These are the conversions cvt.rn.f16.f32 performs, so the encoded
// weights match what the device would have produced.
func float32ToFloat16(f float32) uint16 {
	u := math.Float32bits(f)
	sign := uint16((u >> 16) & 0x8000)
	biased := (u >> 23) & 0xff
	mant := u & 0x7fffff

	if biased == 0xff { // infinity or NaN
		if mant != 0 {
			return sign | 0x7e00 // a quiet NaN; the payload does not survive
		}
		return sign | 0x7c00
	}
	exp := int32(biased) - 127
	switch {
	case exp > 15:
		return sign | 0x7c00 // overflows the half range
	case exp >= -14:
		q := uint16(mant >> 13)
		r := mant & 0x1fff
		h := sign | uint16((exp+15)<<10) | q
		if r > 0x1000 || (r == 0x1000 && q&1 == 1) {
			h++ // may carry into the exponent, and at the top into infinity
		}
		return h
	case exp >= -25:
		// Subnormal: the implicit leading bit becomes explicit and the whole
		// significand shifts down to the fixed 2**-24 scale.
		m := mant | 0x800000
		shift := uint32(-exp - 1)
		q := m >> shift
		r := m & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if r > half || (r == half && q&1 == 1) {
			q++
		}
		return sign | uint16(q)
	}
	return sign // underflows to a signed zero
}

// float16ToFloat32 widens an IEEE binary16 to float32 exactly.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal: shift until the leading bit reaches the implicit
		// position, then bias the exponent by how far it moved. float32 has
		// the range to hold every half subnormal as a normal value.
		k := uint32(0)
		for mant&0x400 == 0 {
			mant <<= 1
			k++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (113-k)<<23 | mant<<13)
	case 0x1f:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000)
		}
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}

// float32ToBfloat16 rounds f to bfloat16 with round-to-nearest-even.
//
// bfloat16 is the top 16 bits of a float32, so this only decides what happens
// to the 16 bits being discarded.
func float32ToBfloat16(f float32) uint16 {
	u := math.Float32bits(f)
	if u&0x7fffffff > 0x7f800000 { // NaN stays a NaN rather than rounding to infinity
		return uint16(u>>16) | 0x0040
	}
	return uint16((u + 0x7fff + ((u >> 16) & 1)) >> 16)
}

// bfloat16ToFloat32 widens a bfloat16 to float32 exactly.
func bfloat16ToFloat32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// Valid reports whether d names a real storage mode.
//
// It exists because an out-of-range WeightDType is not merely unnamed: Bytes
// returns 2 for it and Suffix returns the empty string, so it would select
// two-byte device storage and then bind the unsuffixed float32 kernels, which
// read those bytes as float32. Every boundary that allocates or compiles checks
// this, not just the command-line parser, so a value that reached the engine by
// any other route is refused before it can do that.
func (d WeightDType) Valid() bool {
	return d == WeightF32 || d == WeightF16 || d == WeightBF16
}

// check returns an error unless d names a real storage mode.
func (d WeightDType) check() error {
	if !d.Valid() {
		return fmt.Errorf("invalid weight dtype %d", int(d))
	}
	return nil
}
