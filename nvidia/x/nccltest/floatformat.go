package nccltest

import "math"

// floatFormat describes the remaining NCCL floating representations. FP8
// conversion follows CUDA's saturate-finite constructors; IEEE formats overflow
// to infinity. E4M3 reserves only the all-one magnitude for NaN.
type floatFormat struct {
	exponentBits, fractionBits uint
	bias                       int
	finiteOnly, saturate       bool
}

var (
	float16Format    = floatFormat{5, 10, 15, false, false}
	bfloat16Format   = floatFormat{8, 7, 127, false, false}
	float8e4m3Format = floatFormat{4, 3, 7, true, true}
	float8e5m2Format = floatFormat{5, 2, 15, false, true}
)

func (f floatFormat) maxFiniteBits() uint16 {
	exponent := uint16(1<<f.exponentBits) - 1
	fraction := uint16(1<<f.fractionBits) - 1
	if f.finiteOnly {
		fraction--
	} else {
		exponent--
	}
	return exponent<<f.fractionBits | fraction
}

func (f floatFormat) decode(bits uint16) float64 {
	exponentMask := uint16(1<<f.exponentBits) - 1
	fractionMask := uint16(1<<f.fractionBits) - 1
	exponent, fraction := (bits>>f.fractionBits)&exponentMask, bits&fractionMask
	sign := 1.0
	if bits&(1<<(f.exponentBits+f.fractionBits)) != 0 {
		sign = -1
	}
	if exponent == exponentMask {
		if (f.finiteOnly && fraction == fractionMask) || (!f.finiteOnly && fraction != 0) {
			return math.Copysign(math.NaN(), sign)
		}
		if !f.finiteOnly {
			return math.Copysign(math.Inf(1), sign)
		}
	}
	var value float64
	if exponent == 0 {
		value = math.Ldexp(float64(fraction), 1-f.bias-int(f.fractionBits))
	} else {
		value = math.Ldexp(1+math.Ldexp(float64(fraction), -int(f.fractionBits)), int(exponent)-f.bias)
	}
	return math.Copysign(value, sign)
}

func (f floatFormat) encode(value float64) uint16 {
	var sign uint16
	if math.Signbit(value) {
		sign = 1 << (f.exponentBits + f.fractionBits)
	}
	exponentMask := uint16(1<<f.exponentBits) - 1
	if math.IsNaN(value) {
		fraction := uint16(1 << (f.fractionBits - 1))
		if f.finiteOnly {
			fraction = uint16(1<<f.fractionBits) - 1
		}
		return sign | exponentMask<<f.fractionBits | fraction
	}
	value = math.Abs(value)
	maximum := f.maxFiniteBits()
	if f.saturate && value >= f.decode(maximum) {
		return sign | maximum
	}
	if math.IsInf(value, 0) {
		return sign | exponentMask<<f.fractionBits
	}
	if value == 0 {
		return sign
	}
	if !f.saturate {
		maxExponent := int(maximum>>f.fractionBits) - f.bias
		threshold := f.decode(maximum) + math.Ldexp(1, maxExponent-int(f.fractionBits)-1)
		if value >= threshold {
			return sign | exponentMask<<f.fractionBits
		}
	}
	_, exponent := math.Frexp(value)
	exponent = max(exponent-1, 1-f.bias)
	mantissa := math.RoundToEven(math.Ldexp(value, int(f.fractionBits)-exponent))
	base := math.Ldexp(1, int(f.fractionBits))
	if mantissa < base {
		return sign | uint16(mantissa)
	}
	if mantissa == 2*base {
		mantissa /= 2
		exponent++
	}
	return sign | uint16(exponent+f.bias)<<f.fractionBits | uint16(mantissa-base)
}
