package dsv4cpu

import (
	"errors"
	"math"
)

// QuantizeKV applies the model's E4M3 round trip to the nonrotary coordinates
// of one KV row. Rotary coordinates are unchanged. The nonrotary width must
// be a multiple of 64; x must contain only finite values.
func QuantizeKV(x []float32, rotary int) error {
	if len(x) == 0 || rotary < 0 || rotary > len(x) || (len(x)-rotary)%64 != 0 {
		return errors.New("invalid KV quantization shape")
	}
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("nonfinite KV value")
		}
	}
	for off := 0; off < len(x)-rotary; off += 64 {
		amax := float32(1e-4)
		for _, v := range x[off : off+64] {
			a := float32(math.Abs(float64(v)))
			if a > amax {
				amax = a
			}
		}
		exponent := int(math.Ceil(float64(float32(math.Log2(float64(float32(amax / 448)))))))
		scale := float32(math.Ldexp(1, exponent))
		for i := off; i < off+64; i++ {
			x[i] = e4m3Round(x[i]/scale) * scale
		}
	}
	return nil
}

func e4m3Value(i int) float32 {
	exp, mant := (i>>3)&15, i&7
	if exp == 0 {
		return float32(mant) * 0.001953125
	}
	return float32(1+float32(mant)*0.125) * float32(math.Ldexp(1, exp-7))
}

func e4m3Round(x float32) float32 {
	sign := float32(1)
	if x < 0 {
		sign = -1
	}
	a := float32(math.Abs(float64(x)))
	if a > 448 {
		a = 448
	}
	lo, hi := 0, 126
	for lo < hi {
		mid := (lo + hi + 1) >> 1
		if e4m3Value(mid) <= a {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	best := lo
	if best < 126 {
		d := float32(math.Abs(float64(a - e4m3Value(best))))
		n := float32(math.Abs(float64(a - e4m3Value(best+1))))
		if n < d || (n == d && (best+1)&1 == 0 && best&1 != 0) {
			best++
		}
	}
	return sign * e4m3Value(best)
}
