package dsv4cpu

import (
	"errors"
	"math"
)

// Q8K stores the native CPU activation block used by routed IQ2/Q2 experts.
type Q8K struct {
	Scale  float32
	Values [256]int8
	Sums   [16]int16
}

// QuantizeQ8K implements the signed-maximum upstream activation quantizer.
// Ties in rounding are even; input must consist of whole finite blocks whose
// nonzero signed maximum yields a finite quantizer scale. Non-representable
// scales are refused instead of converting infinities to integers.
func QuantizeQ8K(dst []Q8K, x []float32) error {
	if len(x) == 0 || len(x)%256 != 0 || len(dst) != len(x)/256 {
		return errors.New("invalid Q8_K activation shape")
	}
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("nonfinite Q8_K activation")
		}
	}
	for b := range dst {
		row := x[b*256 : (b+1)*256]
		max, amax := float32(0), float32(0)
		for _, v := range row {
			a := float32(math.Abs(float64(v)))
			if a > amax {
				amax, max = a, v
			}
		}
		dst[b] = Q8K{}
		if amax == 0 {
			continue
		}
		scale := float32(-127) / max
		if math.IsInf(float64(scale), 0) {
			return errors.New("Q8_K quantizer scale overflows")
		}
		for j, v := range row {
			q := int(math.RoundToEven(float64(float32(scale * v))))
			if q > 127 {
				q = 127
			}
			if q < -128 {
				q = -128
			}
			dst[b].Values[j] = int8(q)
			dst[b].Sums[j/16] += int16(q)
		}
		dst[b].Scale = 1 / scale
	}
	return nil
}

// SwiGLU evaluates the expert activation. Only gate's upper bound is clamped;
// up is clamped symmetrically, as in the upstream CPU implementation.
func SwiGLU(dst, gate, up []float32, clamp float32) error {
	if len(dst) == 0 || len(dst) != len(gate) || len(dst) != len(up) {
		return errors.New("invalid SwiGLU shape")
	}
	for i, g := range gate {
		u := up[i]
		if clamp > 1e-6 {
			if g > clamp {
				g = clamp
			}
			if u > clamp {
				u = clamp
			}
			if u < -clamp {
				u = -clamp
			}
		}
		sigmoid := float32(0)
		if g >= 0 {
			e := float32(math.Exp(float64(-g)))
			sigmoid = 1 / (1 + e)
		} else {
			e := float32(math.Exp(float64(g)))
			sigmoid = e / (1 + e)
		}
		silu := g * sigmoid
		dst[i] = silu * u
	}
	return nil
}
