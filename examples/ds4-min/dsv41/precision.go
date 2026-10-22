package dsv41

import (
	"fmt"
	"math"
)

// Format specifies the released V4.1 simulated activation encoding.
type Format uint8

const (
	BFloat16 Format = iota
	FP8E8M0
	FP4E8M0
	FP4E4M3
)

// Quantize rounds activations in place, independently for each row and scale
// block. FP8/FP4 E8M0 blocks have 32 elements; FP4 E4M3 blocks have 16.
// Invalid shape or non-finite input leaves x unchanged.
func Quantize(x []float32, width int, format Format) error {
	block := 32
	if format == FP4E4M3 {
		block = 16
	}
	if width <= 0 || len(x) == 0 || len(x)%width != 0 || format > FP4E4M3 || format != BFloat16 && width%block != 0 {
		return fmt.Errorf("invalid activation shape or format")
	}
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("non-finite activation")
		}
	}
	if format == BFloat16 {
		for i := range x {
			x[i] = BF16(x[i])
		}
		return nil
	}
	for start := 0; start < len(x); start += block {
		peak := float32(0)
		for _, v := range x[start : start+block] {
			peak = max(peak, float32(math.Abs(float64(BF16(v)))))
		}
		var scale float32
		switch format {
		case FP8E8M0:
			scale = pow2ceil(float32(max(peak, 1e-4) * float32(1.0/448)))
		case FP4E8M0:
			scale = pow2ceil(float32(max(peak, 0x1.8p-124) * float32(1.0/6)))
		case FP4E4M3:
			scale = roundMini(float32(max(peak, 0.01171875)/6), false)
		}
		for i := start; i < start+block; i++ {
			v := BF16(x[i])
			q := roundMini(float32(float32(math.Abs(float64(v)))/scale), format != FP8E8M0)
			x[i] = BF16(float32(float32(math.Copysign(float64(q), float64(v))) * scale))
		}
	}
	return nil
}

func pow2ceil(x float32) float32 {
	b := math.Float32bits(x)
	e := b & 0x7f800000
	if b&0x7fffff != 0 {
		e += 0x800000
	}
	return math.Float32frombits(e)
}

func miniValue(i int, fp4 bool) float32 {
	if fp4 {
		return [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6}[i]
	}
	e, m := i>>3, i&7
	if e == 0 {
		return float32(math.Ldexp(float64(m), -9))
	}
	return float32(math.Ldexp(float64(8+m), e-10))
}

func roundMini(x float32, fp4 bool) float32 {
	n := 127
	if fp4 {
		n = 8
	}
	best := 0
	distance := float32(math.Abs(float64(x)))
	for i := 1; i < n; i++ {
		d := float32(math.Abs(float64(float32(x - miniValue(i, fp4)))))
		if d < distance || d == distance && i&1 == 0 && best&1 != 0 {
			best = i
			distance = d
		}
	}
	return miniValue(best, fp4)
}
