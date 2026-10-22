package dsv4cpu

import (
	"errors"
	"math"
)

// RoPE describes the tail-only rotary position encoding for one layer.
// Dense Flash layers use Base 10000 and Scale 1. Compressed Flash layers use
// Base 160000, Scale 1/16, OriginalContext 65536, BetaFast 32, BetaSlow 1.
type RoPE struct {
	Base, Scale        float32
	OriginalContext    int
	BetaFast, BetaSlow float32
}

// Rotate rotates the last rotary coordinates of each contiguous width-value
// head in x. Inverse reverses the rotation before the output projection.
func (r RoPE) Rotate(x []float32, width, rotary, position int, inverse bool) error {
	if width <= 0 || rotary <= 0 || rotary > width || rotary%2 != 0 || len(x) == 0 || len(x)%width != 0 || position < 0 || !(r.Base > 1) || !(r.Scale > 0) || math.IsInf(float64(r.Base), 0) || math.IsInf(float64(r.Scale), 0) {
		return errors.New("invalid rotary position shape")
	}
	compressed := r.Scale < 1
	if compressed && (r.OriginalContext <= 0 || !(r.BetaFast > 0) || !(r.BetaSlow > 0) || math.IsInf(float64(r.BetaFast), 0) || math.IsInf(float64(r.BetaSlow), 0)) {
		return errors.New("invalid rotary interpolation")
	}
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("nonfinite rotary input")
		}
	}
	thetaScale := float32(math.Pow(float64(r.Base), float64(float32(-2/float32(rotary)))))
	sign := float32(1)
	if inverse {
		sign = -1
	}
	var low, high float32
	mscale := float32(1)
	if compressed {
		corr := func(beta float32) float32 {
			numerator := float32(float32(rotary) * float32(math.Log(float64(float32(float32(r.OriginalContext)/float32(float32(beta*2)*float32(math.Pi)))))))
			return numerator / float32(2*float32(math.Log(float64(r.Base))))
		}
		low = float32(math.Floor(float64(corr(r.BetaFast))))
		if low < 0 {
			low = 0
		}
		high = float32(math.Ceil(float64(corr(r.BetaSlow))))
		if high > float32(rotary-1) {
			high = float32(rotary - 1)
		}
		// Match the source's inverse magnitude factor and its cancellation.
		factor := float32(1 + float32(0.1*float32(math.Log(float64(float32(1/r.Scale))))))
		mscale = float32(float32(1/factor) * factor)
	}
	for off := 0; off < len(x); off += width {
		tail := x[off+width-rotary : off+width]
		thetaExtrap := float32(position)
		for i := 0; i < rotary; i += 2 {
			thetaInterp := float32(r.Scale * thetaExtrap)
			theta := thetaInterp
			if compressed {
				den := high - low
				if den < 0.001 {
					den = 0.001
				}
				y := float32(float32(float32(i/2)-low) / den)
				if y < 0 {
					y = 0
				}
				if y > 1 {
					y = 1
				}
				mix := float32(1 - y)
				theta = float32(float32(thetaInterp*float32(1-mix)) + float32(thetaExtrap*mix))
			}
			c := float32(float32(math.Cos(float64(theta))) * mscale)
			s := float32(float32(sign*float32(math.Sin(float64(theta)))) * mscale)
			x0, x1 := tail[i], tail[i+1]
			tail[i] = float32(float32(x0*c) - float32(x1*s))
			tail[i+1] = float32(float32(x0*s) + float32(x1*c))
			thetaExtrap = float32(thetaExtrap * thetaScale)
		}
	}
	return nil
}
