package dsv41

import (
	"fmt"
	"math"
)

// rotary applies the V4.1 adjacent-pair BF16 RoPE, including the compressed
// YaRN frequency table. It is distinct from V4's split-half rotary pairing.
func rotary(x []float32, width, position int, compressed, inverse bool) error {
	if width < 64 || len(x) == 0 || len(x)%width != 0 || position < 0 || position >= 1048576 {
		return fmt.Errorf("invalid V4.1 rotary shape")
	}
	base := float32(10000)
	if compressed {
		base = 160000
	}
	low := float32(math.Floor(64 * math.Log(65536/(32*2*math.Pi)) / (2 * math.Log(float64(base)))))
	high := float32(math.Ceil(64 * math.Log(65536/(2*math.Pi)) / (2 * math.Log(float64(base)))))
	for lane := 0; lane < 32; lane++ {
		frequency := float32(1) / float32(math.Pow(float64(base), float64(float32(lane)/32)))
		if compressed {
			ramp := min(float32(1), max(float32(0), (float32(lane)-low)/(high-low)))
			smooth := 1 - ramp
			frequency = float32(float32(frequency/16)*float32(1-smooth)) + float32(frequency*smooth)
		}
		theta := float32(float32(position) * frequency)
		c, s := float32(math.Cos(float64(theta))), float32(math.Sin(float64(theta)))
		if inverse {
			s = -s
		}
		for head := 0; head < len(x)/width; head++ {
			i := head*width + width - 64 + lane*2
			re, im := x[i], x[i+1]
			x[i] = BF16(float32(float32(re*c) - float32(im*s)))
			x[i+1] = BF16(float32(float32(re*s) + float32(im*c)))
		}
	}
	return nil
}

func indexScores(q, weights, keys []float32) ([]float32, error) {
	if len(q) != 32*128 || len(weights) != 32 || len(keys)%128 != 0 {
		return nil, fmt.Errorf("invalid V4.1 index score shape")
	}
	out := make([]float32, len(keys)/128)
	for key := range out {
		var total float32
		for head := 0; head < 32; head++ {
			var lanes [32]float32
			for lane := 0; lane < 32; lane++ {
				qi, ki := head*128+lane, key*128+lane
				a := float32(float32(q[qi]*keys[ki]) + float32(q[qi+64]*keys[ki+64]))
				b := float32(float32(q[qi+32]*keys[ki+32]) + float32(q[qi+96]*keys[ki+96]))
				lanes[lane] = a + b
			}
			for delta := 16; delta > 0; delta /= 2 {
				for lane := 0; lane < delta; lane++ {
					lanes[lane] += lanes[lane+delta]
				}
			}
			total += float32(max(lanes[0], 0) * weights[head])
		}
		out[key] = total / 64
	}
	return out, nil
}
