package dsv4cpu

import (
	"errors"
	"math"
)

// QuantizeIndexer applies the 128-wide normalized Hadamard and E2M1 FP4
// activation round trip used by both indexer queries and compressed KV.
// Each row is contiguous; x must contain complete finite rows.
// An overflow error can leave x partially transformed.
func QuantizeIndexer(x []float32) error {
	if len(x) == 0 || len(x)%128 != 0 {
		return errors.New("invalid indexer quantization shape")
	}
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("nonfinite indexer input")
		}
	}
	for off := 0; off < len(x); off += 128 {
		row := x[off : off+128]
		for stride := 1; stride < 128; stride *= 2 {
			for base := 0; base < 128; base += 2 * stride {
				for i := 0; i < stride; i++ {
					a, b := row[base+i], row[base+stride+i]
					row[base+i] = a + b
					row[base+stride+i] = a - b
				}
			}
		}
		for i := range row {
			row[i] *= 0.08838834764831845
			if math.IsNaN(float64(row[i])) || math.IsInf(float64(row[i]), 0) {
				return errors.New("nonfinite Hadamard output")
			}
		}
		for block := 0; block < 128; block += 32 {
			amax := float32(7.052966104933725e-38)
			for _, v := range row[block : block+32] {
				a := float32(math.Abs(float64(v)))
				if a > amax {
					amax = a
				}
			}
			exponent := int(math.Ceil(float64(float32(math.Log2(float64(float32(amax / 6)))))))
			scale := float32(math.Ldexp(1, exponent))
			for i := block; i < block+32; i++ {
				row[i] = e2m1Round(row[i]/scale) * scale
			}
		}
	}
	return nil
}

func e2m1Round(x float32) float32 {
	values := [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}
	sign := float32(1)
	if x < 0 {
		sign = -1
	}
	a := float32(math.Abs(float64(x)))
	if a > 6 {
		a = 6
	}
	best := 0
	diff := float32(math.Abs(float64(a - values[0])))
	for i := 1; i < len(values); i++ {
		d := float32(math.Abs(float64(a - values[i])))
		if d < diff || (d == diff && i&1 == 0 && best&1 != 0) {
			best = i
			diff = d
		}
	}
	return sign * values[best]
}
