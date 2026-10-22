package dsv41

import (
	"fmt"
	"math"
)

// AddEngram applies the four-head gate to a projected Engram row. kv is
// [four keys, one value], each width elements. qWeight and kWeight are four
// width-element heads. Masked image positions must not call this operation.
// Shape errors leave residual unchanged. The warp reduction order follows
// ds4_deepseek41_cuda.cuh at 11e8cad; host transcendental math is not a device
// bit-equivalence claim.
func AddEngram(residual, kv, qWeight, kWeight []float32, width int, eps float32) error {
	if width <= 0 || width > math.MaxInt/5 || len(residual) != 4*width || len(kv) != 5*width || len(qWeight) != 4*width || len(kWeight) != 4*width || eps <= 0 || math.IsInf(float64(eps), 0) || math.IsNaN(float64(eps)) {
		return fmt.Errorf("invalid engram shape or epsilon")
	}
	for head := 0; head < 4; head++ {
		var h2, k2, dot [32]float32
		for lane := 0; lane < 32; lane++ {
			for i := lane; i < width; i += 32 {
				j := head*width + i
				h, k := residual[j], BF16(kv[j])
				h2[lane] += h * h
				k2[lane] += k * k
				dot[lane] += float32(float32(h*float32(qWeight[j]*kWeight[j])) * k)
			}
		}
		for delta := 16; delta > 0; delta /= 2 {
			for lane := 0; lane < delta; lane++ {
				h2[lane] += h2[lane+delta]
				k2[lane] += k2[lane+delta]
				dot[lane] += dot[lane+delta]
			}
		}
		d := dot[0] * float32(1/math.Sqrt(float64(h2[0]/float32(width)+eps)))
		d *= float32(1 / math.Sqrt(float64(k2[0]/float32(width)+eps)))
		d *= float32(1 / math.Sqrt(float64(width)))
		s := float32(math.Copysign(math.Sqrt(math.Max(math.Abs(float64(d)), 1e-6)), float64(d)))
		gate := float32(1) / float32(1+float32(math.Exp(float64(-s))))
		for i := 0; i < width; i++ {
			j := head*width + i
			residual[j] = BF16(residual[j] + gate*BF16(kv[4*width+i]))
		}
	}
	return nil
}
