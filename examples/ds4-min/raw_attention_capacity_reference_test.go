package main

import (
	"math"
	"testing"
)

// These references use chronological rows, not ring indices or shader helpers.
func rawCapacityRound(x float32, fraction, minExponent int) float32 {
	if x == 0 {
		return x
	}
	_, exponent := math.Frexp(math.Abs(float64(x)))
	step := math.Ldexp(1, max(exponent-1, minExponent)-fraction)
	return float32(math.RoundToEven(float64(x)/step) * step)
}

func rawCapacityFP8(x float32) float32 {
	value := math.Min(math.Abs(float64(x)), 448)
	best, distance := 0, math.Inf(1)
	decode := func(code int) float64 {
		e, m := code>>3, code&7
		if e == 0 {
			return float64(m) / 512
		}
		return math.Ldexp(1+float64(m)/8, e-7)
	}
	for code := 0; code <= 126; code++ {
		d := math.Abs(value - decode(code))
		if d < distance || d == distance && code%2 == 0 {
			best, distance = code, d
		}
	}
	return float32(math.Copysign(decode(best), float64(x)))
}

func rawCapacityRotate(values []float32, position int, inverse bool) {
	theta := float32(position)
	step := float32(math.Pow(10000, -2.0/64))
	for i := 448; i < 512; i += 2 {
		s, c := math.Sincos(float64(theta))
		if inverse {
			s = -s
		}
		a, b := float64(values[i]), float64(values[i+1])
		values[i], values[i+1] = float32(a*c-b*s), float32(a*s+b*c)
		theta *= step
	}
}

func rawCapacityStoredRow(input []float32) []float32 {
	row := append([]float32(nil), input...)
	for off := 0; off < 448; off += 64 {
		amax := 1e-4
		for _, x := range row[off : off+64] {
			amax = math.Max(amax, math.Abs(float64(x)))
		}
		scale := float32(math.Exp2(math.Ceil(math.Log2(amax / 448))))
		for i := off; i < off+64; i++ {
			row[i] = rawCapacityFP8(row[i]/scale) * scale
		}
	}
	for i := range row {
		row[i] = rawCapacityRound(row[i], 10, -14)
	}
	return row
}

func rawCapacityInputs(position int) (query, current, sinks []float32) {
	query, current, sinks = make([]float32, 64*512), make([]float32, 512), make([]float32, 64)
	for head := range 64 {
		sinks[head] = float32(head%9-4) / 4
		for i := 0; i < 448; i++ {
			query[head*512+i] = float32((i*7+head*3)%17-8) / 16
		}
		// Start from half-lattice centers and analytically invert the rotation.
		// This exercises every frequency while keeping the consumed operands
		// away from half-rounding bin boundaries and signed-zero artifacts.
		for i := 448; i < 512; i++ {
			query[head*512+i] = float32(head%4+1) / 2
		}
		rawCapacityRotate(query[head*512:(head+1)*512], position, true)
	}
	for i := 0; i < 448; i++ {
		current[i] = float32((i*11+position*3)%25-12) / 8
	}
	for i := 448; i < 512; i++ {
		current[i] = 0.5
	}
	rawCapacityRotate(current, position, true)
	return
}

func rawCapacityHistory(position, capacity int) [][]float32 {
	count := min(position, capacity-1)
	history := make([][]float32, count)
	for index := range count {
		row := make([]float32, 512)
		absolute := position - count + index
		for i := 0; i < 448; i++ {
			row[i] = float32((i*5+absolute*7)%31-15) / 8
		}
		row[450] = float32(absolute%7-3) / 8
		row[451] = float32(absolute%5-2) / 8
		history[index] = rawCapacityStoredRow(row)
	}
	return history
}

func rawCapacityAttention(query []float32, history [][]float32, sinks []float32, position int) (output, scores []float32) {
	output, scores = make([]float32, 64*512), make([]float32, 64*len(history))
	for head := range 64 {
		weights := make([]float64, len(history))
		maximum := float64(sinks[head])
		for index, row := range history {
			var dot float64
			for i, value := range row {
				dot += float64(rawCapacityRound(query[head*512+i], 10, -14)) * float64(value)
			}
			weights[index] = dot / math.Sqrt(512)
			scores[head*len(history)+index] = float32(weights[index])
			maximum = math.Max(maximum, weights[index])
		}
		denominator := math.Exp(float64(sinks[head]) - maximum)
		for i := range weights {
			weights[i] = math.Exp(weights[i] - maximum)
			denominator += weights[i]
		}
		for i := range 512 {
			var sum float64
			for index, row := range history {
				sum += weights[index] * float64(row[i])
			}
			output[head*512+i] = float32(sum / denominator)
		}
		rawCapacityRotate(output[head*512:(head+1)*512], position, true)
	}
	return
}

func TestRawCapacityReferenceControls(t *testing.T) {
	if rawCapacityRound(1.00048828125, 10, -14) != 1 || rawCapacityFP8(1.0625) != 1 {
		t.Fatal("ties do not round to even")
	}
	for _, position := range []int{127, 128, 129, 2048} {
		query, current, sinks := rawCapacityInputs(position)
		for head := range 64 {
			rawCapacityRotate(query[head*512:(head+1)*512], position, false)
		}
		rawCapacityRotate(current, position, false)
		history := append(rawCapacityHistory(position, 128), rawCapacityStoredRow(current))
		want, _ := rawCapacityAttention(query, history, sinks, position)
		bad, _ := rawCapacityAttention(query, history[len(history)-1:], sinks, position)
		var maximum float64
		for i := range want {
			maximum = math.Max(maximum, math.Abs(float64(want[i]-bad[i])))
		}
		if maximum < 0.01 {
			t.Fatalf("position %d: omitted-history fixture not discriminating, max=%g", position, maximum)
		}
	}
}
