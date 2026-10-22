package main

import (
	"math"
	"testing"
)

func mixedCapacityTheta(position, r int) float32 {
	corr := func(rotations float64) float64 {
		return 64 * math.Log(65536/(rotations*2*math.Pi)) / (2 * math.Log(160000))
	}
	low, high := math.Max(0, math.Floor(corr(32))), math.Min(63, math.Ceil(corr(1)))
	ramp := 1 - math.Max(0, math.Min(1, (float64(r/2)-low)/math.Max(.001, high-low)))
	return float32(float64(position) * math.Pow(160000, -float64(r)/64) * (.0625*(1-ramp) + ramp))
}

func mixedCapacityRotate(values []float32, position int, inverse bool) {
	for r := 0; r < 64; r += 2 {
		s, c := math.Sincos(float64(mixedCapacityTheta(position, r)))
		if inverse {
			s = -s
		}
		i := 448 + r
		a, b := float64(values[i]), float64(values[i+1])
		values[i], values[i+1] = float32(a*c-b*s), float32(a*s+b*c)
	}
}

func mixedCapacityInputs(position int) (query, current, sinks []float32) {
	query, current, sinks = rawCapacityInputs(0)
	for head := range 64 {
		mixedCapacityRotate(query[head*512:(head+1)*512], position, true)
	}
	mixedCapacityRotate(current, position, true)
	return
}

func mixedCapacityRows(count int) (rows [][]float32, allowed []uint32) {
	rows, allowed = make([][]float32, count), make([]uint32, count)
	for row := range count {
		rows[row] = make([]float32, 512)
		for i := range 512 {
			rows[row][i] = float32((row*13+i*3)%23-11) / 8
		}
		if row%2 == 0 {
			allowed[row] = 1
		}
	}
	if count > 0 {
		// An asymmetric excluded row makes ignoring the mask discriminating.
		for i := range 512 {
			rows[count-1][i] = 4
		}
		if count > 1 {
			allowed[count-1] = 0
		}
	}
	return
}

func mixedCapacityAttention(query []float32, history, compressed [][]float32, allowed []uint32, sinks []float32, position int) (output, scores []float32) {
	output, scores, _ = mixedCapacityAttentionStages(query, history, compressed, allowed, sinks, position)
	return
}

func mixedCapacityAttentionStages(query []float32, history, compressed [][]float32, allowed []uint32, sinks []float32, position int) (output, scores, weighted []float32) {
	count := len(history) + len(compressed)
	output, scores = make([]float32, 64*512), make([]float32, 64*count)
	weighted = make([]float32, len(output))
	rows := append(append([][]float32(nil), history...), compressed...)
	for head := range 64 {
		weights := make([]float64, count)
		maximum := float64(sinks[head])
		for index, row := range rows {
			if index >= len(history) && allowed[index-len(history)] == 0 {
				scores[head*count+index] = -1e30
				continue
			}
			var dot float64
			for i, value := range row {
				dot += float64(rawCapacityRound(query[head*512+i], 10, -14)) * float64(value)
			}
			weights[index] = dot / math.Sqrt(512)
			scores[head*count+index] = float32(weights[index])
			maximum = math.Max(maximum, weights[index])
		}
		denominator := math.Exp(float64(sinks[head]) - maximum)
		for index := range weights {
			if index >= len(history) && allowed[index-len(history)] == 0 {
				weights[index] = 0
				continue
			}
			weights[index] = math.Exp(weights[index] - maximum)
			denominator += weights[index]
		}
		for i := range 512 {
			var sum float64
			for index, row := range rows {
				sum += weights[index] * float64(row[i])
			}
			output[head*512+i] = float32(sum / denominator)
		}
		copy(weighted[head*512:(head+1)*512], output[head*512:(head+1)*512])
		mixedCapacityRotate(output[head*512:(head+1)*512], position, true)
	}
	return
}

func TestMixedCapacityReferenceControls(t *testing.T) {
	for _, position := range []int{127, 128, 129, 2048} {
		query, current, sinks := mixedCapacityInputs(position)
		for head := range 64 {
			mixedCapacityRotate(query[head*512:(head+1)*512], position, false)
		}
		mixedCapacityRotate(current, position, false)
		history := append(rawCapacityHistory(position, 128), rawCapacityStoredRow(current))
		for _, count := range []int{512, 513} {
			rows, mask := mixedCapacityRows(count)
			want, _ := mixedCapacityAttention(query, history, rows, mask, sinks, position)
			all := make([]uint32, count)
			collapsed := make([][]float32, count)
			for i := range count {
				all[i], collapsed[i] = 1, rows[0]
			}
			for _, mutation := range []struct {
				rows [][]float32
				mask []uint32
			}{{rows, all}, {collapsed, mask}} {
				bad, _ := mixedCapacityAttention(query, history, mutation.rows, mutation.mask, sinks, position)
				var maximum float64
				for i := range want {
					maximum = math.Max(maximum, math.Abs(float64(want[i]-bad[i])))
				}
				if maximum < .01 {
					t.Fatalf("position=%d count=%d: mutant fixture not discriminating max=%g", position, count, maximum)
				}
			}
		}
	}
}
