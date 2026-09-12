package nccltest

import (
	"fmt"
	"math"
)

// Timing selects how CUDA-event durations are combined across local ranks.
// The zero value selects the arithmetic mean.
type Timing string

const (
	TimingMean     Timing = "mean"
	TimingRankZero Timing = "rank0"
	TimingMinimum  Timing = "min"
	TimingMaximum  Timing = "max"
)

func (t Timing) valid() bool {
	switch t {
	case "", TimingMean, TimingRankZero, TimingMinimum, TimingMaximum:
		return true
	}
	return false
}

func rankTime(values []float64, mode Timing) (float64, error) {
	if !mode.valid() || len(values) == 0 {
		return 0, fmt.Errorf("nccltest: invalid rank timing selection")
	}
	result := values[0]
	for i, value := range values {
		if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return 0, fmt.Errorf("nccltest: invalid elapsed time for rank %d", i)
		}
		if i == 0 {
			continue
		}
		switch mode {
		case "", TimingMean:
			result += value
		case TimingMinimum:
			result = math.Min(result, value)
		case TimingMaximum:
			result = math.Max(result, value)
		}
	}
	if mode == "" || mode == TimingMean {
		result /= float64(len(values))
	}
	return result, nil
}
