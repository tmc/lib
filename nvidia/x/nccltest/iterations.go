package nccltest

import (
	"errors"
	"fmt"
)

func iterationCounts(iterations, aggregation int) (int, int, error) {
	if aggregation == 0 {
		aggregation = 1
	}
	if iterations <= 0 || aggregation < 1 || iterations > int(^uint(0)>>1)/aggregation {
		return 0, 0, fmt.Errorf("nccltest: invalid or overflowing iteration counts")
	}
	return aggregation, iterations * aggregation, nil
}

// Upstream groups aggregated calls once per outer timed iteration. Warmup and
// correctness passes are separate and do not use this aggregation count.
func timedIterations(iterations, aggregation int, submit, start, end func() error) error {
	return timedIterationsSync(iterations, aggregation, submit, start, end, nil)
}

func timedIterationsSync(iterations, aggregation int, submit, start, end, synchronize func() error) error {
	for i := 0; i < iterations; i++ {
		if aggregation > 1 {
			if err := start(); err != nil {
				return err
			}
		}
		var err error
		for j := 0; j < aggregation; j++ {
			if err = submit(); err != nil {
				break
			}
		}
		if aggregation > 1 {
			err = errors.Join(err, end())
		}
		if err != nil {
			return err
		}
		if synchronize != nil {
			if err := synchronize(); err != nil {
				return err
			}
		}
	}
	return nil
}
