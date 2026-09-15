package cuptirange

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
)

// replayRuns bounds user replay. Only CUPTI's all-passes flag establishes
// completion; an expected pass count alone is not a completion receipt. run
// must restore the same workload state and synchronize its work on each pass.
func replayRuns(limit uint, start func() error, run func(uint) error, stop func() (cupti.RangeProfilerStopParams, error)) error {
	return replayPrepared(limit, nil, start, run, stop)
}

func replayPrepared(limit uint, prepare func(uint) error, start func() error, run func(uint) error, stop func() (cupti.RangeProfilerStopParams, error)) error {
	if limit == 0 || start == nil || run == nil || stop == nil {
		return errors.New("cuptirange: invalid replay configuration")
	}
	for pass := uint(0); pass < limit; pass++ {
		if prepare != nil {
			if err := prepare(pass); err != nil {
				return fmt.Errorf("cuptirange: prepare pass %d: %w", pass, err)
			}
		}
		if err := start(); err != nil {
			return fmt.Errorf("cuptirange: start pass %d: %w", pass, err)
		}
		state, err := replayPass(pass, run, stop)
		if err != nil {
			return fmt.Errorf("cuptirange: pass %d: %w", pass, err)
		}
		if state.IsAllPassSubmitted == 1 {
			return nil
		}
		if state.IsAllPassSubmitted != 0 {
			return errors.New("cuptirange: invalid all-passes flag")
		}
	}
	return fmt.Errorf("cuptirange: replay incomplete after %d passes", limit)
}

// replayPass balances a successful Start even if the workload panics. A panic
// still propagates to the caller after Stop; it is not a successful capture.
func replayPass(pass uint, run func(uint) error, stop func() (cupti.RangeProfilerStopParams, error)) (state cupti.RangeProfilerStopParams, err error) {
	defer func() {
		var stopErr error
		state, stopErr = stop()
		err = errors.Join(err, stopErr)
	}()
	err = run(pass)
	return state, err
}
