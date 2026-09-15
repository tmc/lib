package cuptibench

import (
	"errors"

	"github.com/tmc/lib/nvidia/internal/cuptiowner"
)

// Timing must also exclude cooperating collectors: an uninstrumented timer
// alone does not mean another collector has stopped instrumenting the device.
type ownedTimer struct {
	benchmarkTimer
	lease *cuptiowner.Lease
}

func newOwnedTimer(device uint32, create func() (benchmarkTimer, error)) (benchmarkTimer, error) {
	lease, err := cuptiowner.AcquireDevice(device)
	if err != nil {
		return nil, err
	}
	timer, err := create()
	if timer == nil {
		lease.Release()
		if err == nil {
			err = errors.New("cuptibench: timer construction returned no timer")
		}
		return nil, err
	}
	return &ownedTimer{benchmarkTimer: timer, lease: lease}, err
}

func (t *ownedTimer) Close() error {
	if err := t.benchmarkTimer.Close(); err != nil {
		return err
	}
	t.lease.Release()
	return nil
}
