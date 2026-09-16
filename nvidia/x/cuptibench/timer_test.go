package cuptibench

import (
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/cuptiowner"
)

type cleanupTimer struct{ failure error }

func (*cleanupTimer) Measure(run func() error) (time.Duration, error) { return time.Second, run() }
func (t *cleanupTimer) Close() error                                  { return t.failure }

func TestTimingCollectorExclusion(t *testing.T) {
	lease, err := cuptiowner.AcquireDevice(7)
	if err != nil {
		t.Fatal(err)
	}
	timer, err := newOwnedTimer(7, func() (benchmarkTimer, error) { t.Fatal("created timer during collection"); return nil, nil })
	lease.Release()
	if err == nil || timer != nil {
		t.Fatal("timing overlapped collector")
	}
	for _, partial := range []bool{false, true} {
		failure := errors.New("timer cleanup failed")
		native := &cleanupTimer{failure: failure}
		timer, err = newOwnedTimer(7, func() (benchmarkTimer, error) {
			if partial {
				return native, errors.New("partial construction")
			}
			return native, nil
		})
		if timer == nil || (err != nil) != partial {
			t.Fatalf("timer=%v err=%v", timer, err)
		}
		if err := timer.Close(); !errors.Is(err, failure) {
			t.Fatal(err)
		}
		if other, err := cuptiowner.AcquireDevice(7); err == nil {
			other.Release()
			t.Fatal("failed timer cleanup released exclusion")
		}
		native.failure = nil
		if err := timer.Close(); err != nil {
			t.Fatal(err)
		}
		other, err := cuptiowner.AcquireDevice(7)
		if err != nil {
			t.Fatal(err)
		}
		other.Release()
	}
}

func TestTimingConstructionFailureReleasesDevice(t *testing.T) {
	timer, err := newOwnedTimer(7, func() (benchmarkTimer, error) { return nil, errors.New("creation failed") })
	if timer != nil || err == nil {
		t.Fatal("creation failure lost")
	}
	lease, err := cuptiowner.AcquireDevice(7)
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()
}
