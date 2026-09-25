//go:build linux

package cuptibench_test

import (
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// budgetWorkload runs one range that optionally sleeps on the host and
// optionally launches the 20ms device busy-wait kernel. The four combinations
// separate range duration from device-busy time, which is the distinction
// TimeSet exists to make.
func budgetWorkload(p *physicalPipeline, sleep time.Duration, launch bool) cuptibench.Workload {
	w := cuptibench.Workload{
		Streams: p.streams[:], Synchronize: p.synchronize,
		Preparation: "host sleep and/or one 20ms device busy-wait kernel", View: "full",
	}
	w.Run = func(s *cuptibench.Scope) error {
		return s.Region("budget", func() error {
			if sleep > 0 {
				time.Sleep(sleep)
			}
			if !launch {
				return nil
			}
			return cudalaunch.Kernel(p.wait).Grid(1, 1, 1).Block(physicalBlock, 1, 1).
				On(p.streams[0]).Launch(p.out, uint32(0), uint32(7))
		})
	}
	return w
}

// TestPhysicalTimeBudgetSeparatesIdleFromBusy is the positive control the time
// budget rests on. A range holding only a host sleep and a range holding only a
// kernel must not produce the same numbers. Specifically:
//
//   - sm__cycles_active.sum must be exactly 0 for the sleep-only range, which
//     is what proves it measures device work rather than elapsed time.
//   - gpu__time_duration.sum must track the sleep, which is what proves it does
//     NOT measure device work and must never be reported as busy.
//
// Both halves are required. A counter that moved in both cases, or neither,
// would carry no information about idleness.
func TestPhysicalTimeBudgetSeparatesIdleFromBusy(t *testing.T) {
	p := newPhysicalPipeline(t, 8)
	const sleep = 50 * time.Millisecond

	budget := func(t *testing.T, sleep time.Duration, launch bool) *cuptibench.Budget {
		t.Helper()
		names, err := cuptibench.Resolve(cuptibench.TimeSet)
		physicalCheck(t, err)
		plan, err := cuptibench.New(p.dev, cuptibench.Config{
			CachePolicy: cuptirange.CachePreserve, Metrics: names,
			Mode: cuptibench.Whole, Operations: 1, MaxConfigBytes: 16 << 20,
			Limits: cuptirange.Limits{MaxRanges: 128, MaxRangeTreeNodes: 256,
				MaxCounterBytes: 32 << 20, MaxPasses: 128},
		})
		if plan != nil {
			physicalClose(t, plan.Close)
		}
		physicalCheck(t, err)
		res, err := plan.Collect(budgetWorkload(p, sleep, launch))
		physicalCheck(t, err)
		b, err := res.Budget()
		physicalCheck(t, err)
		t.Logf("sleep=%v kernel=%v -> %s", sleep, launch, b)
		return b
	}

	idle := budget(t, sleep, false)
	busy := budget(t, 0, true)

	if idle.ActiveCyclesSum != 0 {
		t.Errorf("sleep-only range: sm__cycles_active.sum = %v, want exactly 0; "+
			"a nonzero value means this counter does not measure device work",
			idle.ActiveCyclesSum)
	}
	if busy.ActiveCyclesSum <= 0 {
		t.Errorf("kernel-only range: sm__cycles_active.sum = %v, want > 0", busy.ActiveCyclesSum)
	}
	// The duration counter must do the opposite: track the host sleep despite no
	// kernel running. This is the measurement that rules out idle = duration - busy.
	if got := time.Duration(idle.DurationNanos); got < sleep {
		t.Errorf("sleep-only range: duration %v < the %v host sleep; "+
			"gpu__time_duration is expected to include host gaps", got, sleep)
	}
	if idle.ActiveFraction >= busy.ActiveFraction {
		t.Errorf("active fraction failed to separate idle (%v) from busy (%v)",
			idle.ActiveFraction, busy.ActiveFraction)
	}
}

// TestPhysicalBudgetRefusesNonWholeModes confirms on device that the mode
// restriction holds where the counters are real.
func TestPhysicalBudgetRefusesNonWholeModes(t *testing.T) {
	p := newPhysicalPipeline(t, 8)
	names, err := cuptibench.Resolve(cuptibench.TimeSet)
	physicalCheck(t, err)
	plan, err := cuptibench.New(p.dev, cuptibench.Config{
		CachePolicy: cuptirange.CachePreserve, Metrics: names,
		Mode: cuptibench.Regions, Operations: 1, MaxDepth: 4, MaxConfigBytes: 16 << 20,
		Limits: cuptirange.Limits{MaxRanges: 128, MaxRangeTreeNodes: 256,
			MaxCounterBytes: 32 << 20, MaxPasses: 128},
	})
	if plan != nil {
		physicalClose(t, plan.Close)
	}
	physicalCheck(t, err)
	res, err := plan.Collect(budgetWorkload(p, 0, true))
	physicalCheck(t, err)
	if _, err := res.Budget(); err == nil {
		t.Fatal("Budget accepted a regions-mode collection; per-range durations must not be summed")
	} else {
		t.Logf("regions mode correctly refused: %v", err)
	}
}

// TestPhysicalWarmCacheTrafficGuard confirms on device that the guard refuses a
// traffic metric under a no-reset policy and admits it once acknowledged.
func TestPhysicalWarmCacheTrafficGuard(t *testing.T) {
	p := newPhysicalPipeline(t, 8)
	config := func(allow bool) cuptibench.Config {
		return cuptibench.Config{
			CachePolicy: cuptirange.CachePreserve, AllowWarmCacheTraffic: allow,
			Metrics: []string{physicalTrafficRead}, Mode: cuptibench.Whole,
			Operations: 1, MaxConfigBytes: 16 << 20,
			Limits: cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 16,
				MaxCounterBytes: 32 << 20, MaxPasses: 64},
		}
	}
	plan, err := cuptibench.New(p.dev, config(false))
	if plan != nil {
		physicalClose(t, plan.Close)
	}
	if err == nil {
		t.Fatal("New admitted a traffic counter under an uncontrolled cache")
	}
	t.Logf("unacknowledged traffic correctly refused: %v", err)

	allowed, err := cuptibench.New(p.dev, config(true))
	if allowed != nil {
		physicalClose(t, allowed.Close)
	}
	physicalCheck(t, err)
}
