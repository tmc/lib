package cuptibench

import (
	"fmt"
	"strings"
)

// TimeSet names the counters that bound a range in time. It exists to make the
// idle term in a time budget visible: a range whose SMs are active for a small
// fraction of its duration is waiting on something outside the device, and no
// per-kernel counter says so.
//
// What each member measures, established by direct measurement on GB10
// (sm_121, CUDA 13.4, 2026-09-17) rather than from its name:
//
//   - gpu__time_duration.sum is the duration the range was open. A range
//     holding a 50ms host sleep and no kernel at all measures 51.60ms. It is
//     NOT device-busy time.
//   - sm__cycles_active.sum and .avg are genuine busy counters: the same
//     kernel-free range measures exactly 0, and a range holding one 20ms
//     busy-wait kernel measures 51,608,092 gpc_cycles summed over SMs.
//   - sm__cycles_elapsed.avg tracks the range duration in cycles (126,750,196
//     for the 50ms sleep), so it is the denominator ActiveFraction needs.
//
// The pairing matters more than any single value. See ActiveFraction.
const TimeSet = "time"

// timeBudgetMetrics are the members of TimeSet, in the order Budget reads them.
var timeBudgetMetrics = struct{ duration, activeAvg, elapsedAvg, activeSum string }{
	duration:   "gpu__time_duration.sum",
	activeAvg:  "sm__cycles_active.avg",
	elapsedAvg: "sm__cycles_elapsed.avg",
	activeSum:  "sm__cycles_active.sum",
}

// Budget reports how much of a range's duration had SM work in it. Every field
// but ActiveFraction is a counter read; ActiveFraction is the one derived
// quantity and is named so that it cannot be mistaken for a duration.
type Budget struct {
	// DurationNanos is the wall time the range was open, from
	// gpu__time_duration.sum. It includes host gaps inside the range.
	DurationNanos float64
	// ActiveCyclesAvg is SM-active cycles averaged over SMs.
	ActiveCyclesAvg float64
	// ElapsedCyclesAvg is the range duration in the same cycle domain.
	ElapsedCyclesAvg float64
	// ActiveCyclesSum is SM-active cycles summed over SMs. Zero proves no
	// kernel executed in the range.
	ActiveCyclesSum float64
	// ActiveFraction is ActiveCyclesAvg/ElapsedCyclesAvg: the occupancy-weighted
	// share of the range's duration that had SM work in it. It is NOT
	// "busy time / span": an SM-hour spread thinly over many SMs and one
	// saturating a single SM are different programs with the same fraction.
	ActiveFraction float64
	// Partial records which members were absent, so a caller can tell a zero
	// from an unmeasured field. A nonempty Partial means ActiveFraction is not
	// computed and stays zero.
	Partial []string
}

// idleIsNotDerivable states, in the one place a reader would look for it, why
// this type reports no idle duration.
//
// The obvious decomposition is idle = duration - busy. It cannot be computed
// from these counters. gpu__time_duration and gpu__time_active both measure the
// range's open interval, so subtracting either from the other yields ~0
// whatever the program does, and the SM cycle counters measure SM-seconds, not
// wall seconds: a kernel occupying one SM of 48 for the whole range is fully
// busy in wall terms and 1/48 busy in SM terms. Closing the budget in wall time
// needs per-kernel start/end timestamps, which is the activity-trace tier, not
// the counter tier.
const idleIsNotDerivable = "idle duration: not derivable from counters; " +
	"gpu__time_duration measures the range's open interval and the SM counters measure SM-seconds, not wall seconds"

// Budget extracts the time budget from a range, which must come from a Whole
// collection. Whole opens exactly one range, so each counter describes that one
// interval and nothing is summed across ranges; the additivity rule that
// timeDimension enforces is about totalling durations over sibling ranges and
// does not bear on this use. In Regions or Kernels these same counters are
// per-range durations that must not be added together, so Budget refuses them
// rather than inviting the sum.
func (r *Result) Budget() (*Budget, error) {
	if r == nil {
		return nil, fmt.Errorf("cuptibench: nil result")
	}
	if mode := r.Metadata["mode"]; mode != "whole" {
		return nil, fmt.Errorf("cuptibench: time budget needs a whole-mode collection, have %q: "+
			"per-range durations must not be summed across ranges", mode)
	}
	if len(r.Ranges) != 1 {
		return nil, fmt.Errorf("cuptibench: time budget needs exactly one range, have %d", len(r.Ranges))
	}
	have := make(map[string]float64)
	for _, m := range r.Ranges[0].Metrics {
		have[m.Name] = m.Value
	}
	b := new(Budget)
	for _, f := range []struct {
		name string
		dst  *float64
	}{
		{timeBudgetMetrics.duration, &b.DurationNanos},
		{timeBudgetMetrics.activeAvg, &b.ActiveCyclesAvg},
		{timeBudgetMetrics.elapsedAvg, &b.ElapsedCyclesAvg},
		{timeBudgetMetrics.activeSum, &b.ActiveCyclesSum},
	} {
		v, ok := have[f.name]
		if !ok {
			b.Partial = append(b.Partial, f.name)
			continue
		}
		*f.dst = v
	}
	if len(b.Partial) == 0 && b.ElapsedCyclesAvg > 0 {
		b.ActiveFraction = b.ActiveCyclesAvg / b.ElapsedCyclesAvg
	}
	return b, nil
}

// String renders the budget with every denominator named, and states the term
// it cannot supply rather than leaving its absence to be noticed.
func (b *Budget) String() string {
	if b == nil {
		return "<nil budget>"
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "range duration %.3fms (open interval, host gaps included)",
		b.DurationNanos/1e6)
	if len(b.Partial) > 0 {
		fmt.Fprintf(&sb, "; active fraction UNMEASURED (absent: %s)", strings.Join(b.Partial, ", "))
	} else {
		fmt.Fprintf(&sb, "; SM-active %.2f%% of that interval (occupancy-weighted, avg over SMs)",
			100*b.ActiveFraction)
		if b.ActiveCyclesSum == 0 {
			sb.WriteString("; sm__cycles_active.sum is 0, so no kernel executed in this range")
		}
	}
	sb.WriteString("; ")
	sb.WriteString(idleIsNotDerivable)
	return sb.String()
}
