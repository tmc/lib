package cudarepro

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// Measurement retains every replay and its observed reproduction count.
// It describes an empirical frequency, not a confidence interval.
type Measurement struct {
	Reproductions int
	Runs          []cudaschedule.RunRecord
}

// OneOff reports whether exactly one attempt reproduced the failure.
func (m Measurement) OneOff() bool { return m.Reproductions == 1 && len(m.Runs) > 0 }

// Measure replays a finding exactly repeats times unless cancellation, a
// runner error, or invalid evidence stops it. replay must invoke the existing
// campaign runner in a fresh process for candidates that can fault. Returned
// partial measurements retain completed bounded records, including records
// with invalid classification evidence. Cyclic or excessive comparison graphs
// are rejected before retention.
func Measure(ctx context.Context, identity Identity, repeats int, replay func(context.Context) (cudaschedule.RunRecord, error)) (Measurement, error) {
	var m Measurement
	if ctx == nil || replay == nil || repeats <= 0 || identity.Policy != "exact-uint64" {
		return m, fmt.Errorf("measure reproduction: invalid context, runner, replay count, or identity")
	}
	for n := 0; n < repeats; n++ {
		if err := ctx.Err(); err != nil {
			return m, fmt.Errorf("measure reproduction: %w", err)
		}
		run, err := replay(ctx)
		if err != nil {
			return m, fmt.Errorf("replay %d: %w", n+1, err)
		}
		run, err = snapshot(run)
		if err != nil {
			return m, fmt.Errorf("replay %d: %w", n+1, err)
		}
		m.Runs = append(m.Runs, run)
		matches, err := identity.Matches(run)
		if err != nil {
			return m, fmt.Errorf("replay %d: %w", n+1, err)
		}
		if matches {
			m.Reproductions++
		}
	}
	return m, nil
}

// snapshot prevents a runner that reuses comparison receipts from changing
// earlier attempts. Nested comparison records are bounded before classification.
func snapshot(run cudaschedule.RunRecord) (cudaschedule.RunRecord, error) {
	remaining := 128
	active := make(map[*cudaschedule.RunRecord]bool)
	var copyRun func(*cudaschedule.RunRecord, int) (*cudaschedule.RunRecord, error)
	copyRun = func(src *cudaschedule.RunRecord, depth int) (*cudaschedule.RunRecord, error) {
		if src == nil {
			return nil, nil
		}
		if active[src] || depth > 16 || remaining == 0 {
			return nil, fmt.Errorf("snapshot execution: cyclic or excessive comparison records")
		}
		remaining--
		active[src] = true
		defer delete(active, src)
		dst := *src
		if src.Mismatch != nil {
			mismatch := *src.Mismatch
			dst.Mismatch = &mismatch
		}
		var err error
		dst.Pristine, err = copyRun(src.Pristine, depth+1)
		if err != nil {
			return nil, err
		}
		dst.Baseline, err = copyRun(src.Baseline, depth+1)
		if err != nil {
			return nil, err
		}
		return &dst, nil
	}
	copy, err := copyRun(&run, 0)
	if err != nil {
		return cudaschedule.RunRecord{}, err
	}
	return *copy, nil
}

// AcceptRate reports whether a candidate preserves the baseline empirical
// frequency. Both measurements must contain the same declared number of runs.
// An absent failure cannot establish a baseline. A one-off baseline requires
// allowOneOff. Counts are rechecked against retained execution evidence.
func AcceptRate(identity Identity, baseline, candidate Measurement, allowOneOff bool) (bool, error) {
	for _, m := range []Measurement{baseline, candidate} {
		if len(m.Runs) == 0 {
			return false, fmt.Errorf("compare reproduction rate: missing attempts")
		}
		count := 0
		for _, run := range m.Runs {
			run, err := snapshot(run)
			if err != nil {
				return false, err
			}
			ok, err := identity.Matches(run)
			if err != nil {
				return false, fmt.Errorf("compare reproduction rate: %w", err)
			}
			if ok {
				count++
			}
		}
		if count != m.Reproductions {
			return false, fmt.Errorf("compare reproduction rate: inconsistent reproduction count")
		}
	}
	if len(baseline.Runs) != len(candidate.Runs) {
		return false, fmt.Errorf("compare reproduction rate: different replay counts")
	}
	if baseline.Reproductions == 0 {
		return false, fmt.Errorf("compare reproduction rate: baseline did not reproduce")
	}
	return (!baseline.OneOff() || allowOneOff) && candidate.Reproductions >= baseline.Reproductions, nil
}
