package cudarepro

import (
	"context"
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// Case supplies explicit workload inputs and launch dimensions. It is not an
// interpretation of a campaign's free-form input or geometry labels. Callers
// importing campaign records must obtain these values from the workload's
// recorded input contract. This capability supports uint64 inputs only.
// Schedule, when present, binds injection decisions to pristine Source.
type Case struct {
	Source    string
	Inputs    []uint64
	Shape     []uint64
	Grid      [3]uint32
	Block     [3]uint32
	SharedMem uint32
	Schedule  *Schedule
}

// ReduceOptions bounds reduction work. Neither field has a default.
// At most Repeats*(MaxCandidates+1) runner calls are made. AllowOneOff permits
// reducing a baseline that reproduced exactly once.
type ReduceOptions struct {
	Repeats       int
	MaxCandidates int
	AllowOneOff   bool
}

// ReductionAttempt retains the candidate, every executed run, and acceptance.
type ReductionAttempt struct {
	Dimension   string
	Case        Case
	Measurement Measurement
	Accepted    bool
}

// Reduction retains the fresh baseline and every candidate attempt. Original
// and Reduced expose the sizes of all supported dimensions. NoReduction gives
// the reason when no candidate was accepted; BudgetExhausted distinguishes
// bounded stopping from exhausting all proposed reductions.
type Reduction struct {
	Original        Case
	Reduced         Case
	Baseline        Measurement
	Attempts        []ReductionAttempt
	IdentityDrift   int
	CandidateRuns   int
	NoReduction     string
	BudgetExhausted bool
}

func cloneCase(c Case) Case {
	c.Inputs = slices.Clone(c.Inputs)
	c.Shape = slices.Clone(c.Shape)
	c.Schedule = cloneSchedule(c.Schedule)
	return c
}

// Reduce shrinks explicit uint64 inputs, tensor extents and launch dimensions
// in the documented order, preserving source text and the original failure's
// identity and empirical reproduction rate. replay adapts the existing
// campaign runner; it must use fresh processes when candidates can fault.
// Host fixtures may supply simulated run records, which are not native proof.
// Partial results are returned on cancellation or invalid execution evidence.
func Reduce(ctx context.Context, original Case, finding cudaschedule.RunRecord, opts ReduceOptions, replay func(context.Context, Case) (cudaschedule.RunRecord, error)) (Reduction, error) {
	r := Reduction{Original: cloneCase(original), Reduced: cloneCase(original)}
	if ctx == nil || replay == nil || original.Source == "" || opts.Repeats <= 0 || opts.MaxCandidates <= 0 {
		return r, fmt.Errorf("reduce finding: missing source, context, runner, or explicit run budget")
	}
	if opts.MaxCandidates > int(^uint(0)>>1)/opts.Repeats-1 {
		return r, fmt.Errorf("reduce finding: run budget overflows")
	}
	for _, extent := range original.Shape {
		if extent == 0 {
			return r, fmt.Errorf("reduce finding: zero shape extent")
		}
	}
	for n := range original.Grid {
		if original.Grid[n] == 0 || original.Block[n] == 0 {
			return r, fmt.Errorf("reduce finding: zero launch extent")
		}
	}
	if original.Schedule != nil {
		if err := checkScheduleRun(original, finding); err != nil {
			return r, err
		}
	}
	finding, err := snapshot(finding)
	if err != nil {
		return r, err
	}
	identity, err := Identify(finding)
	if err != nil {
		return r, err
	}
	measure := func(c Case) (Measurement, error) {
		return Measure(ctx, identity, opts.Repeats, func(ctx context.Context) (cudaschedule.RunRecord, error) {
			run, err := replay(ctx, cloneCase(c))
			if err == nil && c.Schedule != nil {
				err = checkScheduleRun(c, run)
			}
			return run, err
		})
	}
	r.Baseline, err = measure(original)
	if err != nil {
		return r, err
	}
	if r.Baseline.Reproductions == 0 {
		return r, fmt.Errorf("reduce finding: fresh baseline did not reproduce")
	}
	if r.Baseline.OneOff() && !opts.AllowOneOff {
		r.NoReduction = "one-off baseline requires explicit opt-in"
		return r, nil
	}
	try := func(dimension string, c Case) (bool, error) {
		if len(r.Attempts) == opts.MaxCandidates {
			r.BudgetExhausted = true
			return false, nil
		}
		m, err := measure(c)
		r.CandidateRuns += len(m.Runs)
		a := ReductionAttempt{Dimension: dimension, Case: cloneCase(c), Measurement: m}
		for _, run := range m.Runs {
			v := cudaschedule.Classify(run)
			ok, matchErr := identity.Matches(run)
			if !ok && (matchErr != nil || v != cudaschedule.VerdictPass) {
				r.IdentityDrift++
			}
		}
		if err == nil {
			a.Accepted, err = AcceptRate(identity, r.Baseline, m, opts.AllowOneOff)
		}
		r.Attempts = append(r.Attempts, a)
		if a.Accepted {
			r.Reduced = cloneCase(c)
		}
		return a.Accepted, err
	}
	// Delete successively smaller chunks; repeat a successful deletion at the
	// same position before advancing. Every accepted edit strictly reduces size.
	for width := len(r.Reduced.Inputs); width > 0 && !r.BudgetExhausted; width /= 2 {
		for start := 0; start+width <= len(r.Reduced.Inputs) && !r.BudgetExhausted; {
			c := cloneCase(r.Reduced)
			c.Inputs = append(c.Inputs[:start], c.Inputs[start+width:]...)
			accepted, err := try("input lengths", c)
			if err != nil {
				return r, err
			}
			if !accepted {
				start += width
			}
		}
	}
	for n := range r.Reduced.Inputs {
		for r.Reduced.Inputs[n] > 0 && !r.BudgetExhausted {
			c := cloneCase(r.Reduced)
			c.Inputs[n] /= 2
			accepted, err := try("input values", c)
			if err != nil {
				return r, err
			}
			if !accepted {
				break
			}
		}
	}
	for n := range r.Reduced.Shape {
		for r.Reduced.Shape[n] > 1 && !r.BudgetExhausted {
			c := cloneCase(r.Reduced)
			// Avoid overflow for the largest uint64 extent.
			c.Shape[n] = r.Reduced.Shape[n]/2 + r.Reduced.Shape[n]%2
			accepted, err := try("shape", c)
			if err != nil {
				return r, err
			}
			if !accepted {
				break
			}
		}
	}
	for field := 0; field < 7 && !r.BudgetExhausted; field++ {
		for !r.BudgetExhausted {
			c := cloneCase(r.Reduced)
			var value *uint32
			if field < 3 {
				value = &c.Grid[field]
			} else if field < 6 {
				value = &c.Block[field-3]
			} else {
				value = &c.SharedMem
			}
			minimum := uint32(1)
			if field == 6 {
				minimum = 0
			}
			if *value <= minimum {
				break
			}
			*value /= 2
			if *value < minimum {
				*value = minimum
			}
			accepted, err := try("launch", c)
			if err != nil {
				return r, err
			}
			if !accepted {
				break
			}
		}
	}
	if r.Reduced.Schedule != nil {
		for width := len(r.Reduced.Schedule.Decisions); width > 0 && !r.BudgetExhausted; width /= 2 {
			for start := 0; start+width <= len(r.Reduced.Schedule.Decisions) && !r.BudgetExhausted; {
				c := cloneCase(r.Reduced)
				c.Schedule.Decisions = append(c.Schedule.Decisions[:start], c.Schedule.Decisions[start+width:]...)
				accepted, err := try("injected sites", c)
				if err != nil {
					return r, err
				}
				if !accepted {
					start += width
				}
			}
		}
	}
	changed := false
	for _, a := range r.Attempts {
		changed = changed || a.Accepted
	}
	if !changed {
		r.NoReduction = "every proposed reduction lost failure identity or reproduction rate"
		if r.BudgetExhausted {
			r.NoReduction = "candidate budget exhausted without an accepted reduction"
		}
	}
	return r, nil
}
