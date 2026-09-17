package cudatrial

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// TimingKind identifies the specific measurement mechanism used for a receipt duration.
type TimingKind string

const (
	TimingUnspecified  TimingKind = ""
	TimingWallElapsed  TimingKind = "wall_elapsed"
	TimingSummedKernel TimingKind = "summed_kernel"
	TimingGPUBusy      TimingKind = "gpu_busy"
	TimingAPICall      TimingKind = "api_call"
)

// An Arm represents one experimental variant to be evaluated in a trial.
type Arm struct {
	Name string
	Run  func(context.Context) (Receipt, error)
}

// A Receipt records the execution outcome, output digest, duration, and provenance
// metadata of a single arm execution within a trial round.
type Receipt struct {
	Name           string
	Round          int
	Complete       bool
	Digest         string
	Started        time.Time
	Finished       time.Time
	DeviceDuration time.Duration // Deprecated: use TimingKind and specific duration fields
	TimingKind     TimingKind
	WallElapsed    time.Duration
	SummedKernel   time.Duration
	GPUBusy        time.Duration
	APICall        time.Duration
	Timings        map[TimingKind]time.Duration // Explicitly measured durations, distinguishing measured 0 from unmeasured
	Provenance     map[string]string
	Err            error
}

// RecordTiming records a measured duration for the specified timing kind.
// It returns an error if kind is empty or duration is negative.
func (r *Receipt) RecordTiming(kind TimingKind, d time.Duration) error {
	if kind == "" {
		return errors.New("cudatrial: timing kind cannot be empty")
	}
	if d < 0 {
		return fmt.Errorf("cudatrial: duration cannot be negative: %v", d)
	}
	if r.Timings == nil {
		r.Timings = make(map[TimingKind]time.Duration)
	}
	r.Timings[kind] = d
	switch kind {
	case TimingWallElapsed:
		r.WallElapsed = d
	case TimingSummedKernel:
		r.SummedKernel = d
	case TimingGPUBusy:
		r.GPUBusy = d
	case TimingAPICall:
		r.APICall = d
	}
	if r.TimingKind == "" {
		r.TimingKind = kind
	}
	return nil
}

// HasTiming reports whether the specified timing kind was explicitly measured on this receipt.
func (r Receipt) HasTiming(kind TimingKind) bool {
	if kind == "" {
		return false
	}
	if r.Timings != nil {
		_, ok := r.Timings[kind]
		return ok
	}
	switch kind {
	case TimingWallElapsed:
		return r.WallElapsed > 0 || (!r.Started.IsZero() && !r.Finished.IsZero())
	case TimingSummedKernel:
		return r.SummedKernel > 0
	case TimingGPUBusy:
		return r.GPUBusy > 0
	case TimingAPICall:
		return r.APICall > 0
	default:
		return false
	}
}

// DurationFor returns the duration for the specified timing kind.
// If the timing kind is empty or was not measured on this receipt, it returns an error.
func (r Receipt) DurationFor(kind TimingKind) (time.Duration, error) {
	if kind == "" {
		return 0, errors.New("cudatrial: timing kind cannot be empty")
	}
	if r.Timings != nil {
		if d, ok := r.Timings[kind]; ok {
			return d, nil
		}
	}
	switch kind {
	case TimingWallElapsed:
		if r.WallElapsed != 0 {
			return r.WallElapsed, nil
		}
		if !r.Started.IsZero() && !r.Finished.IsZero() {
			return r.Finished.Sub(r.Started), nil
		}
	case TimingSummedKernel:
		if r.SummedKernel != 0 {
			return r.SummedKernel, nil
		}
	case TimingGPUBusy:
		if r.GPUBusy != 0 {
			return r.GPUBusy, nil
		}
	case TimingAPICall:
		if r.APICall != 0 {
			return r.APICall, nil
		}
	}
	return 0, fmt.Errorf("cudatrial: timing kind %q not available on receipt %q", kind, r.Name)
}

// Duration returns the measured execution duration.
// If TimingKind is set and available, that duration is returned.
// Otherwise, for backwards compatibility, if DeviceDuration > 0 it is returned;
// otherwise wall time (Finished - Started or WallElapsed) is returned.
func (r Receipt) Duration() time.Duration {
	if r.TimingKind != "" {
		if d, err := r.DurationFor(r.TimingKind); err == nil {
			return d
		}
	}
	if r.DeviceDuration > 0 {
		return r.DeviceDuration
	}
	if r.WallElapsed > 0 {
		return r.WallElapsed
	}
	if r.Started.IsZero() || r.Finished.IsZero() {
		return 0
	}
	return r.Finished.Sub(r.Started)
}

// Options configures an interleaved trial run.
type Options struct {
	// ContinueOnError controls whether Interleave continues evaluating subsequent
	// arms and rounds when an arm's Run returns an error.
	// When true, the error is recorded on the Receipt and execution proceeds.
	// When false (default), Interleave halts immediately upon the first error.
	ContinueOnError bool
}

// Interleave executes arms across the specified number of rounds using ABBA
// alternating order (e.g. A,B then B,A) to cancel monotonic background drift.
//
// If rounds <= 0, or if arms is empty, Interleave returns an error. If ctx is
// canceled or times out, Interleave stops and returns ctx.Err().
func Interleave(ctx context.Context, arms []Arm, rounds int) ([]Receipt, error) {
	return InterleaveWithOptions(ctx, arms, rounds, Options{})
}

// InterleaveWithOptions executes arms with custom Options.
func InterleaveWithOptions(ctx context.Context, arms []Arm, rounds int, opts Options) ([]Receipt, error) {
	if rounds <= 0 {
		return nil, fmt.Errorf("cudatrial: rounds must be positive, got %d", rounds)
	}
	if len(arms) == 0 {
		return nil, errors.New("cudatrial: arms must not be empty")
	}
	for i, arm := range arms {
		if arm.Name == "" {
			return nil, fmt.Errorf("cudatrial: arm %d has empty Name", i)
		}
		if arm.Run == nil {
			return nil, fmt.Errorf("cudatrial: arm %q has nil Run function", arm.Name)
		}
	}

	total := len(arms) * rounds
	receipts := make([]Receipt, 0, total)

	for round := 1; round <= rounds; round++ {
		roundArms := arms
		if round%2 == 0 && len(arms) > 1 {
			// ABBA ordering: reverse execution order on alternating rounds to cancel monotonic drift.
			rev := make([]Arm, len(arms))
			for i, a := range arms {
				rev[len(arms)-1-i] = a
			}
			roundArms = rev
		}

		for _, arm := range roundArms {
			if err := ctx.Err(); err != nil {
				return receipts, fmt.Errorf("cudatrial: %w", err)
			}

			start := time.Now()
			receipt, err := arm.Run(ctx)
			finish := time.Now()

			// Stamp standard fields if not already populated.
			receipt.Name = arm.Name
			receipt.Round = round
			if receipt.Started.IsZero() {
				receipt.Started = start
			}
			if receipt.Finished.IsZero() {
				receipt.Finished = finish
			}
			wall := receipt.Finished.Sub(receipt.Started)
			if !receipt.HasTiming(TimingWallElapsed) {
				_ = receipt.RecordTiming(TimingWallElapsed, wall)
			}
			if receipt.TimingKind == "" {
				if receipt.GPUBusy > 0 {
					receipt.TimingKind = TimingGPUBusy
				} else if receipt.SummedKernel > 0 {
					receipt.TimingKind = TimingSummedKernel
				} else if receipt.DeviceDuration > 0 {
					receipt.TimingKind = TimingSummedKernel
				} else {
					receipt.TimingKind = TimingWallElapsed
				}
			}
			if err != nil {
				receipt.Err = err
				receipt.Complete = false
				receipts = append(receipts, receipt)
				if !opts.ContinueOnError {
					return receipts, fmt.Errorf("cudatrial: arm %q round %d: %w", arm.Name, round, err)
				}
				continue
			}

			receipts = append(receipts, receipt)
		}
	}

	return receipts, nil
}

// ByArm groups receipts by arm name, preserving round order.
func ByArm(receipts []Receipt) map[string][]Receipt {
	out := make(map[string][]Receipt)
	for _, r := range receipts {
		out[r.Name] = append(out[r.Name], r)
	}
	return out
}

// SameArmExact verifies that every completed receipt for each arm produced
// an identical output digest across all rounds, and that every arm has at least
// one completed run.
//
// If any arm produced divergent digests, or if an arm has no completed runs,
// SameArmExact returns false and an explanatory description.
func SameArmExact(receipts []Receipt) (bool, string) {
	if len(receipts) == 0 {
		return false, "cudatrial: no receipts provided"
	}
	digests := make(map[string]string)
	completeCounts := make(map[string]int)

	for _, r := range receipts {
		if !r.Complete {
			continue
		}
		completeCounts[r.Name]++
		prev, ok := digests[r.Name]
		if !ok {
			digests[r.Name] = r.Digest
			continue
		}
		if prev != r.Digest {
			return false, fmt.Sprintf("arm %q produced divergent digests: round %d had %q, earlier had %q", r.Name, r.Round, r.Digest, prev)
		}
	}

	byArm := ByArm(receipts)
	for name := range byArm {
		if completeCounts[name] == 0 {
			return false, fmt.Sprintf("arm %q has no completed runs", name)
		}
	}
	return true, ""
}

// DistinctDigests returns a sorted list of unique digests observed across all receipts.
func DistinctDigests(receipts []Receipt) []string {
	seen := make(map[string]bool)
	for _, r := range receipts {
		if r.Digest != "" {
			seen[r.Digest] = true
		}
	}
	out := make([]string, 0, len(seen))
	for d := range seen {
		out = append(out, d)
	}
	sort.Strings(out)
	return out
}

// FormatSummary formats a multi-line tabular summary of the trial receipts,
// showing arm names, round count, completion status, mean duration, median duration,
// and digests.
func FormatSummary(receipts []Receipt) string {
	if len(receipts) == 0 {
		return "no receipts\n"
	}
	byArm := ByArm(receipts)
	names := make([]string, 0, len(byArm))
	for name := range byArm {
		names = append(names, name)
	}
	sort.Strings(names)

	var sb strings.Builder
	sb.WriteString("ARM                  ROUNDS  COMPLETE  CLOCK  MEAN_DUR      MEDIAN_DUR    DIGESTS\n")
	for _, name := range names {
		rs := byArm[name]
		completeCount := 0
		var totalDur time.Duration
		var durs []time.Duration
		digestSet := make(map[string]bool)
		clocks := make(map[string]bool)
		for _, r := range rs {
			if r.Complete {
				completeCount++
			}
			switch r.TimingKind {
			case TimingGPUBusy, TimingSummedKernel:
				clocks["dev"] = true
			case TimingAPICall:
				clocks["api"] = true
			case TimingWallElapsed:
				clocks["wall"] = true
			default:
				if r.DeviceDuration > 0 {
					clocks["dev"] = true
				} else {
					clocks["wall"] = true
				}
			}
			d := r.Duration()
			totalDur += d
			durs = append(durs, d)
			if r.Digest != "" {
				digestSet[r.Digest] = true
			}
		}
		clock := "wall"
		if len(clocks) > 1 {
			clock = "mix"
		} else {
			for c := range clocks {
				clock = c
			}
		}

		var meanDur time.Duration
		var medianDur time.Duration
		if len(durs) > 0 {
			meanDur = totalDur / time.Duration(len(durs))
			sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
			medianDur = durs[len(durs)/2]
		}

		dList := make([]string, 0, len(digestSet))
		for d := range digestSet {
			dList = append(dList, d)
		}
		sort.Strings(dList)
		digestsStr := strings.Join(dList, ",")
		if len(digestsStr) > 24 {
			digestsStr = digestsStr[:21] + "..."
		}

		fmt.Fprintf(&sb, "%-20s %-7d %-9s %-6s %-13s %-13s %s\n",
			name,
			len(rs),
			fmt.Sprintf("%d/%d", completeCount, len(rs)),
			clock,
			meanDur.Round(time.Microsecond),
			medianDur.Round(time.Microsecond),
			digestsStr,
		)
	}
	return sb.String()
}
