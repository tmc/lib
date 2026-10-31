package cudadiag

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// HostCallEvent represents a host runtime or driver API invocation.
type HostCallEvent struct {
	Span      tracecontract.Interval `json:"span"`
	Name      string                 `json:"name"`
	HasReturn bool                   `json:"has_return"`
}

// PreEntryEvent measures application submission-ready to API entry, separately
// from the API call. Submission identifies the operation at API entry; its TID
// does not identify an earlier unpinned application interval. SourceClass is
// native, synthetic, imported or managed. These declarations do not authenticate
// a producer; the caller must retain and validate the source receipt and bytes.
type PreEntryEvent struct {
	Span        tracecontract.Interval      `json:"span"`
	Submission  cudaworkload.HostSubmission `json:"submission"`
	Complete    bool                        `json:"complete"`
	SourceClass string                      `json:"source_class"`
}

// KernelEvent represents a GPU device kernel execution.
type KernelEvent struct {
	Span      tracecontract.Interval `json:"span"`
	Name      string                 `json:"name"`
	StreamID  uint64                 `json:"stream_id"`
	ContextID uint64                 `json:"context_id"`
	DeviceID  uint32                 `json:"device_id"`
}

// TimeAccounting holds separated time dimensions for an analyzed workload or operation.
type TimeAccounting struct {
	WallInterval         tracecontract.Interval `json:"wall_interval"`
	SummedKernelDuration time.Duration          `json:"summed_kernel_duration"`
	BusyDuration         time.Duration          `json:"busy_duration"`
	HostAPIDuration      time.Duration          `json:"host_api_duration"`
	// PreEntryDuration is nil when application timing is absent, not zero.
	PreEntryDuration  *time.Duration `json:"pre_entry_duration,omitempty"`
	RunnableDuration  time.Duration  `json:"runnable_duration"`
	ScheduledDuration time.Duration  `json:"scheduled_duration"`
	WaitDuration      time.Duration  `json:"wait_duration"`
}

// SumDuration computes the arithmetic sum of positive interval durations.
// Empty, zero-length and inverted intervals contribute zero. It returns an error
// if an interval duration or the total cannot be represented by time.Duration.
func SumDuration(intervals []tracecontract.Interval) (time.Duration, error) {
	var total time.Duration
	for _, iv := range intervals {
		d, err := intervalDuration(iv)
		if err != nil {
			return 0, err
		}
		total, err = addDuration(total, d)
		if err != nil {
			return 0, err
		}
	}
	return total, nil
}

func intervalDuration(iv tracecontract.Interval) (time.Duration, error) {
	if iv.End <= iv.Start {
		return 0, nil
	}
	if iv.Start < 0 && iv.End > math.MaxInt64+iv.Start {
		return 0, fmt.Errorf("interval duration overflows time.Duration")
	}
	return time.Duration(iv.End - iv.Start), nil
}

func addDuration(total, d time.Duration) (time.Duration, error) {
	if d > time.Duration(math.MaxInt64)-total {
		return 0, fmt.Errorf("duration total overflows time.Duration")
	}
	return total + d, nil
}

// ComputeIntervalUnion takes a set of intervals and merges all overlapping or contiguous spans.
func ComputeIntervalUnion(intervals []tracecontract.Interval) []tracecontract.Interval {
	if len(intervals) == 0 {
		return nil
	}

	valid := make([]tracecontract.Interval, 0, len(intervals))
	for _, iv := range intervals {
		if iv.End > iv.Start {
			valid = append(valid, iv)
		}
	}
	if len(valid) == 0 {
		return nil
	}

	sort.Slice(valid, func(i, j int) bool {
		if valid[i].Start != valid[j].Start {
			return valid[i].Start < valid[j].Start
		}
		return valid[i].End < valid[j].End
	})

	merged := make([]tracecontract.Interval, 0, len(valid))
	current := valid[0]

	for i := 1; i < len(valid); i++ {
		next := valid[i]
		if next.Start <= current.End {
			if next.End > current.End {
				current.End = next.End
			}
		} else {
			merged = append(merged, current)
			current = next
		}
	}
	merged = append(merged, current)
	return merged
}

// UnionDuration computes the duration of the interval union. Empty, zero-length
// and inverted intervals contribute zero. It returns an error if the duration
// cannot be represented by time.Duration.
func UnionDuration(intervals []tracecontract.Interval) (time.Duration, error) {
	return SumDuration(ComputeIntervalUnion(intervals))
}

// ComputeTimeAccounting computes separate host, device and scheduler durations.
// It returns zero accounting and an error if any duration is unrepresentable.
func ComputeTimeAccounting(
	wall tracecontract.Interval,
	hostCalls []HostCallEvent,
	kernels []KernelEvent,
	sched SchedEvidence,
) (TimeAccounting, error) {
	if _, err := intervalDuration(wall); err != nil {
		return TimeAccounting{}, fmt.Errorf("wall: %w", err)
	}
	kernelIntervals := make([]tracecontract.Interval, 0, len(kernels))
	for _, k := range kernels {
		kernelIntervals = append(kernelIntervals, k.Span)
	}
	hostIntervals := make([]tracecontract.Interval, 0, len(hostCalls))
	for _, h := range hostCalls {
		if h.HasReturn {
			hostIntervals = append(hostIntervals, h.Span)
		}
	}
	summed, err := SumDuration(kernelIntervals)
	if err != nil {
		return TimeAccounting{}, fmt.Errorf("summed kernel duration: %w", err)
	}
	busy, err := UnionDuration(kernelIntervals)
	if err != nil {
		return TimeAccounting{}, fmt.Errorf("device busy duration: %w", err)
	}
	host, err := UnionDuration(hostIntervals)
	if err != nil {
		return TimeAccounting{}, fmt.Errorf("host API duration: %w", err)
	}
	runnable, scheduled, wait, err := sched.Durations()
	if err != nil {
		return TimeAccounting{}, fmt.Errorf("scheduler durations: %w", err)
	}
	return TimeAccounting{
		WallInterval:         wall,
		SummedKernelDuration: summed,
		BusyDuration:         busy,
		HostAPIDuration:      host,
		RunnableDuration:     runnable,
		ScheduledDuration:    scheduled,
		WaitDuration:         wait,
	}, nil
}
