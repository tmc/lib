package cudadiag

import (
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

// HostCallEvent represents a host runtime or driver API invocation.
type HostCallEvent struct {
	Span      tracecontract.Interval `json:"span"`
	Name      string                 `json:"name"`
	HasReturn bool                   `json:"has_return"`
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
	RunnableDuration     time.Duration          `json:"runnable_duration"`
	ScheduledDuration    time.Duration          `json:"scheduled_duration"`
	WaitDuration         time.Duration          `json:"wait_duration"`
}

// SumDuration computes the arithmetic sum of durations across all provided intervals.
func SumDuration(intervals []tracecontract.Interval) time.Duration {
	var total int64
	for _, iv := range intervals {
		if iv.End > iv.Start {
			total += iv.End - iv.Start
		}
	}
	return time.Duration(total) * time.Nanosecond
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

// UnionDuration calculates the Lebesgue measure (total non-overlapping duration)
// of the union of the given intervals.
func UnionDuration(intervals []tracecontract.Interval) time.Duration {
	merged := ComputeIntervalUnion(intervals)
	var total int64
	for _, iv := range merged {
		total += iv.End - iv.Start
	}
	return time.Duration(total) * time.Nanosecond
}

// ComputeTimeAccounting computes the separated time metrics across host, device,
// and scheduler states.
func ComputeTimeAccounting(
	wall tracecontract.Interval,
	hostCalls []HostCallEvent,
	kernels []KernelEvent,
	sched SchedEvidence,
) TimeAccounting {
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

	runnable, scheduled, wait := sched.Durations()

	return TimeAccounting{
		WallInterval:         wall,
		SummedKernelDuration: SumDuration(kernelIntervals),
		BusyDuration:         UnionDuration(kernelIntervals),
		HostAPIDuration:      UnionDuration(hostIntervals),
		RunnableDuration:     runnable,
		ScheduledDuration:    scheduled,
		WaitDuration:         wait,
	}
}
