package cudadiag

import (
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// SchedState represents a thread scheduling state from Linux scheduler tracepoints.
type SchedState string

const (
	// SchedRunning indicates the thread was actively executing on a CPU.
	SchedRunning SchedState = "running"
	// SchedRunnable indicates the thread was queued and ready to run, waiting for CPU.
	SchedRunnable SchedState = "runnable"
	// SchedBlocked indicates the thread was off-CPU waiting for an event, lock, or I/O.
	SchedBlocked SchedState = "blocked"
)

// SchedInterval captures one contiguous scheduling interval for a thread.
type SchedInterval struct {
	State SchedState             `json:"state"`
	Span  tracecontract.Interval `json:"span"`
	CPU   int                    `json:"cpu"`
}

// SchedEvidence holds operating system scheduler trace evidence for a target thread.
type SchedEvidence struct {
	Process      cudaworkload.ProcessGenID `json:"process"`
	TID          uint64                    `json:"tid"`
	ProbePresent bool                      `json:"probe_present"`
	Loss         uint64                    `json:"loss"`
	UnknownLoss  bool                      `json:"unknown_loss"`
	Intervals    []SchedInterval           `json:"intervals"`
}

// Matches reports whether the scheduler evidence belongs to the specified process and thread.
func (s SchedEvidence) Matches(proc cudaworkload.ProcessGenID, tid uint64) bool {
	if proc.IsZero() || tid == 0 {
		return false
	}
	return s.Process == proc && s.TID == tid
}

// Durations computes the total runnable (CPU starvation), scheduled (running on CPU),
// and wait (blocked off-CPU) times.
func (s SchedEvidence) Durations() (runnable, scheduled, wait time.Duration) {
	var rNs, sNs, wNs int64
	for _, iv := range s.Intervals {
		dur := iv.Span.Duration()
		switch iv.State {
		case SchedRunnable:
			rNs += dur
		case SchedRunning:
			sNs += dur
		case SchedBlocked:
			wNs += dur
		}
	}
	return time.Duration(rNs) * time.Nanosecond,
		time.Duration(sNs) * time.Nanosecond,
		time.Duration(wNs) * time.Nanosecond
}

// Validate checks whether scheduler evidence is present, complete, and properly matched.
func (s SchedEvidence) Validate(proc cudaworkload.ProcessGenID, tid uint64) error {
	if !s.ProbePresent {
		return errors.New("scheduler probe missing or inactive")
	}
	if s.UnknownLoss {
		return errors.New("unknown loss detected in scheduler stream")
	}
	if s.Loss > 0 {
		return fmt.Errorf("scheduler events lost: %d dropped", s.Loss)
	}
	if !s.Matches(proc, tid) {
		return fmt.Errorf("scheduler identity mismatch: evidence (%v, tid:%d) does not match target (%v, tid:%d)",
			s.Process, s.TID, proc, tid)
	}
	intervals := append([]SchedInterval(nil), s.Intervals...)
	sort.Slice(intervals, func(i, j int) bool { return intervals[i].Span.Start < intervals[j].Span.Start })
	for i, iv := range intervals {
		if iv.State != SchedRunning && iv.State != SchedRunnable && iv.State != SchedBlocked {
			return errors.New("unknown scheduler state")
		}
		if iv.Span.End < iv.Span.Start {
			return errors.New("inverted scheduler interval")
		}
		if i > 0 && intervals[i-1].Span.End > iv.Span.Start {
			return errors.New("overlapping scheduler intervals")
		}
	}
	return nil
}
