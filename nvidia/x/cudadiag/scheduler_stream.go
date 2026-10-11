package cudadiag

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// SchedulerEvent is a decoded Linux sched_switch or sched_wakeup event.
// Callers must preserve the capture clock and the kernel's thread identifiers.
type SchedulerEvent struct {
	Kind                          string
	Timestamp                     int64
	CPU                           int
	PreviousTID, NextTID, WakeTID uint64
	PreviousState                 string
}

// SchedulerEvidence derives intervals between observed target-thread transitions.
// It does not extrapolate before the first or after the last transition. The
// process generation must come from the captured process, not PID alone. A raw
// tracefs capture without quantified loss must pass unknownLoss=true; its
// intervals remain diagnostic evidence and cannot admit a complete diagnosis.
func SchedulerEvidence(proc cudaworkload.ProcessGenID, tid uint64, clock string, events []SchedulerEvent, loss uint64, unknownLoss bool) (SchedEvidence, error) {
	out := SchedEvidence{Process: proc, TID: tid, ProbePresent: true, Loss: loss, UnknownLoss: unknownLoss}
	if proc.PID == 0 || proc.Generation == 0 || tid == 0 || clock == "" {
		return out, fmt.Errorf("scheduler evidence: missing process, thread or clock identity")
	}
	if len(events) > 65536 {
		return out, fmt.Errorf("scheduler evidence: event limit exceeded")
	}
	var state SchedState
	var start int64
	var cpu int
	transition := func(next SchedState, timestamp int64, processor int) {
		if state != "" && state != next && timestamp > start {
			out.Intervals = append(out.Intervals, SchedInterval{State: state, Span: tracecontract.Interval{Start: start, End: timestamp, Clock: clock}, CPU: cpu})
		}
		if state != next {
			state, start, cpu = next, timestamp, processor
		}
	}
	for i, ev := range events {
		if ev.Timestamp < 0 || (i > 0 && ev.Timestamp < events[i-1].Timestamp) {
			return out, fmt.Errorf("scheduler evidence: unordered timestamps")
		}
		switch ev.Kind {
		case "sched_wakeup", "sched_wakeup_new":
			if ev.WakeTID == tid {
				transition(SchedRunnable, ev.Timestamp, ev.CPU)
			}
		case "sched_switch":
			if ev.PreviousTID == tid {
				switch ev.PreviousState {
				case "R", "R+":
					transition(SchedRunnable, ev.Timestamp, ev.CPU)
				case "S", "D", "I":
					transition(SchedBlocked, ev.Timestamp, ev.CPU)
				default:
					return out, fmt.Errorf("scheduler evidence: unsupported previous state %q", ev.PreviousState)
				}
			}
			if ev.NextTID == tid {
				transition(SchedRunning, ev.Timestamp, ev.CPU)
			}
		default:
			return out, fmt.Errorf("scheduler evidence: unsupported event %q", ev.Kind)
		}
	}
	return out, nil
}
