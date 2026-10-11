package cudatracehost

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var cpuField = regexp.MustCompile(`\[(\d+)\]`)

// SchedulerEvent preserves kernel scheduling evidence. PrevState is the raw
// tracepoint state, not an inferred cause of blocking. Timestamps use tracefs
// mono clock. No application or GPU association is inferred.
type SchedulerEvent struct {
	Clock                         string
	Raw, Kind, PrevState          string
	Timestamp                     uint64
	CPU                           uint32
	PreviousTID, NextTID, WakeTID uint32
}

// ParseScheduler decodes sched_switch and sched_wakeup lines. Event PID fields
// refer to Linux TIDs. Process membership must be checked independently.
func ParseScheduler(line string) (SchedulerEvent, error) {
	e := SchedulerEvent{Raw: line, Clock: "linux_monotonic"}
	o, err := Parse(line)
	if err != nil {
		return e, err
	}
	e.Timestamp = o.Timestamp
	e.Kind = o.Name
	cpu := cpuField.FindStringSubmatch(line)
	if len(cpu) != 2 {
		return e, fmt.Errorf("cudatracehost: missing scheduler cpu")
	}
	n, err := strconv.ParseUint(cpu[1], 10, 32)
	if err != nil {
		return e, err
	}
	e.CPU = uint32(n)
	values := map[string]string{}
	for _, f := range strings.Fields(line) {
		k, v, ok := strings.Cut(f, "=")
		if ok {
			if _, dup := values[k]; dup {
				return e, fmt.Errorf("cudatracehost: duplicate scheduler field %s", k)
			}
			values[k] = v
		}
	}
	pid := func(k string) (uint32, error) {
		v, ok := values[k]
		if !ok {
			return 0, fmt.Errorf("cudatracehost: missing scheduler %s", k)
		}
		n, err := strconv.ParseUint(v, 10, 32)
		return uint32(n), err
	}
	switch e.Kind {
	case "sched_switch":
		e.PreviousTID, err = pid("prev_pid")
		if err != nil {
			return e, err
		}
		e.NextTID, err = pid("next_pid")
		if err != nil {
			return e, err
		}
		e.PrevState = values["prev_state"]
		if e.PrevState == "" {
			return e, fmt.Errorf("cudatracehost: missing scheduler state")
		}
	case "sched_wakeup", "sched_wakeup_new":
		e.WakeTID, err = pid("pid")
		if err != nil {
			return e, err
		}
	default:
		return e, fmt.Errorf("cudatracehost: unsupported scheduler event %s", e.Kind)
	}
	return e, nil
}
