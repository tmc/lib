package cudaprof

import (
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

// KernelExecution represents a device kernel execution attributed to a host call stack.
// The call stack is the stack of the Do call site (not per-launch).
// If the kernel has no external correlation ID, Callstack is nil.
type KernelExecution struct {
	KernelName string
	Callstack  []string
	StartNs    uint64
	EndNs      uint64
}

// ExportFoldedStacks writes folded stacks to w in flamegraph.pl format:
//
//	cpu_frame;...;[GPU]kernel count
//
// Stacks are weighted by device busy time computed from the union of device execution
// intervals, ensuring that concurrent/overlapping executions on the device do not
// double-count duration, and host-side duration (such as sleeping in a Do region)
// does not inflate device execution weight.
//
// If a kernel execution has no host call stack (Callstack is nil or empty),
// it is formatted as "[GPU]kernel count".
// Lines are sorted alphabetically for deterministic output.
func ExportFoldedStacks(w io.Writer, executions []KernelExecution) error {
	if w == nil {
		return errors.New("cudaprof: nil writer for folded stack export")
	}
	if len(executions) == 0 {
		return nil
	}

	byStack := make(map[string][]timeInterval)
	for _, e := range executions {
		var stackKey string
		if len(e.Callstack) > 0 {
			stackKey = strings.Join(e.Callstack, ";") + ";[GPU]" + e.KernelName
		} else {
			stackKey = "[GPU]" + e.KernelName
		}
		byStack[stackKey] = append(byStack[stackKey], timeInterval{start: e.StartNs, end: e.EndNs})
	}

	keys := make([]string, 0, len(byStack))
	for k := range byStack {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		intervals := byStack[k]
		dur := computeUnionDuration(intervals)
		if _, err := fmt.Fprintf(w, "%s %d\n", k, dur); err != nil {
			return fmt.Errorf("cudaprof: export folded stacks: %w", err)
		}
	}
	return nil
}

type timeInterval struct {
	start uint64
	end   uint64
}

func computeUnionDuration(intervals []timeInterval) uint64 {
	if len(intervals) == 0 {
		return 0
	}
	sorted := make([]timeInterval, len(intervals))
	copy(sorted, intervals)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].start == sorted[j].start {
			return sorted[i].end < sorted[j].end
		}
		return sorted[i].start < sorted[j].start
	})

	var total uint64
	curStart := sorted[0].start
	curEnd := sorted[0].end
	if curEnd < curStart {
		curEnd = curStart
	}

	for _, iv := range sorted[1:] {
		end := iv.end
		if end < iv.start {
			end = iv.start
		}
		if iv.start <= curEnd {
			if end > curEnd {
				curEnd = end
			}
		} else {
			total += curEnd - curStart
			curStart = iv.start
			curEnd = end
		}
	}
	total += curEnd - curStart
	return total
}
