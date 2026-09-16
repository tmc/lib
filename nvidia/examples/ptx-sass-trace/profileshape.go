package main

// This file holds the decisions about what the emitted profile claims, as
// distinct from what the workload measures. Each is a property a pprof consumer
// reads directly from the file, so each is set from something the capture
// actually knows rather than left at its zero value.

import (
	"runtime"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// sampleTypes reports the profile's value types. Stall counters exist only
// under PC sampling, which measures them.
func sampleTypes(pcSampling bool) []string {
	return sampleTypesWithHotspots(pcSampling, false)
}

// sampleTypesWithHotspots reports the value types for a run. The two modes get
// disjoint sets, and -pcsampling and -hotspots are rejected together, so a
// profile never carries both a measured and a modeled value.
//
// Hotspot attribution derives its numbers from a static opcode table, not from
// the device. Writing them as pc_samples, inst_executed, stall_* or gpu_time
// would name a model's output after a hardware counter, and nothing downstream
// could tell the two apart: pprof shows a sample type, not its lineage. They
// are reported as modeled_cost, which no measurement writes.
//
// samples and gpu_time mean the same thing in all three modes: one row per
// launch, holding that launch's measured CUDA event interval. The modeled rows
// under -hotspots add modeled_cost and leave both of those zero.
func sampleTypesWithHotspots(pcSampling, hotspots bool) []string {
	if pcSampling {
		return []string{
			"samples", "pc_samples", "inst_executed",
			"stall_mem", "stall_wait", "stall_scoreboard", "stall_math",
			"stall_branch", "stall_fetch", "stall_drain", "stall_dispatch",
			"stall_not_selected",
			// A reason the classifier does not recognize is counted here
			// rather than dropped. Selecting the type is what puts it in the
			// capture: SetSampleTypes writes only what this list names, so a
			// default arm upstream counts into a column the artifact does not
			// carry unless the column is asked for.
			"stall_other",
			"gpu_time",
		}
	}
	if hotspots {
		return []string{"samples", "modeled_cost", "gpu_time"}
	}
	return []string{"samples", "gpu_time"}
}

// numericLabelUnits names the labels holding numbers. pprof reads a bare value
// or "a:b" in -tagfocus as a numeric range, which never matches a string label,
// so an index left as a string cannot be filtered by range.
func numericLabelUnits() map[string]string {
	return map[string]string{
		"pass":       "index",
		"layer":      "index",
		"moe.expert": "index",
		"moe.slot":   "index",
	}
}

// goFrames reports the Go call stack above the caller, outermost last, as pprof
// expects. It skips its own frame and its caller's, so the leaf is the launch
// site rather than the code that asked for the stack.
func goFrames(skip int) []cudaprof.Frame {
	var pcs [32]uintptr
	n := runtime.Callers(skip+2, pcs[:])
	if n == 0 {
		return nil
	}
	var out []cudaprof.Frame
	frames := runtime.CallersFrames(pcs[:n])
	for {
		f, more := frames.Next()
		if f.Function == "" {
			break
		}
		// Filter out context label plumbing so the callstack shows application signal only.
		if strings.Contains(f.Function, "cudaprof.Do") || strings.Contains(f.Function, "runtime/pprof.Do") {
			if !more {
				break
			}
			continue
		}
		// Stop at the runtime's entry into main; the scheduler frames above it
		// describe the process, not the launch.
		if f.Function == "main.main" {
			out = append(out, cudaprof.Frame{FunctionName: f.Function, FileName: trimPath(f.File), Line: int64(f.Line)})
			break
		}
		out = append(out, cudaprof.Frame{FunctionName: f.Function, FileName: trimPath(f.File), Line: int64(f.Line)})
		if !more {
			break
		}
	}
	return out
}

// trimPath reduces a compile-time absolute path to its base name. The capture
// bundle holds the Go sources by base name, so a path from the build host would
// resolve to nothing on a reader's machine.
func trimPath(path string) string {
	if i := strings.LastIndexByte(path, '/'); i >= 0 {
		return path[i+1:]
	}
	return path
}
