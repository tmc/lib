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

// ptxBase is the logical image base for the PTX text. It is the same constant
// cmd/cudaprof uses when it annotates the profile, so a statement has one
// address whether it is written at capture time or synthesized at view time.
const ptxBase = cudaprof.PTXMappingStart

// sampleTypes reports the profile's value types. Stall counters exist under
// PC sampling or when instruction hotspot attribution is active.
func sampleTypes(pcSampling bool) []string {
	return sampleTypesWithHotspots(pcSampling, false)
}

func sampleTypesWithHotspots(pcSampling, hotspots bool) []string {
	if pcSampling || hotspots {
		return []string{
			"samples", "pc_samples", "inst_executed",
			"stall_mem", "stall_wait", "stall_scoreboard", "stall_math",
			"stall_branch", "stall_fetch", "stall_drain", "stall_dispatch",
			"stall_not_selected",
			"gpu_time",
		}
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

// ptxAddress returns the logical address of a one-based PTX line, or zero when
// the line is outside the text. Zero leaves the frame unmapped, which is the
// correct representation of an address the capture cannot establish.
func ptxAddress(offsets []uint64, line int) uint64 {
	if line < 1 || line >= len(offsets) {
		return 0
	}
	return ptxBase + offsets[line]
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

// rewriteSASSSourcePath replaces absolute source paths nvdisasm embeds in
// SASS line directives (including inline annotations) with relative bundle names.
// The compiler records build host paths that do not match the sources the capture
// bundles, so directives must be rewritten to match bundle names.
func rewriteSASSSourcePath(sass []byte, name string) []byte {
	const marker = `//## File "`
	const inlineMarker = ` inlined at "`
	out := make([]byte, 0, len(sass))
	for _, line := range strings.SplitAfter(string(sass), "\n") {
		i := strings.Index(line, marker)
		if i < 0 {
			out = append(out, line...)
			continue
		}
		rest := line[i+len(marker):]
		j := strings.IndexByte(rest, '"')
		if j < 0 {
			out = append(out, line...)
			continue
		}
		out = append(out, line[:i+len(marker)]...)
		filePath := rest[:j]
		if filePath == ".nv_debug_ptx_txt" {
			out = append(out, filePath...)
		} else {
			out = append(out, name...)
		}

		remaining := rest[j:]
		for {
			k := strings.Index(remaining, inlineMarker)
			if k < 0 {
				out = append(out, remaining...)
				break
			}
			out = append(out, remaining[:k+len(inlineMarker)]...)
			inlineRest := remaining[k+len(inlineMarker):]
			endQuote := strings.IndexByte(inlineRest, '"')
			if endQuote < 0 {
				out = append(out, inlineRest...)
				break
			}
			inlinePath := inlineRest[:endQuote]
			if inlinePath == ".nv_debug_ptx_txt" {
				out = append(out, inlinePath...)
			} else {
				out = append(out, name...)
			}
			remaining = inlineRest[endQuote:]
		}
	}
	return out
}
