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

// ptxBase is the logical image base for the PTX text, matching the constant
// cmd/cudaprof uses when it synthesizes the same mapping at view time. It sits
// below cudaprof.DefaultMappingStart so the two images never overlap.
const ptxBase = uint64(0x7e000000)

// sampleTypes reports the profile's value types. Stall counters exist only
// under PC sampling, so a run without it advertises the two types it can fill
// rather than five of which three are always zero.
func sampleTypes(pcSampling bool) []string {
	if pcSampling {
		return []string{"samples", "stall_mem", "stall_wait", "stall_scoreboard", "gpu_time"}
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

// lineOffsets indexes the byte offset of each one-based line in text, with an
// end-of-file sentinel. PTX addresses are byte offsets into the PTX text, so a
// statement's address is the offset of the line it starts on; cmd/cudaprof
// derives the same addresses the same way when it disassembles the PTX.
func lineOffsets(text string) []uint64 {
	offsets := []uint64{0}
	var offset uint64
	for _, line := range strings.SplitAfter(text, "\n") {
		offsets = append(offsets, offset)
		offset += uint64(len(line))
	}
	return append(offsets, offset)
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

// rewriteSASSSourcePath replaces the absolute source path nvdisasm embeds in
// SASS line directives with name. The compiler records the path it built from,
// which exists only on the build host and does not match the source the capture
// bundles, so a reader that trusts the directive resolves nothing.
func rewriteSASSSourcePath(sass []byte, name string) []byte {
	const marker = `//## File "`
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
		out = append(out, name...)
		out = append(out, rest[j:]...)
	}
	return out
}
