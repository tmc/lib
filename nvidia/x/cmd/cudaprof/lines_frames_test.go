package main

import (
	"testing"

	"github.com/google/pprof/profile"
)

// A frame is one location of a sample's stack.
type frame struct {
	function string
	file     string
	line     int64
}

// TestDeviceFrames checks the shape the line tables read out of a sample.
// A device stack is leaf first: the SASS instruction, then the tiers above
// it, then the kernel's own frame carrying the program line. The tiers in
// between are named for the kernel with a suffix, and an inlined frame is
// one of them: it names where a device function's code came from, which is
// a header of the generated source rather than a line of the program, so
// it must not be mistaken for the kernel frame that carries the site.
func TestDeviceFrames(t *testing.T) {
	for _, tt := range []struct {
		name     string
		stack    []frame
		kernel   string
		sassLine int
		file     string
		line     int
	}{{
		name:     "sass then kernel",
		stack:    []frame{{"fused_gate.SASS", "fused_gate.sass", 2}, {"fused_gate", "kernels.go", 15}},
		kernel:   "fused_gate",
		sassLine: 2,
		file:     "kernels.go",
		line:     15,
	}, {
		name: "every tier",
		stack: []frame{
			{"fused_gate.SASS", "fused_gate.sass", 2},
			{"fused_gate.PTX", "fused_gate.ptx", 40},
			{"fused_gate.inlined", "math_functions.h", 1200},
			{"fused_gate", "kernels.go", 15},
			{"main.main", "main.go", 46},
		},
		kernel:   "fused_gate",
		sassLine: 2,
		file:     "kernels.go",
		line:     15,
	}, {
		name:   "inlined leaf without sass",
		stack:  []frame{{"fused_gate.inlined", "math_functions.h", 1200}, {"fused_gate", "kernels.go", 15}},
		kernel: "fused_gate",
		file:   "kernels.go",
		line:   15,
	}, {
		// The kernel frame of a capture with no unit names the
		// disassembly, which is a tier and not a program line.
		name:     "kernel frame naming a tier file",
		stack:    []frame{{"fused_gate.SASS", "fused_gate.sass", 2}, {"fused_gate", "fused_gate.sass", 7}},
		kernel:   "fused_gate",
		sassLine: 2,
	}} {
		t.Run(tt.name, func(t *testing.T) {
			f := deviceFrames(stack(tt.stack))
			if f.kernel != tt.kernel {
				t.Errorf("kernel = %q, want %q", f.kernel, tt.kernel)
			}
			if f.sassLine != tt.sassLine {
				t.Errorf("SASS ordinal = %d, want %d", f.sassLine, tt.sassLine)
			}
			if f.site.File != tt.file || f.site.Line != tt.line {
				t.Errorf("site = %s:%d, want %s:%d", f.site.File, f.site.Line, tt.file, tt.line)
			}
		})
	}
}

// stack builds a sample whose locations are the given frames, leaf first.
func stack(frames []frame) *profile.Sample {
	s := &profile.Sample{Value: []int64{1}}
	for i, f := range frames {
		s.Location = append(s.Location, &profile.Location{
			ID: uint64(i + 1),
			Line: []profile.Line{{
				Function: &profile.Function{ID: uint64(i + 1), Name: f.function, Filename: f.file},
				Line:     f.line,
			}},
		})
	}
	return s
}
