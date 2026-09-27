package nvrtcx

import (
	"strings"
	"testing"
)

func TestTraceHeader_ContainsRequiredSymbols(t *testing.T) {
	hdr := TraceHeader()

	required := []string{
		"rl_trace",
		"trace_reset",
		"trace_finish",
		"trace_reset_kernel",
		"trace_read_kernel",
		"trace_write_byte",
		"dump_var",
		"dump_vars",
		"atomicCAS",
		"14695981039346656037ULL",
		"1099511628211ULL",
	}

	for _, req := range required {
		if !strings.Contains(hdr, req) {
			t.Errorf("TraceHeader missing required string %q", req)
		}
	}
}

func TestEmitDumpVar(t *testing.T) {
	got := EmitDumpVar("x + 1")
	want := "dump_var(x + 1);\n"
	if got != want {
		t.Errorf("EmitDumpVar: got %q, want %q", got, want)
	}
}

func TestEmitDumpVars(t *testing.T) {
	got := EmitDumpVars("arr", "16")
	want := "dump_vars(arr, 16);\n"
	if got != want {
		t.Errorf("EmitDumpVars: got %q, want %q", got, want)
	}
}
