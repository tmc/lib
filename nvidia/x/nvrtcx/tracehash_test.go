package nvrtcx

import (
	"fmt"
	"strings"
	"testing"
)

func TestTraceHeader_ContainsRequiredSymbols(t *testing.T) {
	hdr := TraceHeader()

	required := []string{
		"rl_trace",
		"trace_write_byte",
		"dump_var",
		"dump_vars",
		"atomicCAS",
		"rl_trace_next",
		"trace_ordered_float",
		"14695981039346656037ULL",
		"1099511628211ULL",
	}

	for _, req := range required {
		if !strings.Contains(hdr, req) {
			t.Errorf("TraceHeader missing required string %q", req)
		}
	}
}

func TestTraceHeader_ExactFoldGoldens(t *testing.T) {
	hdr := TraceHeader()
	for _, want := range []string{
		"__device__ unsigned long long rl_trace = 14695981039346656037ULL;",
		"__device__ unsigned int rl_trace_next = 0;",
		"__device__ unsigned int rl_trace_last_bits = 0;",
		"trace_write_byte((unsigned char)(v & 0xff));",
		"trace_write_byte((unsigned char)((v >> 8) & 0xff));",
		"trace_write_byte((unsigned char)((v >> 16) & 0xff));",
		"trace_write_byte((unsigned char)((v >> 24) & 0xff));",
		"trace_write_byte((unsigned char)(bits.u & 0xff));",
		"trace_write_byte((unsigned char)((v >> 8) & 0xff));",
		"dump_var((unsigned int)(v >> 32));",
		"trace_write_byte((unsigned char)((bits.u >> 24) & 0xff));",
		"while (atomicAdd(&rl_trace_next, 0U) != ticket)",
	} {
		if !strings.Contains(hdr, want) {
			t.Errorf("TraceHeader missing golden %q", want)
		}
	}
	if got := EmitOrderedDumpVar("value", "base + idx", "limit"); got != "trace_ordered_float(value, base + idx, limit);\n" {
		t.Fatalf("EmitOrderedDumpVar = %q", got)
	}
}

func ExampleEmitOrderedDumpVar() {
	fmt.Print(EmitOrderedDumpVar("value", "base + index", "limit"))
	// Output: trace_ordered_float(value, base + index, limit);
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
