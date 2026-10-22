package dsv4eval

import (
	"bytes"
	"strings"
	"testing"
)

func TestTraceUpstreamCountedSeparators(t *testing.T) {
	for _, tc := range []struct{ name, wire, want string }{
		{"uncounted LF", "MODEL_OUTPUT_BEGIN bytes=9\nAnswer: A\nMODEL_OUTPUT_END\n", "Answer: A"},
		{"counted LF", "MODEL_OUTPUT_BEGIN bytes=10\nAnswer: A\nMODEL_OUTPUT_END\n", "Answer: A\n"},
		{"empty", "MODEL_OUTPUT_BEGIN bytes=0\nMODEL_OUTPUT_END\n", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := "# ds4-eval trace\n===== CASE 1/1 source/id =====\n" + tc.wire + "===== SUMMARY =====\ntotal: 1\n"
			trace, err := ReadTrace(strings.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			if got := trace.Cases[0].Blocks["MODEL_OUTPUT"]; got != tc.want {
				t.Fatalf("body=%q want%q", got, tc.want)
			}
		})
	}
	bad := "# ds4-eval trace\n===== CASE 1/1 source/id =====\nMODEL_OUTPUT_BEGIN bytes=9\nAnswer: AMODEL_OUTPUT_END\n===== SUMMARY =====\n"
	if _, err := ReadTrace(strings.NewReader(bad)); err == nil {
		t.Fatal("missing separator accepted")
	}
}

func TestTraceRoundTrip(t *testing.T) {
	var out bytes.Buffer
	w := NewTraceWriter(&out, TraceHeader{Fields: map[string]string{"source": "go", "ctx": "128"}})
	if err := w.Case(TraceCase{
		Index: 1, Total: 1, Source: "GPQA Diamond", ID: "case-1",
		Fields: map[string]string{"status": "PASSED", "answer_kind": "choice"},
		Blocks: map[string]string{"MODEL_OUTPUT": "reason\nAnswer: B\n"},
	}); err != nil {
		t.Fatal(err)
	}
	if err := w.Summary(map[string]string{"passed": "1"}); err != nil {
		t.Fatal(err)
	}
	trace, err := ReadTrace(bytes.NewReader(out.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if trace.Header.Fields["ctx"] != "128" || len(trace.Cases) != 1 || trace.Cases[0].Blocks["MODEL_OUTPUT"] != "reason\nAnswer: B\n" {
		t.Fatalf("trace = %+v", trace)
	}
	if !strings.Contains(out.String(), "MODEL_OUTPUT_BEGIN bytes=17\n") {
		t.Fatalf("counted block missing: %q", out.String())
	}
	if trace.Cases[0].Fields["source"] != "GPQA Diamond" || trace.Cases[0].Fields["id"] != "case-1" {
		t.Fatal("upstream regrader identity fields missing")
	}
}

func TestTraceIdentityFields(t *testing.T) {
	for _, key := range []string{"source", "id"} {
		var out bytes.Buffer
		writer := NewTraceWriter(&out, TraceHeader{})
		fields := map[string]string{key: "wrong"}
		if err := writer.Case(TraceCase{Index: 1, Total: 1, Source: "s", ID: "i", Fields: fields}); err == nil {
			t.Fatal("header/field mismatch accepted")
		}
		if strings.Contains(out.String(), "===== CASE") {
			t.Fatal("rejected identity emitted partial case")
		}
	}
	fields := map[string]string{"status": "PASSED"}
	var out bytes.Buffer
	w := NewTraceWriter(&out, TraceHeader{})
	if err := w.Case(TraceCase{Index: 1, Total: 1, Source: "s", ID: "i", Fields: fields}); err != nil {
		t.Fatal(err)
	}
	if _, ok := fields["source"]; ok {
		t.Fatal("writer mutated caller fields")
	}
	for _, key := range []string{"source", "id"} {
		bad := "# ds4-eval trace\n===== CASE 1/1 s/i =====\n" + key + ": wrong\n===== SUMMARY =====\n"
		if _, err := ReadTrace(strings.NewReader(bad)); err == nil {
			t.Fatal("read identity mismatch accepted")
		}
	}
}

func TestReadTraceRejectsMutations(t *testing.T) {
	base := "# ds4-eval trace\n===== CASE 1/1 s/i =====\nstatus: PASSED\nMODEL_OUTPUT_BEGIN bytes=2\nX\nMODEL_OUTPUT_END\n===== SUMMARY =====\n"
	tests := []struct {
		name string
		text string
	}{
		{"wrong byte count", strings.Replace(base, "bytes=2", "bytes=3", 1)},
		{"duplicate field", strings.Replace(base, "status: PASSED\n", "status: PASSED\nstatus: FAILED\n", 1)},
		{"missing summary", strings.TrimSuffix(base, "===== SUMMARY =====\n")},
		{"bad terminator", strings.Replace(base, "MODEL_OUTPUT_END", "MODEL_OUTPUT_DONE", 1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ReadTrace(strings.NewReader(tt.text)); err == nil {
				t.Fatal("mutated trace accepted")
			}
		})
	}
}
