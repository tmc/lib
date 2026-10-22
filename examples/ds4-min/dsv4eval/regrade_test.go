package dsv4eval

import (
	"fmt"
	"strings"
	"testing"
)

func regradeFixture(source, id, body string) string {
	separator := ""
	if body != "" && !strings.HasSuffix(body, "\n") {
		separator = "\n"
	}
	return fmt.Sprintf("# ds4-eval trace\n===== CASE 1/1 %s/%s =====\nstatus: PASSED\nexpected: B\nanswer_kind: choice\nMODEL_OUTPUT_BEGIN bytes=%d\n%s%sMODEL_OUTPUT_END\n===== SUMMARY =====\ntotal: 1\n", source, id, len(body), body, separator)
}

func TestRegradeCountedTrace(t *testing.T) {
	for _, body := range []string{"reason\nAnswer: B\n", "reason\nAnswer: B", strings.Repeat("reason", 22000) + "\nAnswer: B", "MODEL_OUTPUT_END\n===== CASE 99/99 fake/source =====\nstatus: FAILED\nexpected: A\nAnswer: B"} {
		r, err := Regrade(strings.NewReader(regradeFixture("GPQA Diamond", "recNu3MXkvWUzHZr9", body)))
		if err != nil {
			t.Fatal(err)
		}
		if r.Cases != 1 || r.Parsed != 1 || r.Passed != 1 || r.Failed != 0 || r.Unknown != 0 || r.ParseErrors != 0 {
			t.Fatalf("report=%+v", r)
		}
	}
}

func TestRegradeUnknownCase(t *testing.T) {
	r, err := Regrade(strings.NewReader(regradeFixture("no-such", "source", "Answer: B")))
	if err != nil || r.Unknown != 1 || r.Parsed != 0 {
		t.Fatalf("report=%+v err=%v", r, err)
	}
}

func TestRegradeRejectsMalformedCountedTrace(t *testing.T) {
	valid := regradeFixture("GPQA Diamond", "recNu3MXkvWUzHZr9", "reason\nAnswer: B")
	// 17 is valid: it counts the existing LF as part of the body. 18 crosses
	// the allowed separator and consumes the first byte of the terminator.
	for _, input := range []string{
		strings.Replace(valid, "bytes=16", "bytes=10", 1),
		strings.Replace(valid, "bytes=16", "bytes=18", 1),
		strings.Replace(valid, "bytes=16", "bytes=9223372036854775807", 1),
		valid[:strings.Index(valid, "MODEL_OUTPUT_END")],
		strings.Replace(valid, "===== SUMMARY =====", "missing summary", 1),
		strings.TrimPrefix(valid, "# ds4-eval trace\n"),
	} {
		r, err := Regrade(strings.NewReader(input))
		if err == nil || r.ParseErrors != 1 {
			t.Fatalf("malformed report=%+v err=%v", r, err)
		}
	}
}

func TestSuiteName(t *testing.T) {
	if got, err := SuiteName("all"); err != nil || got != Core|Hard {
		t.Fatalf("all = %v, %v", got, err)
	}
	if _, err := SuiteName("missing"); err == nil {
		t.Fatal("unknown suite accepted")
	}
}
