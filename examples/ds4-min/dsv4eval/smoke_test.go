package dsv4eval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func smokeTrace(status, closeKind string, summary bool) string {
	var b strings.Builder
	b.WriteString("# ds4-eval trace\n")
	b.WriteString("build: dgx@92069b8\nmodel_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n")
	b.WriteString("suite: hard-smoke\nctx: 4096\ntemp: 0\nthink: HIGH\n")
	cs := Cases()[:12]
	for i, c := range cs {
		index := i + 1
		answer := smokeAnswer(c)
		picked := answer
		if status == "FAILED" {
			picked = "wrong"
		}
		b.WriteString("===== CASE ")
		b.WriteString(strconv.Itoa(index))
		b.WriteString("/12 ")
		b.WriteString(c.Source)
		b.WriteString("/")
		b.WriteString(c.ID)
		b.WriteString(" =====\npicked: ")
		b.WriteString(picked)
		b.WriteString("\nexpected: ")
		b.WriteString(answer)
		b.WriteString("\nstatus: ")
		b.WriteString(status)
		b.WriteString("\nthink_close: ")
		b.WriteString(closeKind)
		b.WriteString("\nMODEL_OUTPUT_BEGIN bytes=")
		b.WriteString(strconv.Itoa(len([]byte("Answer: " + picked + "\n"))))
		b.WriteString("\nAnswer: ")
		b.WriteString(picked)
		b.WriteString("\nMODEL_OUTPUT_END\n")
	}
	if summary {
		passed, failed, incomplete := 0, 0, 0
		switch status {
		case "PASSED":
			passed = 12
		case "FAILED":
			failed = 12
		case "INCOMPLETE":
			incomplete = 12
		}
		fmt.Fprintf(&b, "===== SUMMARY =====\npassed: %d\nfailed: %d\nincomplete: %d\ntotal: 12\n", passed, failed, incomplete)
	}
	return b.String()
}

func smokeAnswer(c Case) string {
	switch c.Kind {
	case Integer, Rational:
		return "1"
	default:
		return "A"
	}
}

func TestCompareSmoke(t *testing.T) {
	left := smokeTrace("PASSED", "natural", true)
	right := smokeTrace("PASSED", "natural", true)
	r, err := CompareSmoke(strings.NewReader(left), strings.NewReader(right))
	if err != nil || !r.Pass() || r.Ran != 12 || r.Skipped != 0 || r.StatusMatches != 12 || r.ThinkCloseMatches != 12 {
		t.Fatalf("report=%+v err=%v", r, err)
	}
	if got := FormatSmoke(r); got != "ds4-eval: smoke cases=12 ran=12 skipped=0 status_matches=12/12 passed=12/12 think_close_matches=12/12 verdict=PASS" {
		t.Fatalf("summary=%q", got)
	}
}

func TestCompareSmokeFailsClosed(t *testing.T) {
	tests := []struct {
		name string
		left func(string) string
		want string
	}{
		{"missing summary", func(s string) string { return smokeTrace(s, "natural", false) }, "missing summary"},
		{"missing summary counters", func(s string) string {
			return strings.Replace(smokeTrace(s, "natural", true), "passed: 12\n", "", 1)
		}, "missing summary passed"},
		{"wrong summary counters", func(s string) string {
			return strings.Replace(smokeTrace(s, "natural", true), "passed: 12\n", "passed: 11\n", 1)
		}, "summary passed"},
		{"missing provenance", func(s string) string {
			return strings.Replace(smokeTrace(s, "natural", true), "model_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n", "", 1)
		}, "missing model_sha256 provenance"},
		{"unknown status", func(s string) string { return smokeTrace("SKIPPED", "natural", true) }, "nonterminal status"},
		{"missing model output", func(s string) string {
			return strings.Replace(smokeTrace(s, "natural", true), "MODEL_OUTPUT_BEGIN bytes=10\nAnswer: A\nMODEL_OUTPUT_END\n", "", 1)
		}, "missing model output"},
		{"declared answer disagrees", func(s string) string {
			trace := smokeTrace(s, "natural", true)
			return strings.Replace(trace, "MODEL_OUTPUT_BEGIN bytes=10\nAnswer: A\nMODEL_OUTPUT_END\n", "MODEL_OUTPUT_BEGIN bytes=14\nAnswer: wrong\nMODEL_OUTPUT_END\n", 1)
		}, "picked answer disagrees"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := CompareSmoke(strings.NewReader(tt.left("PASSED")), strings.NewReader(smokeTrace("PASSED", "natural", true)))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("err=%v, want %q", err, tt.want)
			}
		})
	}
	left := smokeTrace("PASSED", "natural", true)
	right := smokeTrace("FAILED", "soft_forced", true)
	r, err := CompareSmoke(strings.NewReader(left), strings.NewReader(right))
	if err != nil || r.Pass() || r.StatusMatches != 0 || r.ThinkCloseMatches != 0 {
		t.Fatalf("mutation report=%+v err=%v", r, err)
	}
	failed, err := CompareSmoke(strings.NewReader(smokeTrace("FAILED", "natural", true)), strings.NewReader(smokeTrace("FAILED", "natural", true)))
	if err != nil || failed.Pass() || failed.StatusMatches != 12 || failed.LeftPassed != 0 || failed.RightPassed != 0 {
		t.Fatalf("identical failure report=%+v err=%v", failed, err)
	}
}

func TestCompareSmokeRejectsDuplicateCaseIdentity(t *testing.T) {
	trace := smokeTrace("PASSED", "natural", true)
	lines := strings.Split(trace, "\n")
	first, second := "", ""
	for _, line := range lines {
		if strings.HasPrefix(line, "===== CASE 1/12 ") {
			first = strings.TrimSuffix(strings.TrimPrefix(line, "===== CASE 1/12 "), " =====")
		}
		if strings.HasPrefix(line, "===== CASE 2/12 ") {
			second = strings.TrimSuffix(strings.TrimPrefix(line, "===== CASE 2/12 "), " =====")
		}
	}
	if first == "" || second == "" {
		t.Fatal("test trace has no first case")
	}
	trace = strings.Replace(trace, "===== CASE 2/12 "+second+" =====", "===== CASE 2/12 "+first+" =====", 1)
	if _, err := CompareSmoke(strings.NewReader(trace), strings.NewReader(smokeTrace("PASSED", "natural", true))); err == nil || !strings.Contains(err.Error(), "duplicate case identity") {
		t.Fatalf("duplicate case accepted: %v", err)
	}
}

func TestCompareSmokeReportsIncompleteDenominator(t *testing.T) {
	left := smokeTrace("INCOMPLETE", "", true)
	right := smokeTrace("PASSED", "natural", true)
	report, err := CompareSmoke(strings.NewReader(left), strings.NewReader(right))
	if err != nil {
		t.Fatal(err)
	}
	if report.Pass() || report.Ran != 0 || report.Skipped != 12 {
		t.Fatalf("report=%+v, want a failed skipped denominator", report)
	}
}

func TestCompareSmokeFilesMissingIsUnmeasured(t *testing.T) {
	r := CompareSmokeFiles(filepath.Join(t.TempDir(), "left.trace"), "right.trace")
	if r.Status != "UNMEASURED" || !strings.Contains(r.Reason, "left trace unavailable") {
		t.Fatalf("receipt=%+v", r)
	}
}

func TestCompareSmokeFilesReportsPassAndMutation(t *testing.T) {
	dir := t.TempDir()
	leftPath, rightPath := filepath.Join(dir, "left.trace"), filepath.Join(dir, "right.trace")
	if err := os.WriteFile(leftPath, []byte(smokeTrace("PASSED", "natural", true)), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(rightPath, []byte(smokeTrace("PASSED", "natural", true)), 0o600); err != nil {
		t.Fatal(err)
	}
	if r := CompareSmokeFiles(leftPath, rightPath); r.Status != "PASS" || r.Cases != 12 {
		t.Fatalf("pass receipt=%+v", r)
	}
	if err := os.WriteFile(rightPath, []byte(smokeTrace("FAILED", "soft_forced", true)), 0o600); err != nil {
		t.Fatal(err)
	}
	if r := CompareSmokeFiles(leftPath, rightPath); r.Status != "FAIL" {
		t.Fatalf("mutation receipt=%+v", r)
	}
}

func TestSmokeReceiptJSONKeepsTraceSidesDistinct(t *testing.T) {
	receipt := SmokeReceipt{Status: "PASS", Left: "left.trace", Right: "right.trace"}
	var encoded bytes.Buffer
	if err := receipt.WriteJSON(&encoded); err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(encoded.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["left_trace"] != "left.trace" || got["right_trace"] != "right.trace" {
		t.Fatalf("trace paths = %#v, want distinct left_trace/right_trace", got)
	}
}

func TestCompareSmokeReceiptRetainsProvenance(t *testing.T) {
	dir := t.TempDir()
	leftPath, rightPath := filepath.Join(dir, "left.trace"), filepath.Join(dir, "right.trace")
	for _, path := range []string{leftPath, rightPath} {
		if err := os.WriteFile(path, []byte(smokeTrace("PASSED", "natural", true)), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	receipt := CompareSmokeFiles(leftPath, rightPath)
	if receipt.Status != "PASS" || receipt.LeftBuild == "" || receipt.RightBuild == "" ||
		receipt.ModelSHA256 == "" || receipt.Context != "4096" || receipt.Temperature != "0" || receipt.Think != "HIGH" {
		t.Fatalf("receipt lost provenance: %+v", receipt)
	}
}
