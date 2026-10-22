package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRegradeCountedAdmission(t *testing.T) {
	body := strings.Repeat("reason", 22000) + "\nMODEL_OUTPUT_END\nAnswer: B"
	valid := fmt.Sprintf("# ds4-eval trace\n===== CASE 1/1 GPQA Diamond/recNu3MXkvWUzHZr9 =====\nstatus: PASSED\nexpected: B\nMODEL_OUTPUT_BEGIN bytes=%d\n%s\nMODEL_OUTPUT_END\n===== SUMMARY =====\n", len(body), body)
	for _, tc := range []struct {
		name, data string
		rc         int
	}{
		{"valid", valid, 0},
		{"wrong-length", strings.Replace(valid, fmt.Sprint("bytes=", len(body)), "bytes=10", 1), 1},
		{"truncated", valid[:len(valid)-20], 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "trace")
			if err := os.WriteFile(path, []byte(tc.data), 0600); err != nil {
				t.Fatal(err)
			}
			var out, stderr bytes.Buffer
			if rc := run([]string{"-regrade-trace", path}, &out, &stderr); rc != tc.rc {
				t.Fatalf("rc=%d want%d stdout=%s stderr=%s", rc, tc.rc, &out, &stderr)
			}
		})
	}
}

func TestRuntimeAdmission(t *testing.T) {
	for _, args := range [][]string{
		{"-trace", "out", "-suite", "hard-smoke"},
		{"-trace", "out", "-suite", "hard-smoke", "-model", "model", "-temp", "1"},
		{"-trace", "out", "-suite", "hard-smoke", "-model", "model", "-case-id", "2826"},
		{"-trace", "out", "-suite", "core", "-model", "model"},
	} {
		var out, stderr bytes.Buffer
		if got := run(args, &out, &stderr); got != 2 {
			t.Fatalf("args=%v rc=%d stderr=%s", args, got, stderr.String())
		}
	}
	var out, stderr bytes.Buffer
	if got := run([]string{"-trace", "out", "-suite", "hard-smoke", "-model", "model", "-corpus", "external", "-runner", "/missing/ds4-min"}, &out, &stderr); got != 1 {
		t.Fatalf("missing executable rc=%d", got)
	}
}

func TestRuntimeArtifactsForwarded(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runner")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nprintf '%s\\n' \"$@\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	args := []string{"-trace", "out", "-suite", "hard-smoke", "-model", "model", "-corpus", "external", "-runner", path, "-eval-artifacts-dir", "new-artifacts"}
	if got := run(args, &out, &stderr); got != 0 {
		t.Fatalf("rc=%d %s", got, &stderr)
	}
	if !strings.Contains(out.String(), "-eval-artifacts-dir\nnew-artifacts\n") {
		t.Fatal("artifact flag lost before runner")
	}
	if got := run([]string{"-list-cases", "-eval-artifacts-dir", "x"}, &out, &stderr); got != 2 {
		t.Fatal("nonruntime artifacts accepted")
	}
}

func TestRunListAndValidate(t *testing.T) {
	var out, errOut bytes.Buffer
	if got := run([]string{"-validate-cases"}, &out, &errOut); got != 0 {
		t.Fatalf("validate rc=%d stderr=%s", got, errOut.String())
	}
	if got := run([]string{"-list-cases", "-suite", "hard"}, &out, &errOut); got != 0 {
		t.Fatalf("list rc=%d stderr=%s", got, errOut.String())
	}
	if lines := strings.Count(out.String(), "\n"); lines != 51 {
		t.Fatalf("hard list lines=%d", lines)
	}
}

func TestRunFailsClosed(t *testing.T) {
	var out, errOut bytes.Buffer
	if got := run(nil, &out, &errOut); got != 2 {
		t.Fatalf("missing operation rc=%d", got)
	}
	if got := run([]string{"-list-cases", "-validate-cases"}, &out, &errOut); got != 2 {
		t.Fatalf("conflicting operations rc=%d", got)
	}
	if got := run([]string{"-self-test-extractors", "-validate-cases"}, &out, &errOut); got != 2 {
		t.Fatalf("self-test/validate conflict rc=%d", got)
	}
	if got := run([]string{"-g7-smoke", "missing-left:missing-right"}, &out, &errOut); got != 3 {
		t.Fatalf("missing G7 inputs rc=%d, stderr=%s", got, errOut.String())
	}
	if !strings.Contains(out.String(), `"status":"UNMEASURED"`) {
		t.Fatalf("missing G7 receipt=%q", out.String())
	}
}

func TestRunSelfTestExtractors(t *testing.T) {
	var out, errOut bytes.Buffer
	if got := run([]string{"-self-test-extractors"}, &out, &errOut); got != 0 {
		t.Fatalf("self-test rc=%d stderr=%s", got, errOut.String())
	}
	if got := out.String(); got != "ds4-eval: extractor self-test passed\n" {
		t.Fatalf("self-test output = %q", got)
	}
}

func TestRunCaseSelection(t *testing.T) {
	var out, errOut bytes.Buffer
	if got := run([]string{"-list-cases", "-source", "gpqa diamond", "-domain", "physics", "-questions", "2"}, &out, &errOut); got != 0 {
		t.Fatalf("filtered list rc=%d stderr=%s", got, errOut.String())
	}
	if lines := strings.Count(out.String(), "\n"); lines != 3 {
		t.Fatalf("filtered list lines=%d", lines)
	}
	if got := run([]string{"-list-cases", "-case-id", "missing"}, &out, &errOut); got != 1 {
		t.Fatalf("zero-match rc=%d stderr=%s", got, errOut.String())
	}
	if got := run([]string{"-list-cases", "-source", "GPQA Diamond", "-questions", "100"}, &out, &errOut); got != 1 {
		t.Fatalf("over-limit rc=%d stderr=%s", got, errOut.String())
	}
	if got := run([]string{"-regrade-trace", "trace", "-source", "GPQA Diamond"}, &out, &errOut); got != 2 {
		t.Fatalf("selection on trace rc=%d", got)
	}
}

func TestRunCompareSeeded(t *testing.T) {
	dir := t.TempDir()
	trace := `{"metadata":{"build":"oracle","model_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","context":4096,"temperature":1,"seed":12345,"mtp_disabled":true,"artifact_only":true},"steps":[{"step":0,"token":17,"logits_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}`
	left := filepath.Join(dir, "left.json")
	right := filepath.Join(dir, "right.json")
	if err := os.WriteFile(left, []byte(trace), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(right, []byte(trace), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if got := run([]string{"-compare-seeded", left + ":" + right}, &out, &errOut); got != 0 {
		t.Fatalf("seeded compare rc=%d stderr=%s", got, errOut.String())
	}
	if !strings.Contains(out.String(), `"logit_digest_matches": 1`) {
		t.Fatalf("seeded report=%s", out.String())
	}
	if got := run([]string{"-compare-seeded", left + ":missing"}, &out, &errOut); got != 2 {
		t.Fatalf("missing seeded trace rc=%d", got)
	}
	if err := os.WriteFile(right, []byte(trace+" {}"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := run([]string{"-compare-seeded", left + ":" + right}, &out, &errOut); got != 2 {
		t.Fatalf("trailing seeded trace rc=%d", got)
	}
}
