package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const input = "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,context_total_bytes,context_raw_bytes,context_compressed_bytes,context_scratch_bytes,peak_rss_bytes\n2048,2048,800,8,18,50,7,19,1,2,3,4,5,6\n4096,2048,780,8,17,52,7,18,2,3,4,5,6,7\n"

func TestRunCanonicalizesFrontier(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runWithIO([]string{"-frontier"}, strings.NewReader(input), &out, &errOut); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	if got := strings.Count(out.String(), "\n"); got != 3 {
		t.Fatalf("lines=%d output=%q", got, out.String())
	}
}

func TestRunSummary(t *testing.T) {
	var out, errOut bytes.Buffer
	if rc := runWithIO([]string{"-summary"}, strings.NewReader(input), &out, &errOut); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	want := `{"samples":2,"prefill_tps":{"min":780,"median":790,"max":800},"gen_tps":{"min":17,"median":17.5,"max":18},"gen_first_ms":{"min":50,"median":51,"max":52},"gen_steady_tps":{"min":18,"median":18.5,"max":19}}` + "\n"
	if out.String() != want {
		t.Fatalf("summary = %q, want %q", out.String(), want)
	}
}

func TestRunComparison(t *testing.T) {
	dir := t.TempDir()
	baseName := filepath.Join(dir, "base.csv")
	targetName := filepath.Join(dir, "target.csv")
	if err := os.WriteFile(baseName, []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	target := strings.Replace(input, "800,8,18,50,7,19", "820,8,20,49,7,21", 1)
	target = strings.Replace(target, "780,8,17,52,7,18", "790,8,16,53,7,17", 1)
	if err := os.WriteFile(targetName, []byte(target), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if rc := runWithIO([]string{"-compare-base", baseName, "-compare-target", targetName}, strings.NewReader(""), &out, &errOut); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var got struct {
		Samples int `json:"samples"`
		Prefill struct {
			Delta struct {
				Median float64 `json:"median"`
			} `json:"delta"`
		} `json:"prefill_tps"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("comparison JSON: %v (%q)", err, out.String())
	}
	if got.Samples != 2 || got.Prefill.Delta.Median != 15 {
		t.Fatalf("comparison = %+v", got)
	}
}

func TestRunFrontierComparisonRejectsConflictingModes(t *testing.T) {
	dir := t.TempDir()
	base, target := filepath.Join(dir, "base.jsonl"), filepath.Join(dir, "target.jsonl")
	data := `{"source":"ds4","model":"model","backend":"cuda","quality":"fast","quant_bits":8,"prompt_tokens":32,"frontier_tokens":64,"ctx":128,"vocab":2,"argmax_id":0,"argmax_logit":2,"logits":[2,1]}` + "\n"
	if err := os.WriteFile(base, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{
		{"-compare-frontier-base", base, "-compare-frontier-target", target, "-frontier"},
		{"-compare-frontier-base", base, "-compare-frontier-target", target, "-summary"},
		{"-compare-frontier-base", base, "-compare-frontier-target", target, "-input", base},
	} {
		if rc := run(args, &bytes.Buffer{}, &bytes.Buffer{}); rc != 2 {
			t.Fatalf("args %v: rc=%d, want 2", args, rc)
		}
	}
}

func TestRunComparisonRejectsIncompleteOrConflictingFlags(t *testing.T) {
	dir := t.TempDir()
	baseName := filepath.Join(dir, "base.csv")
	if err := os.WriteFile(baseName, []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "missing target", args: []string{"-compare-base", baseName}, want: "must be used together"},
		{name: "summary conflict", args: []string{"-compare-base", baseName, "-compare-target", baseName, "-summary"}, want: "cannot be combined"},
		{name: "missing target", args: []string{"-compare-base", baseName, "-compare-target", filepath.Join(dir, "missing.csv")}, want: "open comparison input"},
		{name: "input conflict", args: []string{"-compare-base", baseName, "-compare-target", baseName, "-input", baseName}, want: "cannot be combined"},
		{name: "stdin target", args: []string{"-compare-base", baseName, "-compare-target", "-"}, want: "must be a file"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var out, errOut bytes.Buffer
			if rc := runWithIO(test.args, strings.NewReader(""), &out, &errOut); rc == 0 || !strings.Contains(errOut.String(), test.want) {
				t.Fatalf("rc=%d stderr=%q", rc, errOut.String())
			}
		})
	}
}

func TestRunComparisonModesAreMutuallyExclusive(t *testing.T) {
	args := []string{
		"-compare-base", "base.csv",
		"-compare-target", "target.csv",
		"-compare-frontier-base", "base.jsonl",
		"-compare-frontier-target", "target.jsonl",
	}
	var stderr bytes.Buffer
	if rc := runWithIO(args, strings.NewReader(""), &bytes.Buffer{}, &stderr); rc != 2 {
		t.Fatalf("rc=%d stderr=%q", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "cannot be combined") {
		t.Fatalf("stderr=%q", stderr.String())
	}
}

func TestRunFrontierComparison(t *testing.T) {
	dir := t.TempDir()
	baseName := filepath.Join(dir, "base.jsonl")
	targetName := filepath.Join(dir, "target.jsonl")
	base := `{"source":"ds4-bench","model":"m","backend":"cuda","quality":"fast","quant_bits":8,"prompt_tokens":32,"frontier_tokens":2048,"ctx":2048,"vocab":2,"argmax_id":0,"argmax_logit":2,"logits":[2,1]}` + "\n"
	target := `{"source":"ds4-min","model":"m","backend":"cuda-go","quality":"fast","quant_bits":8,"prompt_tokens":32,"frontier_tokens":2048,"ctx":2048,"vocab":2,"argmax_id":0,"argmax_logit":2.00001,"logits":[2.00001,1]}` + "\n"
	if err := os.WriteFile(baseName, []byte(base), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(targetName, []byte(target), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	args := []string{"-compare-frontier-base", baseName, "-compare-frontier-target", targetName}
	if rc := runWithIO(args, strings.NewReader(""), &out, &errOut); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var got struct {
		Compared int `json:"compared"`
		Pass     bool
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("comparison JSON: %v (%q)", err, out.String())
	}
	if got.Compared != 1 {
		t.Fatalf("comparison = %+v", got)
	}
}
