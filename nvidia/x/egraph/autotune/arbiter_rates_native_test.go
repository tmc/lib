package autotune_test

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// arbiterRates lists the rows TestAutotuneSamplingRates repeats: a
// TestAutotuneSamplingArbiter variant, or with test set, a whole native
// test, each with extra environment for its child processes.
var arbiterRates = []struct {
	variant string
	test    string
	env     []string
}{
	{variant: "bare"},
	{variant: "replica"},
	{variant: "module_once"},
	{variant: "collector"},
	{variant: "collector", env: []string{"GOGC=off"}},
	{variant: "replica", env: []string{"GOGC=off"}},
	{variant: "replica_keep_sessions"},
	{variant: "replica_sync"},
	{variant: "load_all_first"},
	{variant: "load_then_sync"},
	{variant: "collector_preload"},
	{variant: "qualification", test: "TestAutotuneNativeQualification"},
}

var (
	arbiterFailure = regexp.MustCompile(`session \d+ \([^)]*\): .*`)
	childFailure   = regexp.MustCompile(`(?m)^\s+\S+_test\.go:\d+: .*`)
)

// failureKind classifies a failed child's output: "enable" for CUPTI
// rejecting an enable, "loss" for the final-buffer loss check at Close,
// and "other" for anything else.
func failureKind(out []byte) string {
	switch {
	case bytes.Contains(out, []byte("CUPTI_ERROR_INVALID_OPERATION")):
		return "enable"
	case bytes.Contains(out, []byte("loss or remaining PCs")):
		return "loss"
	}
	return "other"
}

// TestAutotuneSamplingRates runs each arbiterRates row
// AUTOTUNE_ARBITER_RUNS times (default 5), every run a fresh process,
// and logs how many passed, how many failed by kind (see failureKind) and
// the first failure of each failing run. It does not open a device itself.
// AUTOTUNE_ARBITER_ROWS, a comma-separated list of row labels such as
// "load_all_first,collector GOGC=off,qualification", restricts the rows.
// It is gated like TestAutotuneNativeQualification and fails only if a
// child neither passes nor fails cleanly.
func TestAutotuneSamplingRates(t *testing.T) {
	if os.Getenv(autotuneNativeEnv) != "1" {
		t.Skip("native arbiter pending: CUDA_EGRAPH_AUTOTUNE_NATIVE=1 required")
	}
	if err := checkAutotuneNativeGate(os.Getenv); err != nil {
		t.Fatalf("%v: run under gpu-lock", err)
	}
	runs := 5
	if s := os.Getenv("AUTOTUNE_ARBITER_RUNS"); s != "" {
		n, err := strconv.Atoi(s)
		if err != nil || n < 1 {
			t.Fatalf("AUTOTUNE_ARBITER_RUNS=%q", s)
		}
		runs = n
	}
	only := make(map[string]bool)
	if s := os.Getenv("AUTOTUNE_ARBITER_ROWS"); s != "" {
		for _, r := range strings.Split(s, ",") {
			only[strings.TrimSpace(r)] = true
		}
	}
	var table strings.Builder
	for _, row := range arbiterRates {
		label := row.variant
		if len(row.env) > 0 {
			label += " " + strings.Join(row.env, " ")
		}
		if len(only) > 0 && !only[label] {
			continue
		}
		pattern, name, timeout := "^TestAutotuneSamplingArbiter$/^"+row.variant+"$", "TestAutotuneSamplingArbiter/"+row.variant, "10m"
		if row.test != "" {
			pattern, name, timeout = "^"+row.test+"$", row.test, "30m"
		}
		pass := 0
		kinds := make(map[string]int)
		var failures []string
		for run := range runs {
			cmd := exec.Command(os.Args[0], "-test.run="+pattern, "-test.v", "-test.count=1", "-test.timeout="+timeout)
			cmd.Env = append(os.Environ(), row.env...)
			out, err := cmd.CombinedOutput()
			switch {
			case err == nil && bytes.Contains(out, []byte("--- PASS: "+name+" ")):
				pass++
			case bytes.Contains(out, []byte("--- FAIL: "+name+" ")):
				kind := failureKind(out)
				kinds[kind]++
				why := "unrecognized failure"
				if m := arbiterFailure.Find(out); m != nil {
					why = string(m)
				} else if m := childFailure.Find(out); m != nil {
					why = strings.TrimSpace(string(m))
				}
				failures = append(failures, fmt.Sprintf("run %d [%s]: %s", run, kind, why))
			default:
				t.Errorf("%s run %d: child neither passed nor failed (%v):\n%s", label, run, err, out)
			}
		}
		fmt.Fprintf(&table, "%-34s %d/%d pass, fail enable=%d loss=%d other=%d\n", label, pass, runs, kinds["enable"], kinds["loss"], kinds["other"])
		for _, f := range failures {
			fmt.Fprintf(&table, "    %s\n", f)
		}
	}
	t.Logf("PC-sampling arbiter pass rates:\n%s", table.String())
}
