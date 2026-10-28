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

// arbiterRates lists the variants TestAutotuneSamplingRates repeats, each
// with extra environment for its child processes.
var arbiterRates = []struct {
	variant string
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
}

var arbiterFailure = regexp.MustCompile(`session \d+ \([^)]*\): .*`)

// TestAutotuneSamplingRates runs each arbiterRates row
// AUTOTUNE_ARBITER_RUNS times (default 5), every run a fresh process
// executing one TestAutotuneSamplingArbiter variant, and logs how many
// passed and the first failure of each failing run. It does not open a
// device itself. It is gated like TestAutotuneNativeQualification and
// fails only if a child neither passes nor fails cleanly.
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
	var table strings.Builder
	for _, row := range arbiterRates {
		label := row.variant
		if len(row.env) > 0 {
			label += " " + strings.Join(row.env, " ")
		}
		pass := 0
		var failures []string
		for run := range runs {
			cmd := exec.Command(os.Args[0], "-test.run=^TestAutotuneSamplingArbiter$/^"+row.variant+"$", "-test.v", "-test.count=1", "-test.timeout=10m")
			cmd.Env = append(os.Environ(), row.env...)
			out, err := cmd.CombinedOutput()
			switch {
			case err == nil && bytes.Contains(out, []byte("--- PASS: TestAutotuneSamplingArbiter/"+row.variant+" ")):
				pass++
			case bytes.Contains(out, []byte("--- FAIL: TestAutotuneSamplingArbiter/"+row.variant+" ")):
				why := "unrecognized failure"
				if m := arbiterFailure.Find(out); m != nil {
					why = string(m)
				}
				failures = append(failures, fmt.Sprintf("run %d: %s", run, why))
			default:
				t.Errorf("%s run %d: child neither passed nor failed (%v):\n%s", label, run, err, out)
			}
		}
		fmt.Fprintf(&table, "%-34s %d/%d pass\n", label, pass, runs)
		for _, f := range failures {
			fmt.Fprintf(&table, "    %s\n", f)
		}
	}
	t.Logf("PC-sampling arbiter pass rates:\n%s", table.String())
}
