package main

import (
	"context"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

// TestCaptureOfFailedTargetIsNotSuccessful runs the real producer against a
// stub CUPTI in a target that finalizes its journal and then dies. The
// journal is complete and the records are real, so every collection check
// passes; only the target's own status says the run did not finish. A
// capture of such a run was reported as outcome collected with
// capture_success true, which is how a SIGSEGV under CUDA graphs read as a
// good capture on dgx.
func TestCaptureOfFailedTargetIsNotSuccessful(t *testing.T) {
	target := stubAgentTarget(t)
	for _, tt := range []struct {
		name   string
		status string
		signal bool
		exit   int
		why    string
	}{
		{name: "sigsegv", status: "0", signal: true, exit: -1, why: "target died by SIGSEGV"},
		{name: "exit", status: "2", exit: 2, why: "target exited with status 2"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result, err := runCapture(context.Background(), captureOptions{
				command:      []string{target, "finalize-crash", "1", "1", "8", "0", tt.status},
				Env:          []string{"PATH=" + os.Getenv("PATH")},
				NoInject:     true,
				Stdout:       io.Discard,
				DrainTimeout: time.Second,
			})
			if err != nil {
				t.Fatal(err)
			}
			if (result.TargetSignal != "") != tt.signal || result.TargetExitCode != tt.exit {
				t.Fatalf("target signal %q exit %d, want signal %v exit %d", result.TargetSignal, result.TargetExitCode, tt.signal, tt.exit)
			}
			// The records must still be there: the target failing is not the
			// collection failing, and the fix must not throw them away.
			if !result.CollectorFinalized || result.ActivityRecords == 0 {
				t.Fatalf("collector_finalized=%v activity_records=%d, want a finalized journal with records (outcome %s: %s)", result.CollectorFinalized, result.ActivityRecords, result.Outcome, result.CaptureError)
			}
			if result.CaptureSuccess {
				t.Errorf("capture_success = true for a target that did not finish (outcome %s)", result.Outcome)
			}
			if result.Outcome != captureOutcomeTargetFailed {
				t.Errorf("outcome = %q, want %q", result.Outcome, captureOutcomeTargetFailed)
			}
			if !strings.Contains(result.CaptureError, tt.why) {
				t.Errorf("capture_error = %q, want it to say %q", result.CaptureError, tt.why)
			}
		})
	}
}

func TestTargetFailureMessage(t *testing.T) {
	memsetsOn := []byte(`{"kind":"capability","memsets":true}`)
	memsetsOff := []byte(`{"kind":"capability","memsets":false}`)
	goTarget := &targetEnvironment{Executable: &executableIdentity{Go: &goBuildIdentity{GoVersion: "go1.27"}}}
	for _, tt := range []struct {
		name   string
		result captureResult
		want   string
	}{
		{"segv memsets on", captureResult{TargetSignal: "segmentation fault", Capability: memsetsOn},
			"target died by SIGSEGV; if it uses CUDA graphs, try --memsets=false"},
		{"segv memsets off", captureResult{TargetSignal: "segmentation fault", Capability: memsetsOff},
			"target died by SIGSEGV"},
		{"segv no capability", captureResult{TargetSignal: "segmentation fault"},
			"target died by SIGSEGV"},
		{"sigterm", captureResult{TargetSignal: "terminated", Capability: memsetsOn},
			"target died by SIGTERM"},
		{"exit", captureResult{TargetExitCode: 7, Capability: memsetsOn},
			"target exited with status 7"},
		{"go exit 2", captureResult{TargetExitCode: 2, Capability: memsetsOn, Target: goTarget},
			"target exited with status 2, which is also how a Go program reports a fatal signal such as SIGSEGV; if it uses CUDA graphs, try --memsets=false"},
		{"go exit 2 memsets off", captureResult{TargetExitCode: 2, Capability: memsetsOff, Target: goTarget},
			"target exited with status 2"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := targetFailureMessage(tt.result); got != tt.want {
				t.Errorf("targetFailureMessage = %q, want %q", got, tt.want)
			}
		})
	}
}
