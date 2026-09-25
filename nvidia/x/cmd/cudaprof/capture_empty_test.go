package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// testExecutables returns the running test binary, which is a Go binary, and
// sh, which is not.
func testExecutables(t *testing.T) (goExe, cExe string) {
	t.Helper()
	goExe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cExe, err = exec.LookPath("sh")
	if err != nil {
		t.Skip("no sh to stand in for a non-Go target")
	}
	return goExe, cExe
}

func TestGoExitRisk(t *testing.T) {
	goExe, cExe := testExecutables(t)
	tests := []struct {
		name     string
		id       *executableIdentity
		finalize string
		want     bool
	}{
		{"go binary, finalize unset", identifyExecutable(goExe), "", true},
		{"go binary, finalize exit", identifyExecutable(goExe), finalizeExit, true},
		{"go binary, finalize profiler-stop", identifyExecutable(goExe), finalizeProfilerStop, false},
		{"go binary, finalize exit-group", identifyExecutable(goExe), finalizeExitGroup, false},
		{"c binary, finalize unset", identifyExecutable(cExe), "", false},
		{"c binary, finalize profiler-stop", identifyExecutable(cExe), finalizeProfilerStop, false},
		{"unreadable binary", identifyExecutable(filepath.Join(t.TempDir(), "missing")), "", false},
		{"no identity", nil, "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := goExitRisk(tt.id, tt.finalize); got != tt.want {
				t.Errorf("goExitRisk = %v, want %v (identity %+v)", got, tt.want, tt.id)
			}
		})
	}
}

// TestObserveGoExit covers what a capture reports once the target has run.
// Only a Go target whose journal ends unfinalized with no activity is the
// Go exit; a flushed journal has its own diagnosis, and a finalized one was
// published by some other route.
func TestObserveGoExit(t *testing.T) {
	goExe, _ := testExecutables(t)
	goID := identifyExecutable(goExe)
	bootstrapOnly := func(t *testing.T, dir string, sid [16]byte) {
		appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		})
	}
	finalized := func(state string) func(t *testing.T, dir string, sid [16]byte) {
		return func(t *testing.T, dir string, sid [16]byte) {
			records := []wire.Record{
				{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
				{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			}
			if state == "collected" {
				records = append(records, wire.Record{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)})
			}
			records = append(records, wire.Record{Kind: recordFinalized, Payload: []byte(`{"state":"` + state + `"}`)})
			appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, records)
		}
	}
	tests := []struct {
		name        string
		goTarget    *executableIdentity
		journal     func(t *testing.T, dir string, sid [16]byte)
		wantOutcome string
		wantSuccess bool
		wantError   string
	}{
		{"go target exits without finalizing", goID, bootstrapOnly, captureOutcomeGoExit, false, "--finalize=profiler-stop"},
		{"non-go target exits without finalizing", nil, bootstrapOnly, captureOutcomeNotFinalized, false, "finalized record"},
		{"go target flushed records", goID, flushedTargetJournal, captureOutcomeNotFinalized, false, "--recover-unfinalized"},
		{"go target finalized its records", goID, finalized("collected"), captureOutcomeCollected, true, ""},
		{"go target finalized with no cuda", goID, finalized("no_cuda_observed"), captureOutcomeObserved, false, "no activity records"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := targetSession(t, tt.journal)
			got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, goTarget: tt.goTarget})
			if got.Outcome != tt.wantOutcome || got.CaptureSuccess != tt.wantSuccess {
				t.Fatalf("outcome %q success %v, want %q %v; error %q", got.Outcome, got.CaptureSuccess, tt.wantOutcome, tt.wantSuccess, got.CaptureError)
			}
			if !strings.Contains(got.CaptureError, tt.wantError) {
				t.Errorf("capture error %q does not contain %q", got.CaptureError, tt.wantError)
			}
			if got.ActivityRecords == 0 && got.RunCompleteness == runComplete {
				t.Errorf("empty capture reports run_completeness %q", got.RunCompleteness)
			}
		})
	}
}

func TestRefuseEmpty(t *testing.T) {
	tests := []struct {
		name        string
		in          captureResult
		wantSuccess bool
		wantError   string
	}{
		{"collected", captureResult{Outcome: captureOutcomeCollected, CaptureSuccess: true, ActivityRecords: 3}, true, ""},
		{"no cuda observed", captureResult{Outcome: captureOutcomeObserved, CaptureSuccess: true}, false, "the capture holds no activity records: the agent finalized and reports that the target never used CUDA"},
		{"other empty success", captureResult{Outcome: "some-outcome", CaptureSuccess: true}, false, "the capture holds no activity records: outcome some-outcome"},
		{"failure keeps its error", captureResult{Outcome: captureOutcomeTransport, CaptureError: "read handshake"}, false, "read handshake"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := refuseEmpty(tt.in)
			if got.CaptureSuccess != tt.wantSuccess || got.CaptureError != tt.wantError {
				t.Errorf("refuseEmpty = success %v error %q, want %v %q", got.CaptureSuccess, got.CaptureError, tt.wantSuccess, tt.wantError)
			}
			if got.Outcome != tt.in.Outcome {
				t.Errorf("outcome = %q, want %q unchanged", got.Outcome, tt.in.Outcome)
			}
		})
	}
}

// TestRunCaptureWarnsGoTarget checks the warning printed before the target
// runs. The agent is loaded through CUDA_INJECTION64_PATH, which a target that
// never initializes CUDA does not load, so the file need not be a library.
func TestRunCaptureWarnsGoTarget(t *testing.T) {
	goExe, cExe := testExecutables(t)
	agent := filepath.Join(t.TempDir(), "agent.so")
	if err := os.WriteFile(agent, nil, 0600); err != nil {
		t.Fatal(err)
	}
	goCommand := []string{goExe, "-test.run=TestCaptureHelper"}
	tests := []struct {
		name     string
		command  []string
		finalize string
		env      []string
		noInject bool
		warn     bool
	}{
		{"go target without flag", goCommand, "", nil, false, true},
		{"go target with flag", goCommand, finalizeProfilerStop, nil, false, false},
		{"go target with flag in environment", goCommand, "", []string{"CUDAPROF_FINALIZE=" + finalizeProfilerStop}, false, false},
		{"go target with finalize exit", goCommand, finalizeExit, nil, false, true},
		{"go target uninstrumented", goCommand, "", nil, true, false},
		{"c target without flag", []string{cExe, "-c", "true"}, "", nil, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var log, out bytes.Buffer
			o := captureOptions{
				command:      tt.command,
				Env:          append([]string{"PATH=" + os.Getenv("PATH")}, tt.env...),
				Finalize:     tt.finalize,
				Stdout:       &out,
				Stderr:       &out,
				Log:          &log,
				DrainTimeout: time.Second,
			}
			if tt.noInject {
				o.NoInject = true
			} else {
				o.Inject, o.InjectMode = agent, injectModeCUDA
			}
			if _, err := runCapture(context.Background(), o); err != nil {
				t.Fatal(err)
			}
			warned := strings.Contains(log.String(), "--finalize="+finalizeProfilerStop)
			if warned != tt.warn {
				t.Errorf("warned = %v, want %v; log %q", warned, tt.warn, log.String())
			}
		})
	}
}
