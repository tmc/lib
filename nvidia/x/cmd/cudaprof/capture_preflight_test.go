package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

// failedCaptureArtifact writes the artifact a failed capture leaves behind:
// a real pprof carrying capture-status.json with capture_success false.
func failedCaptureArtifact(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "failed.pprof")
	result := captureResult{
		Protocol:     captureProtocolVersion,
		Session:      "00112233445566778899aabbccddeeff",
		State:        captureStateFinalized,
		Outcome:      captureOutcomeTransport,
		CaptureError: "read handshake: unexpected EOF",
	}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}.effective()); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestCaptureFailureIsDetectable is the §6.2 control's instrument check: the
// failure has to be readable back out of the artifact before anything can
// refuse on it.
func TestCaptureFailureIsDetectable(t *testing.T) {
	path := failedCaptureArtifact(t)
	cap, err := readCapturePathLimits(path, cudacapture.Limits{})
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	reason, failed := capturefiles.Failure(cap)
	if !failed {
		t.Fatal("Failure did not detect capture_success:false")
	}
	if !strings.Contains(reason, captureOutcomeTransport) {
		t.Errorf("reason does not name the transport outcome: %q", reason)
	}
	if !strings.Contains(reason, "read handshake") {
		t.Errorf("reason does not name the failing stage: %q", reason)
	}
}

// TestSucceededCaptureIsNotRefused is the negative half: an ordinary capture
// must not be refused. Without it the preflight could pass by refusing
// everything.
func TestSucceededCaptureIsNotRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ok.pprof")
	result := captureResult{
		Protocol:        captureProtocolVersion,
		Session:         "00112233445566778899aabbccddeeff",
		State:           captureStateFinalized,
		Outcome:         captureOutcomeObserved,
		CaptureSuccess:  true,
		RunCompleteness: runComplete,
	}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}.effective()); err != nil {
		t.Fatal(err)
	}
	cap, err := readCapturePathLimits(path, cudacapture.Limits{})
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	if reason, failed := capturefiles.Failure(cap); failed {
		t.Errorf("a successful capture was reported as failed: %q", reason)
	}
}

// TestPProfPathRefusesFailedCapture is the §6.2 control proper. A capture that
// records capture_success:false carries no measurement, and opening it in
// pprof prints an empty profile and exits 0 -- which invites exactly the
// mistake of analysing it. The read path must fail closed.
func TestPProfPathRefusesFailedCapture(t *testing.T) {
	path := failedCaptureArtifact(t)

	cap, err := readCapturePathLimits(path, cudacapture.Limits{})
	if err != nil {
		t.Fatalf("read capture: %v", err)
	}
	err = preflightCapture(cap)
	if err == nil {
		t.Fatal("pprof path accepted a capture whose status records capture_success:false")
	}
	msg := err.Error()
	if !strings.Contains(msg, captureOutcomeTransport) {
		t.Errorf("refusal does not name the transport: %q", msg)
	}
	if !strings.Contains(msg, "read handshake") {
		t.Errorf("refusal does not name the failing stage: %q", msg)
	}
	if !strings.Contains(msg, "advise") {
		t.Errorf("refusal does not point at a command that can still inspect the failure: %q", msg)
	}
	if _, statErr := os.Stat(path); statErr != nil {
		t.Errorf("refusal removed the artifact: %v", statErr)
	}
}
