package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// captureWithRecords writes a real capture artifact holding n activity
// records, which is the number the overwrite predicate reads.
func captureWithRecords(t *testing.T, path string, n uint64) {
	t.Helper()
	result := captureResult{
		Protocol:        captureProtocolVersion,
		Session:         "00112233445566778899aabbccddeeff",
		State:           captureStateFinalized,
		Outcome:         captureOutcomeObserved,
		CaptureSuccess:  true,
		ActivityRecords: n,
	}
	if err := writeCaptureStatus(path, result, captureArtifactBudget{}.effective()); err != nil {
		t.Fatal(err)
	}
}

func captureArgsFor(out string) []string {
	return []string{"--output", out, "--", "/bin/true"}
}

// TestEmptyOutputDoesNotBlockItsOwnRetry is 10.6(iii). The file a failed run
// leaves behind holds no measurement, and refusing to overwrite it puts the
// refusal on the retry rather than on anything worth protecting.
func TestEmptyOutputDoesNotBlockItsOwnRetry(t *testing.T) {
	out := filepath.Join(t.TempDir(), "run.pprof")
	captureWithRecords(t, out, 0)
	o, err := parseCaptureArgs(captureArgsFor(out))
	if err != nil {
		t.Fatalf("empty capture blocked its retry: %v", err)
	}
	if !o.ReplaceEmptyOutput {
		t.Fatal("parse allowed the output but did not mark it for replacement, so publication would still fail on the existing file")
	}
}

// TestOutputHoldingRecordsIsStillProtected is the half the predicate exists to
// keep. Lifting the refusal for an empty file must not lift it for a capture
// someone paid a run for.
func TestOutputHoldingRecordsIsStillProtected(t *testing.T) {
	out := filepath.Join(t.TempDir(), "run.pprof")
	captureWithRecords(t, out, 9711)
	o, err := parseCaptureArgs(captureArgsFor(out))
	if err == nil {
		t.Fatal("overwrote a capture holding 9711 activity records")
	}
	if o.ReplaceEmptyOutput {
		t.Error("a protected output was marked for replacement")
	}
	if !strings.Contains(err.Error(), "9711") {
		t.Errorf("refusal does not state the count it refused on: %v", err)
	}
}

// TestUnreadableOutputIsProtected fixes the direction of the unknown case. A
// file whose record count cannot be read is not a file cudaprof may delete.
func TestUnreadableOutputIsProtected(t *testing.T) {
	out := filepath.Join(t.TempDir(), "run.pprof")
	if err := os.WriteFile(out, []byte("not a capture"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := parseCaptureArgs(captureArgsFor(out)); err == nil {
		t.Fatal("deleted a file whose record count could not be read")
	}
}

// TestAbsentOutputIsNotMarkedForReplacement is the negative control: the
// ordinary case must not take the removal path.
func TestAbsentOutputIsNotMarkedForReplacement(t *testing.T) {
	out := filepath.Join(t.TempDir(), "fresh.pprof")
	o, err := parseCaptureArgs(captureArgsFor(out))
	if err != nil {
		t.Fatal(err)
	}
	if o.ReplaceEmptyOutput {
		t.Error("a capture to a path that does not exist was marked for replacement")
	}
}

// TestOutputHoldsWorkReadsTheCountNotTheOutcome states the discriminator
// directly: a run that reported failure but retained records is protected, and
// a run that reported success but retained none is not. Deciding on the
// outcome token instead would invert both.
func TestOutputHoldsWorkReadsTheCountNotTheOutcome(t *testing.T) {
	dir := t.TempDir()

	failedButRetained := filepath.Join(dir, "failed.pprof")
	if err := writeCaptureStatus(failedButRetained, captureResult{
		Protocol:        captureProtocolVersion,
		Session:         "00112233445566778899aabbccddeeff",
		State:           captureStateFinalized,
		Outcome:         captureOutcomeTransport,
		CaptureError:    "read handshake: unexpected EOF",
		ActivityRecords: 412,
	}, captureArtifactBudget{}.effective()); err != nil {
		t.Fatal(err)
	}
	if _, holds := outputHoldsWork(failedButRetained); !holds {
		t.Error("a failed capture that retained 412 records was treated as empty")
	}

	succeededEmpty := filepath.Join(dir, "empty.pprof")
	captureWithRecords(t, succeededEmpty, 0)
	if _, holds := outputHoldsWork(succeededEmpty); holds {
		t.Error("a successful capture that retained nothing was treated as work to protect")
	}
}
