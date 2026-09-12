package main

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/journal"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// recoverySession builds a capture directory whose target generation finalizes
// cleanly and whose single worker generation is written by writeWorker. It
// reproduces the shape a force-killed serving worker leaves behind: the parent
// exits normally, the worker never runs its finalization path.
func recoverySession(t *testing.T, writeWorker func(t *testing.T, dir string, sid [16]byte)) string {
	t.Helper()
	const session = "00112233445566778899aabbccddeeff"
	dir := t.TempDir()
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)

	writeHandshake(t, dir, session, 41, 2)
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"initialized_no_activity"}`)},
	})

	workerDir := filepath.Join(dir, "process-42-3")
	if err := os.MkdirAll(workerDir, 0700); err != nil {
		t.Fatal(err)
	}
	writeHandshake(t, workerDir, session, 42, 3)
	writeWorker(t, workerDir, sid)
	return dir
}

func writeHandshake(t *testing.T, dir, session string, process, generation uint64) {
	t.Helper()
	b, err := json.Marshal(captureHandshake{Protocol: 1, Session: session, Process: process, Generation: generation, TransportReady: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func appendRecords(t *testing.T, path string, sid [16]byte, process, generation uint64, records []wire.Record) {
	t.Helper()
	j, err := journal.Create(path, sid, journal.Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i := range records {
		records[i].Identity = wire.Identity{Session: sid, Process: process, Generation: generation}
		records[i].Sequence = uint64(i + 1)
		if err := j.Append(records[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
}

// unfinalizedWorker writes two decodable activity records and stops, exactly as
// a SIGKILLed worker does: no finalized record, nothing else missing.
func unfinalizedWorker(t *testing.T, dir string, sid [16]byte) {
	t.Helper()
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 42, 3, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"memcpy","valid":true,"start":30,"end":40,"bytes":64}`)},
	})
}

// TestRecoverUnfinalizedDefaultsOff pins that the recovery is explicit: with
// the zero options an unfinalized worker contributes nothing at all, exactly as
// before the recovery path existed.
func TestRecoverUnfinalizedDefaultsOff(t *testing.T) {
	dir := recoverySession(t, unfinalizedWorker)
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true})
	if got.CaptureSuccess || got.Outcome != "initialized-no-activity" {
		t.Fatalf("default outcome = %+v", got)
	}
	if got.ActivityRecords != 0 || len(got.Activities) != 0 || got.CopyBytes != 0 || got.ActivityNanos != 0 {
		t.Fatalf("default admitted worker records: %+v", got)
	}
	if got.RecoveredGenerations != 0 || got.RecoveredActivityRecords != 0 {
		t.Fatalf("default reported recovery: %+v", got)
	}
	if len(got.WorkerGenerations) != 1 || got.WorkerGenerations[0].Disposition != "incomplete" || got.WorkerGenerations[0].Finalized {
		t.Fatalf("default worker manifest = %+v", got.WorkerGenerations)
	}
}

// TestRecoverUnfinalizedPartialOnly is the F1 control: the readable worker is
// preserved as a partial and never promoted.
func TestRecoverUnfinalizedPartialOnly(t *testing.T) {
	dir := recoverySession(t, unfinalizedWorker)
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.CaptureSuccess {
		t.Fatalf("recovery promoted capture: %+v", got)
	}
	if got.Outcome != captureOutcomePartialRecovered {
		t.Fatalf("outcome = %q, want %q", got.Outcome, captureOutcomePartialRecovered)
	}
	if got.CaptureError == "" {
		t.Fatal("partial recovery left no capture error")
	}
	if got.ActivityRecords != 2 || len(got.Activities) != 2 {
		t.Fatalf("recovered activity = %d records %+v", got.ActivityRecords, got.Activities)
	}
	if got.Activities[0].Process != 42 || got.Activities[1].Process != 42 {
		t.Fatalf("recovered activity lost worker identity: %+v", got.Activities)
	}
	if got.CopyBytes != 64 || got.ActivityNanos != 10 {
		t.Fatalf("recovered metrics = bytes %d nanos %d", got.CopyBytes, got.ActivityNanos)
	}
	if got.RecoveredGenerations != 1 || got.RecoveredActivityRecords != 2 {
		t.Fatalf("recovery counters = %d/%d", got.RecoveredGenerations, got.RecoveredActivityRecords)
	}
	w := got.WorkerGenerations
	if len(w) != 1 || w[0].Process != 42 || w[0].Generation != 3 {
		t.Fatalf("worker manifest = %+v", w)
	}
	if w[0].Finalized || w[0].Disposition != captureWorkerUnfinalizedPartial || w[0].Failure == "" {
		t.Fatalf("worker disposition = %+v", w[0])
	}
	if w[0].ActivityRecords != 2 {
		t.Fatalf("worker activity count = %d", w[0].ActivityRecords)
	}
}

// TestRecoverUnfinalizedKeepsFinalizedPositive pins that enabling recovery does
// not change a generation that did finalize.
func TestRecoverUnfinalizedKeepsFinalizedPositive(t *testing.T) {
	dir := recoverySession(t, func(t *testing.T, dir string, sid [16]byte) {
		appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 42, 3, []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
			{Kind: recordFinalized, Payload: []byte(`{"state":"collected"}`)},
		})
	})
	for _, recover := range []bool{false, true} {
		got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: recover})
		if !got.CaptureSuccess || got.Outcome != captureOutcomeCollected {
			t.Fatalf("recover=%v finalized worker = %+v", recover, got)
		}
		if got.ActivityRecords != 1 || got.RecoveredGenerations != 0 {
			t.Fatalf("recover=%v counts = %d records, %d recovered", recover, got.ActivityRecords, got.RecoveredGenerations)
		}
		if w := got.WorkerGenerations; len(w) != 1 || !w[0].Finalized || w[0].Disposition != "finalized" {
			t.Fatalf("recover=%v worker manifest = %+v", recover, w)
		}
	}
}

// TestRecoverUnfinalizedDemotesCollectedTarget pins the strongest form of "not
// complete": even when the target itself collected successfully, one recovered
// worker demotes the whole capture.
func TestRecoverUnfinalizedDemotesCollectedTarget(t *testing.T) {
	const session = "00112233445566778899aabbccddeeff"
	dir := t.TempDir()
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	writeHandshake(t, dir, session, 41, 2)
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected"}`)},
	})
	workerDir := filepath.Join(dir, "process-42-3")
	if err := os.MkdirAll(workerDir, 0700); err != nil {
		t.Fatal(err)
	}
	writeHandshake(t, workerDir, session, 42, 3)
	unfinalizedWorker(t, workerDir, sid)

	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.CaptureSuccess || got.Outcome != captureOutcomePartialRecovered {
		t.Fatalf("collected target kept success alongside recovery: %+v", got)
	}
	if got.ActivityRecords != 3 || got.RecoveredActivityRecords != 2 {
		t.Fatalf("counts = %d total, %d recovered", got.ActivityRecords, got.RecoveredActivityRecords)
	}
}

// TestRecoverUnfinalizedRejectsTruncated pins that recovery admits only
// structurally intact journals: a torn tail is still a hard rejection.
func TestRecoverUnfinalizedRejectsTruncated(t *testing.T) {
	dir := recoverySession(t, func(t *testing.T, dir string, sid [16]byte) {
		path := filepath.Join(dir, "capture.journal")
		unfinalizedWorker(t, dir, sid)
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte{1, 2, 3}); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	})
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.CaptureSuccess || got.Outcome != "initialized-no-activity" {
		t.Fatalf("truncated worker = %+v", got)
	}
	if got.ActivityRecords != 0 || got.RecoveredGenerations != 0 {
		t.Fatalf("truncated worker contributed: %+v", got)
	}
	if w := got.WorkerGenerations; len(w) != 1 || w[0].Disposition != "incomplete" || w[0].Failure == "" {
		t.Fatalf("truncated worker manifest = %+v", got.WorkerGenerations)
	}
}

// TestRecoverUnfinalizedRejectsIdentityMismatch pins that recovery still checks
// the handshake generation before admitting anything.
func TestRecoverUnfinalizedRejectsIdentityMismatch(t *testing.T) {
	dir := recoverySession(t, func(t *testing.T, dir string, sid [16]byte) {
		appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 42, 4, []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
		})
	})
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.CaptureSuccess || got.ActivityRecords != 0 || got.RecoveredGenerations != 0 {
		t.Fatalf("identity mismatch contributed: %+v", got)
	}
	if w := got.WorkerGenerations; len(w) != 1 || w[0].Disposition != "identity-mismatch" {
		t.Fatalf("identity mismatch manifest = %+v", got.WorkerGenerations)
	}
}

// TestRecoverUnfinalizedPreservesLossDisposition pins that a producer that
// reported loss before dying keeps that disposition through recovery, rather
// than having it replaced by the missing-finalize message.
func TestRecoverUnfinalizedPreservesLossDisposition(t *testing.T) {
	dir := recoverySession(t, func(t *testing.T, dir string, sid [16]byte) {
		appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 42, 3, []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
			{Kind: recordFailure, Payload: []byte(`{"state":"producer_failure"}`)},
		})
	})
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.CaptureSuccess || got.Outcome != captureOutcomePartialRecovered {
		t.Fatalf("loss recovery = %+v", got)
	}
	w := got.WorkerGenerations
	if len(w) != 1 || w[0].Disposition != captureWorkerUnfinalizedPartial || w[0].Failure != "producer_failure" {
		t.Fatalf("loss disposition = %+v", w)
	}
}
