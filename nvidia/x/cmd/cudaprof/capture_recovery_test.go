package main

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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

// TestUnfinalizedTargetStillReportsWhatItRead pins the diagnosis on a target
// generation that decoded a bootstrap record and then stopped without
// finalizing -- the shape a Go binary leaves behind, since runtime.exit issues
// exit_group and runs neither the agent's atexit handler nor its ELF
// destructor. The capture is still refused. What it must not do is describe the
// bootstrap it just read as absent: bootstrap_seen:false on this journal reads
// as "the agent never loaded" and sends the reader after an injection or
// libcupti fault that did not happen.
func TestUnfinalizedTargetStillReportsWhatItRead(t *testing.T) {
	const session = "00112233445566778899aabbccddeeff"
	dir := t.TempDir()
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)

	writeHandshake(t, dir, session, 41, 2)
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
	})

	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.CaptureSuccess {
		t.Fatalf("unfinalized target was published: %+v", got)
	}
	if !got.BootstrapSeen {
		t.Fatalf("journal holds a bootstrap record but bootstrap_seen is false: %+v", got)
	}
	// The refusal is correct and, on its own, useless: it describes the journal
	// and names nothing to change. Everything below is what a caller has to do
	// to get a Go target to collect, which is why it belongs in this message.
	for _, want := range []string{
		"never finalized",
		"exit_group",
		"--capture-range=cudaProfilerApi",
		"--finalize=profiler-stop",
		"--flush-interval",
		"ProfilerStop",
	} {
		if !strings.Contains(got.CaptureError, want) {
			t.Errorf("capture error omits %q:\n%s", want, got.CaptureError)
		}
	}
}

// targetSession writes a session whose target generation is written by
// writeTarget and which has no worker generations at all -- the shape of a
// plain single-process capture, as opposed to recoverySession's parent plus
// worker.
func targetSession(t *testing.T, writeTarget func(t *testing.T, dir string, sid [16]byte)) string {
	t.Helper()
	const session = "00112233445566778899aabbccddeeff"
	dir := t.TempDir()
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	writeHandshake(t, dir, session, 41, 2)
	writeTarget(t, dir, sid)
	return dir
}

// flushedTargetJournal is what a periodic flush leaves behind: real activity
// records published while the target was alive, and no finalized record,
// because the target exited without running any of the agent's exit paths.
func flushedTargetJournal(t *testing.T, dir string, sid [16]byte) {
	t.Helper()
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"memcpy","valid":true,"start":30,"end":40,"bytes":64}`)},
	})
}

// TestRecoverUnfinalizedTargetGeneration is the --flush-interval outcome. The
// records are the producer's own, structurally intact and identity-checked;
// only the generation's completion is unknown. Refusing them outright, which is
// what happened before, threw away the entire capture a periodic flush exists
// to produce.
func TestRecoverUnfinalizedTargetGeneration(t *testing.T) {
	dir := targetSession(t, flushedTargetJournal)
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.CaptureSuccess {
		t.Fatalf("recovered target was published as a successful capture: %+v", got)
	}
	if got.Outcome != captureOutcomePartialRecovered {
		t.Fatalf("outcome = %q, want %q", got.Outcome, captureOutcomePartialRecovered)
	}
	if got.CollectorFinalized {
		t.Fatal("recovery claimed the collector finalized")
	}
	// Nothing in this run qualified: the generation that did not finalize is
	// the target's own, which is exactly what runFailed means.
	if got.RunCompleteness != runFailed {
		t.Errorf("run_completeness = %q, want %q", got.RunCompleteness, runFailed)
	}
	if got.ActivityRecords != 2 || len(got.Activities) != 2 {
		t.Fatalf("recovered activity = %d records, %d samples", got.ActivityRecords, len(got.Activities))
	}
	if got.CopyBytes != 64 || got.ActivityNanos != 10 {
		t.Errorf("recovered metrics = bytes %d nanos %d", got.CopyBytes, got.ActivityNanos)
	}
	if got.RecoveredGenerations != 1 || got.RecoveredActivityRecords != 2 {
		t.Errorf("recovery counters = %d/%d", got.RecoveredGenerations, got.RecoveredActivityRecords)
	}
	// The recovered contribution must be subtractable from the published
	// totals, so the generation it came from has to be in the manifest.
	w := got.WorkerGenerations
	if len(w) != 1 || w[0].Process != 41 || w[0].Generation != 2 {
		t.Fatalf("manifest = %+v", w)
	}
	if w[0].Finalized || w[0].Disposition != captureWorkerUnfinalizedPartial || w[0].ActivityRecords != 2 {
		t.Errorf("recovered target disposition = %+v", w[0])
	}
	if got.UnqualifiedGenerations != 1 {
		t.Errorf("unqualified_generations = %d, want 1", got.UnqualifiedGenerations)
	}
	if got.Activities[0].Process != 41 {
		t.Errorf("recovered activity lost target identity: %+v", got.Activities[0])
	}
}

// TestRecoverUnfinalizedTargetDefaultsOff pins that the target path is as
// explicit as the worker one: without the flag the same journal contributes
// nothing and the capture is refused.
func TestRecoverUnfinalizedTargetDefaultsOff(t *testing.T) {
	dir := targetSession(t, flushedTargetJournal)
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true})
	if got.CaptureSuccess || got.Outcome != captureOutcomeNotFinalized {
		t.Fatalf("default outcome = %+v", got)
	}
	if got.ActivityRecords != 0 || got.RecoveredGenerations != 0 {
		t.Fatalf("default admitted target records: %+v", got)
	}
	// This refusal is throwing away records it just decoded, which is a
	// defensible default and an indefensible thing to be silent about.
	for _, want := range []string{"2 activity records", "--recover-unfinalized"} {
		if !strings.Contains(got.CaptureError, want) {
			t.Errorf("capture error omits %q:\n%s", want, got.CaptureError)
		}
	}
}

// TestRecoverUnfinalizedTargetKeepsDiagnosisWhenNothingWasFlushed is where the
// two features meet. A Go target with no periodic flush writes a bootstrap and
// a capability record and nothing else; there is no record to recover, and
// answering with an empty partial capture would replace the one message that
// says what to change with a well-labelled zero.
func TestRecoverUnfinalizedTargetKeepsDiagnosisWhenNothingWasFlushed(t *testing.T) {
	dir := targetSession(t, func(t *testing.T, dir string, sid [16]byte) {
		appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		})
	})
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.Outcome != captureOutcomeNotFinalized || got.RecoveredGenerations != 0 {
		t.Fatalf("empty target was recovered: %+v", got)
	}
	if !strings.Contains(got.CaptureError, "--flush-interval") {
		t.Errorf("capture error lost the diagnosis: %q", got.CaptureError)
	}
}

// TestRecoverUnfinalizedTargetKeepsProducerFailure pins the precedence: a
// target that recorded why it failed reports that cause, not the generic
// partial label. The cause is the more actionable of the two.
func TestRecoverUnfinalizedTargetKeepsProducerFailure(t *testing.T) {
	dir := targetSession(t, func(t *testing.T, dir string, sid [16]byte) {
		appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
			{Kind: recordFailure, Payload: []byte(`{"state":"cupti_unavailable","error":"libcupti.so not found"}`)},
		})
	})
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.Outcome != captureOutcomeUnsupported {
		t.Fatalf("outcome = %q, want %q: %+v", got.Outcome, captureOutcomeUnsupported, got)
	}
	if !strings.Contains(got.CaptureError, "libcupti.so not found") {
		t.Errorf("capture error lost the producer's reason: %q", got.CaptureError)
	}
	if got.RecoveredGenerations != 0 {
		t.Errorf("failed target was counted as recovered: %+v", got)
	}
}

// TestRecoverUnfinalizedTargetRejectsTruncated: recovery admits only journals
// that decoded completely, on the target path as on the worker one.
func TestRecoverUnfinalizedTargetRejectsTruncated(t *testing.T) {
	dir := targetSession(t, func(t *testing.T, dir string, sid [16]byte) {
		flushedTargetJournal(t, dir, sid)
		f, err := os.OpenFile(filepath.Join(dir, "capture.journal"), os.O_WRONLY|os.O_APPEND, 0600)
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
	if got.CaptureSuccess || got.RecoveredGenerations != 0 || got.ActivityRecords != 0 {
		t.Fatalf("truncated target contributed: %+v", got)
	}
}

// TestCaptureHelpDocumentsTheCollectionPolicyFlags: these flags are the only
// way to collect anything from a target that does not finalize, and a flag that
// is parsed but not documented is one nobody can find. --recover-unfinalized
// had been in exactly that state.
func TestCaptureHelpDocumentsTheCollectionPolicyFlags(t *testing.T) {
	help := captureHelp()
	for _, want := range []string{
		"--capture-range",
		captureRangeProfilerAPI,
		"--finalize",
		finalizeProfilerStop,
		finalizeExitGroup,
		"--flush-interval",
		"CUDAPROF_FLUSH_INTERVAL_MS",
		"--recover-unfinalized",
		captureOutcomePartialRecovered,
	} {
		if !strings.Contains(help, want) {
			t.Errorf("capture help omits %q:\n%s", want, help)
		}
	}
}

// TestUnfinalizedTargetReportsTheTransportItUsed is the field next to the one
// TestUnfinalizedTargetStillReportsWhatItRead covers, with the same defect.
//
// transport_ready was set only on the path that reaches a usable journal, so a
// target that completed its handshake and wrote records published
// transport_ready:false when it failed to finalize. runCompletenessOf turns
// that into "unknown", documented as "no transport was established" -- so two
// published fields told the reader the channel never came up, about a run
// where it demonstrably had. A GB10 capture was diagnosed as a transport fault
// on the strength of it; the transport was fine and the target had exited
// through Go's runtime without finalizing.
//
// Three arms, because the obvious two do not discriminate. A session directory
// with no handshake returns before the refusal path is ever reached, so it
// cannot tell a reported value from a hardcoded one -- it only guards the
// earlier path, where nothing observed the transport and "unknown" is right.
// The arm that pins the value to its source is the one whose handshake says
// transport_ready:false: the published field has to follow it.
func TestUnfinalizedTargetReportsTheTransportItUsed(t *testing.T) {
	const session = "00112233445566778899aabbccddeeff"
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)

	dir := t.TempDir()
	writeHandshake(t, dir, session, 41, 2)
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
	})

	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.CaptureSuccess {
		t.Fatalf("unfinalized target was published: %+v", got)
	}
	if !got.TransportReady {
		t.Errorf("target completed a handshake and wrote records but transport_ready is false: %+v", got)
	}
	// The point of the field is what it makes the classifier say. "unknown"
	// claims no evidence either way; there is evidence, and it says the
	// collector never closed the books.
	if got.RunCompleteness != runFailed {
		t.Errorf("run_completeness = %q, want %q: the transport carried a journal, the producer never finalized it", got.RunCompleteness, runFailed)
	}

	// No handshake at all: nothing observed the transport, so absence of
	// evidence must still read as unknown rather than as failure. This arm
	// returns before the refusal path above and so proves nothing about where
	// the value came from; it guards the early path only.
	empty := observeCaptureSession(captureResult{TargetPID: 41}, t.TempDir(), session, observeOptions{injectionRequested: true})
	if empty.TransportReady {
		t.Errorf("transport_ready true with no handshake on disk: %+v", empty)
	}
	if empty.RunCompleteness != runUnknown {
		t.Errorf("run_completeness = %q, want %q with no transport observed", empty.RunCompleteness, runUnknown)
	}

	// A handshake that reports its transport as not ready. The agent writes
	// the field as a literal true today, so this is not a shape it currently
	// emits -- which is the point: it is the only arm that fails if the
	// published field stops following the handshake and becomes a constant.
	denied := t.TempDir()
	b, err := json.Marshal(captureHandshake{Protocol: 1, Session: session, Process: 41, Generation: 2, TransportReady: false})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(denied, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	appendRecords(t, filepath.Join(denied, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
	})
	if got := observeCaptureSession(captureResult{TargetPID: 41}, denied, session, observeOptions{injectionRequested: true}); got.TransportReady {
		t.Errorf("handshake said transport_ready:false and the capture published true: %+v", got)
	}
}

// TestNotFinalizedIsNotATransportFailure pins the outcome token for a target
// that got its records across and never closed the books.
//
// Both arms fail the capture and both set targetJournalErr, so a single arm
// could not tell the two tokens apart -- it would pass with the branch deleted.
// The truncated arm is what makes the split mean something: a journal that did
// not arrive intact is a transport failure and must keep saying so.
func TestNotFinalizedIsNotATransportFailure(t *testing.T) {
	const session = "00112233445566778899aabbccddeeff"
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)

	write := func(t *testing.T) string {
		t.Helper()
		dir := t.TempDir()
		writeHandshake(t, dir, session, 41, 2)
		appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
			{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
		})
		return dir
	}

	got := observeCaptureSession(captureResult{TargetPID: 41}, write(t), session, observeOptions{injectionRequested: true})
	if got.Outcome != captureOutcomeNotFinalized {
		t.Errorf("outcome = %q, want %q: the handshake completed and records arrived, so the transport is the one thing that worked", got.Outcome, captureOutcomeNotFinalized)
	}

	// Same journal, cut short so it no longer decodes. The records did not
	// arrive intact, which is what transport-failed says.
	cut := write(t)
	path := filepath.Join(cut, "capture.journal")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, info.Size()-16); err != nil {
		t.Fatal(err)
	}
	short := observeCaptureSession(captureResult{TargetPID: 41}, cut, session, observeOptions{injectionRequested: true})
	if short.Outcome != captureOutcomeTransport {
		t.Errorf("truncated journal outcome = %q, want %q", short.Outcome, captureOutcomeTransport)
	}
	if short.Outcome == got.Outcome {
		t.Fatalf("both journal failures report %q: the test cannot discriminate the two tokens", got.Outcome)
	}
}
