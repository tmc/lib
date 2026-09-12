package main

import (
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

const completenessSessionID = "00112233445566778899aabbccddeeff"

// completenessWorker describes one worker generation to place in a fixture.
// A nonempty injectionFailure writes the failure into the handshake and no
// journal at all, which is the shape an atfork registration failure leaves.
type completenessWorker struct {
	process, generation uint64
	injectionFailure    string
	records             []wire.Record
}

// completenessSession builds a capture whose target generation finalizes
// cleanly with one kernel, plus the given worker generations. The target is
// deliberately healthy in every case: these controls exist to show that a
// qualified target does not qualify the run.
func completenessSession(t *testing.T, workers ...completenessWorker) string {
	t.Helper()
	dir := t.TempDir()
	var sid [16]byte
	raw, _ := hex.DecodeString(completenessSessionID)
	copy(sid[:], raw)

	writeHandshake(t, dir, completenessSessionID, 41, 2)
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":1}`)},
	})

	for _, w := range workers {
		workerDir := filepath.Join(dir, "process-"+strconv.FormatUint(w.process, 10)+"-"+strconv.FormatUint(w.generation, 10))
		if err := os.MkdirAll(workerDir, 0700); err != nil {
			t.Fatal(err)
		}
		if w.injectionFailure != "" {
			writeFailedHandshake(t, workerDir, completenessSessionID, w.process, w.generation, w.injectionFailure)
			continue
		}
		writeHandshake(t, workerDir, completenessSessionID, w.process, w.generation)
		appendRecords(t, filepath.Join(workerDir, "capture.journal"), sid, w.process, w.generation, w.records)
	}
	return dir
}

func writeFailedHandshake(t *testing.T, dir, session string, process, generation uint64, failure string) {
	t.Helper()
	b, err := json.Marshal(captureHandshake{Protocol: 1, Session: session, Process: process, Generation: generation, InjectionFailure: failure})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func healthyWorker(process, generation uint64, activities int) completenessWorker {
	records := []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
	}
	for i := 0; i < activities; i++ {
		records = append(records, wire.Record{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)})
	}
	state := `{"state":"initialized_no_activity"}`
	if activities > 0 {
		state = `{"state":"collected","activity":1}`
	}
	records = append(records, wire.Record{Kind: recordFinalized, Payload: []byte(state)})
	return completenessWorker{process: process, generation: generation, records: records}
}

func failedWorker(process, generation uint64, activities int) completenessWorker {
	w := healthyWorker(process, generation, activities)
	w.records[len(w.records)-1] = wire.Record{Kind: recordFinalized, Payload: []byte(`{"state":"producer_failure","activity":1,"dropped":7}`)}
	return w
}

func observeCompleteness(t *testing.T, dir string) captureResult {
	t.Helper()
	return observeCaptureSession(captureResult{TargetPID: 41}, dir, completenessSessionID, observeOptions{injectionRequested: true})
}

// TestRunCompletenessAtforkFailedWorker is the required control: a healthy
// target next to a worker generation whose atfork registration failed. The
// target's own capture stays successful, and the run does not.
func TestRunCompletenessAtforkFailedWorker(t *testing.T) {
	dir := completenessSession(t, completenessWorker{process: 50, generation: 60, injectionFailure: "atfork registration failed"})
	got := observeCompleteness(t, dir)

	if !got.CaptureSuccess || got.Outcome != captureOutcomeCollected {
		t.Fatalf("target qualification changed: outcome=%q success=%v", got.Outcome, got.CaptureSuccess)
	}
	if got.RunCompleteness != runPartial {
		t.Fatalf("run_completeness = %q, want %q", got.RunCompleteness, runPartial)
	}
	if got.UnqualifiedGenerations != 1 {
		t.Fatalf("unqualified generations = %d, want 1", got.UnqualifiedGenerations)
	}
	if len(got.WorkerGenerations) != 1 {
		t.Fatalf("worker manifest = %+v", got.WorkerGenerations)
	}
	w := got.WorkerGenerations[0]
	if w.Disposition != "injection-failed" || w.Failure != "atfork registration failed" {
		t.Fatalf("worker disposition = %+v", w)
	}
}

// TestRunCompletenessSeparatesTargetFromRun is the discriminating control for
// the pair of fields. Without it the partial verdicts above could be nothing
// but capture_success under another name.
func TestRunCompletenessSeparatesTargetFromRun(t *testing.T) {
	healthy := observeCompleteness(t, completenessSession(t, healthyWorker(50, 60, 1)))
	broken := observeCompleteness(t, completenessSession(t, completenessWorker{process: 50, generation: 60, injectionFailure: "atfork registration failed"}))

	if healthy.CaptureSuccess != broken.CaptureSuccess {
		t.Fatalf("capture_success moved with the worker: healthy=%v broken=%v", healthy.CaptureSuccess, broken.CaptureSuccess)
	}
	if !healthy.CaptureSuccess {
		t.Fatal("healthy target was not successful")
	}
	if healthy.RunCompleteness == broken.RunCompleteness {
		t.Fatalf("run_completeness did not move with the worker: %q", healthy.RunCompleteness)
	}
	if healthy.RunCompleteness != runComplete || broken.RunCompleteness != runPartial {
		t.Fatalf("healthy=%q broken=%q", healthy.RunCompleteness, broken.RunCompleteness)
	}
}

// TestRunCompletenessWorkerFailureAfterActivity covers the required case where
// the failure arrives after the generation has already produced records. The
// records stay, and the run is still not complete.
func TestRunCompletenessWorkerFailureAfterActivity(t *testing.T) {
	got := observeCompleteness(t, completenessSession(t, failedWorker(50, 60, 3)))

	if got.RunCompleteness != runPartial || got.UnqualifiedGenerations != 1 {
		t.Fatalf("completeness = %q / %d", got.RunCompleteness, got.UnqualifiedGenerations)
	}
	// One target kernel plus the worker's three: a failed generation's records
	// are retained, never discarded because the generation failed.
	if got.ActivityRecords != 4 || len(got.Activities) != 4 {
		t.Fatalf("activity = %d records, %d entries", got.ActivityRecords, len(got.Activities))
	}
	w := got.WorkerGenerations[0]
	if w.Disposition != "producer_failure" || w.Failure != "producer_failure" || w.ActivityRecords != 3 {
		t.Fatalf("worker manifest = %+v", w)
	}
	if !w.Finalized {
		t.Fatal("worker that wrote a terminal record is not marked finalized")
	}
}

// TestRunCompletenessZeroRecordFailedWorker covers the required zero-record
// failure. The verdict must come from the reported failure, never from the
// count, and no loss may be invented to explain the empty generation.
func TestRunCompletenessZeroRecordFailedWorker(t *testing.T) {
	got := observeCompleteness(t, completenessSession(t, failedWorker(50, 60, 0)))

	if got.RunCompleteness != runPartial || got.UnqualifiedGenerations != 1 {
		t.Fatalf("completeness = %q / %d", got.RunCompleteness, got.UnqualifiedGenerations)
	}
	w := got.WorkerGenerations[0]
	if w.ActivityRecords != 0 || w.Disposition != "producer_failure" {
		t.Fatalf("worker manifest = %+v", w)
	}
	if got.DroppedRecords != 0 {
		t.Fatalf("dropped records invented for an empty generation: %d", got.DroppedRecords)
	}
	if got.ActivityRecords != 1 {
		t.Fatalf("target activity changed: %d", got.ActivityRecords)
	}
}

// TestRunCompletenessZeroRecordHealthyWorkerIsComplete is the negative twin of
// the control above: an empty generation that finalized cleanly is qualified.
// A zero count is not evidence that anything was lost or unobserved.
func TestRunCompletenessZeroRecordHealthyWorkerIsComplete(t *testing.T) {
	got := observeCompleteness(t, completenessSession(t, healthyWorker(50, 60, 0)))

	if got.RunCompleteness != runComplete || got.UnqualifiedGenerations != 0 {
		t.Fatalf("empty finalized generation was not qualified: %q / %d", got.RunCompleteness, got.UnqualifiedGenerations)
	}
	if len(got.WorkerGenerations) != 1 || got.WorkerGenerations[0].ActivityRecords != 0 {
		t.Fatalf("worker manifest = %+v", got.WorkerGenerations)
	}
}

// TestRunCompletenessAllFinalizedPositive is the required positive: every
// generation qualified, so the run is complete and nothing is reported partial.
func TestRunCompletenessAllFinalizedPositive(t *testing.T) {
	got := observeCompleteness(t, completenessSession(t, healthyWorker(50, 60, 2), healthyWorker(51, 61, 0)))

	if got.RunCompleteness != runComplete {
		t.Fatalf("run_completeness = %q, want %q", got.RunCompleteness, runComplete)
	}
	if got.UnqualifiedGenerations != 0 {
		t.Fatalf("unqualified generations = %d, want 0", got.UnqualifiedGenerations)
	}
	if !got.CaptureSuccess || got.Outcome != captureOutcomeCollected {
		t.Fatalf("healthy run not collected: %+v", got.Outcome)
	}
	if len(got.WorkerGenerations) != 2 {
		t.Fatalf("worker manifest = %+v", got.WorkerGenerations)
	}
}

// TestRunCompletenessUnknownWithoutTransport pins that absence of evidence is
// reported as unknown rather than as a failed or partial run.
func TestRunCompletenessUnknownWithoutTransport(t *testing.T) {
	got := observeCaptureSession(captureResult{TargetPID: 41}, t.TempDir(), completenessSessionID, observeOptions{injectionRequested: true})
	if got.RunCompleteness != runUnknown {
		t.Fatalf("run_completeness = %q, want %q", got.RunCompleteness, runUnknown)
	}
	if got.UnqualifiedGenerations != 0 {
		t.Fatalf("unqualified generations invented without transport: %d", got.UnqualifiedGenerations)
	}
}

// TestRunCompletenessRecoveredGenerationIsPartial keeps the F1 recovery path
// consistent with this verdict: a recovered unfinalized generation never
// qualifies, whatever its records say.
func TestRunCompletenessRecoveredGenerationIsPartial(t *testing.T) {
	dir := recoverySession(t, unfinalizedWorker)
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, "00112233445566778899aabbccddeeff", observeOptions{injectionRequested: true, recoverUnfinalized: true})
	if got.RunCompleteness != runPartial || got.UnqualifiedGenerations != 1 {
		t.Fatalf("recovered generation completeness = %q / %d", got.RunCompleteness, got.UnqualifiedGenerations)
	}
}

// TestRunCompletenessSurvivesStatusWrite covers the published surface: the
// verdict and the per-generation dispositions must both reach a reader, and the
// compact status must not drop them along with the record arrays.
func TestRunCompletenessSurvivesStatusWrite(t *testing.T) {
	result := observeCompleteness(t, completenessSession(t, failedWorker(50, 60, 2), healthyWorker(51, 61, 1)))
	result.ScratchDir = t.TempDir()
	out := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	_, status := readWrittenCapture(t, out)

	if status.RunCompleteness != runPartial || status.UnqualifiedGenerations != 1 {
		t.Fatalf("published completeness = %q / %d", status.RunCompleteness, status.UnqualifiedGenerations)
	}
	if !status.CaptureSuccess {
		t.Fatal("published capture_success lost its target-qualified meaning")
	}
	if len(status.WorkerGenerations) != 2 {
		t.Fatalf("published worker manifest = %+v", status.WorkerGenerations)
	}
	if len(status.Activities) != 0 || len(status.APIRecords) != 0 {
		t.Fatal("compact status regained its record arrays")
	}
}

// TestInspectReportsRunCompleteness covers the reader: inspect republishes the
// verdict and every generation rather than the target boolean alone.
func TestInspectReportsRunCompleteness(t *testing.T) {
	result := observeCompleteness(t, completenessSession(t, failedWorker(50, 60, 2)))
	result.ScratchDir = t.TempDir()
	out := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	c, _ := readWrittenCapture(t, out)
	coverage := inspectCoverageOf(t, c.Files["capture-status.json"])

	if coverage.RunCompleteness != runPartial || coverage.UnqualifiedGenerations != 1 {
		t.Fatalf("inspect completeness = %q / %d", coverage.RunCompleteness, coverage.UnqualifiedGenerations)
	}
	if !coverage.CaptureSuccess {
		t.Fatal("inspect lost the target-qualified success")
	}
	if len(coverage.Workers) != 1 || coverage.Workers[0].Disposition != "producer_failure" {
		t.Fatalf("inspect worker manifest = %+v", coverage.Workers)
	}
}

// TestInspectDerivesCompletenessForLegacyStatus covers artifacts written before
// this field existed. They carry dispositions but no verdict, and a reader must
// not present their target-qualified success on its own.
func TestInspectDerivesCompletenessForLegacyStatus(t *testing.T) {
	legacy := []byte(`{"status":{"outcome":"collected","capture_success":true,"transport_ready":true,` +
		`"bootstrap_seen":true,"collector_finalized":true,"activity_records":4285,"dropped_records":249418,` +
		`"worker_generations":[` +
		`{"process":25752,"generation":2055307139432,"records":24403,"activity_records":2159,"finalized":true,"failure":"producer_failure","disposition":"producer_failure"},` +
		`{"process":25641,"generation":2044490862919,"records":3,"activity_records":0,"finalized":true,"disposition":"finalized"},` +
		`{"process":25644,"generation":2044501224599,"records":0,"activity_records":0,"finalized":false,"disposition":"incomplete"}]}}`)
	coverage := inspectCoverageOf(t, legacy)

	if coverage.RunCompleteness != runPartial {
		t.Fatalf("legacy run_completeness = %q, want %q", coverage.RunCompleteness, runPartial)
	}
	if coverage.UnqualifiedGenerations != 2 {
		t.Fatalf("legacy unqualified generations = %d, want 2", coverage.UnqualifiedGenerations)
	}
	if len(coverage.Workers) != 3 {
		t.Fatalf("legacy worker manifest = %+v", coverage.Workers)
	}
}

// TestInspectReportsFailedRunWithoutFinalization covers the remaining verdict.
// A status whose target generation never finalized cannot qualify any
// generation, and that is distinct from having no transport evidence at all.
func TestInspectReportsFailedRunWithoutFinalization(t *testing.T) {
	unfinalized := []byte(`{"status":{"capture_success":false,"transport_ready":true,"collector_finalized":false,` +
		`"worker_generations":[` +
		`{"process":1,"generation":2,"records":3,"activity_records":1,"finalized":true,"disposition":"finalized"}]}}`)
	coverage := inspectCoverageOf(t, unfinalized)
	if coverage.RunCompleteness != runFailed {
		t.Fatalf("run_completeness = %q, want %q", coverage.RunCompleteness, runFailed)
	}
	if coverage.UnqualifiedGenerations != 0 {
		t.Fatalf("qualified generation counted against a failed run: %d", coverage.UnqualifiedGenerations)
	}
}

// TestInspectKeepsPublishedCompleteness pins that the derivation above never
// overrides a verdict the capture itself published.
func TestInspectKeepsPublishedCompleteness(t *testing.T) {
	// The published verdict here disagrees with what this reader would derive
	// from the same worker rows. That is the point: the capture knows things
	// the status does not spell out, and the reader must not overrule it.
	published := []byte(`{"status":{"capture_success":true,"transport_ready":true,"collector_finalized":true,` +
		`"run_completeness":"partial","unqualified_generations":4,"worker_generations":[` +
		`{"process":1,"generation":2,"records":3,"activity_records":0,"finalized":true,"disposition":"finalized"}]}}`)
	coverage := inspectCoverageOf(t, published)
	if coverage.RunCompleteness != runPartial {
		t.Fatalf("published verdict was rewritten: %q", coverage.RunCompleteness)
	}
	if coverage.UnqualifiedGenerations != 4 {
		t.Fatalf("published count was recomputed: %d, want 4", coverage.UnqualifiedGenerations)
	}
}

func inspectCoverageOf(t *testing.T, status []byte) inspectCoverage {
	t.Helper()
	var envelope struct {
		Status inspectCoverage `json:"status"`
	}
	if err := json.Unmarshal(status, &envelope); err != nil {
		t.Fatal(err)
	}
	addCoverageDetails(&envelope.Status, status)
	return envelope.Status
}
