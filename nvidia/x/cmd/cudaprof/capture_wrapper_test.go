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

// A wrapper process is the ordinary way a CUDA program is started: a shell, a
// launcher, srun, a test runner. Injection is inherited across exec, so the
// wrapper writes a journal of its own, and a shell leaves through _exit with
// nothing in it -- the same journal, byte for byte, that a target which
// collected records and lost them to exit_group leaves behind.
//
// Refusing both costs the run every record the program the wrapper started
// collected. Admitting both reports a failed capture as an empty success. The
// producer's device_active note is what separates them, so all three arms run
// here: without it the capture is the children's, with it the target's own
// loss stands, and a producer too old to write it gets the refusal it always
// got rather than a silent reinterpretation of its journal.
func TestObserveCaptureSessionKeepsAWrapperTargetsChildren(t *testing.T) {
	const session = "00112233445566778899aabbccddeeff"
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)

	// writeGeneration writes one process's handshake and journal. A nil
	// finalized payload leaves the journal unfinalized, which is what both a
	// wrapper and a lost capture produce.
	writeGeneration := func(t *testing.T, dir, name string, process, generation uint64, capability string, records []wire.Record, finalized string) {
		t.Helper()
		streamDir := filepath.Join(dir, name)
		if err := os.MkdirAll(streamDir, 0700); err != nil {
			t.Fatal(err)
		}
		h := captureHandshake{Protocol: captureProtocolVersion, Session: session, Process: process, Generation: generation, TransportReady: true}
		b, _ := json.Marshal(h)
		if err := os.WriteFile(filepath.Join(streamDir, "handshake.json"), b, 0600); err != nil {
			t.Fatal(err)
		}
		j, err := journal.Create(filepath.Join(streamDir, "capture.journal"), sid, journal.Options{})
		if err != nil {
			t.Fatal(err)
		}
		all := []wire.Record{
			{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
			{Kind: recordCapability, Payload: []byte(capability)},
		}
		all = append(all, records...)
		if finalized != "" {
			all = append(all, wire.Record{Kind: recordFinalized, Payload: []byte(finalized)})
		}
		for i := range all {
			all[i].Identity = wire.Identity{Session: sid, Process: process, Generation: generation}
			all[i].Sequence = uint64(i + 1)
			if err := j.Append(all[i]); err != nil {
				t.Fatal(err)
			}
		}
		if err := j.Close(); err != nil {
			t.Fatal(err)
		}
	}
	const notes = `{"kind":"capability","device_active_note":true}`
	deviceActive := wire.Record{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"device_active"}`)}
	kernel := wire.Record{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":30}`)}

	// session builds one capture session: a target generation of the given
	// shape and one CUDA child that finalized cleanly.
	sessionDir := func(t *testing.T, capability string, targetRecords []wire.Record) string {
		t.Helper()
		dir := t.TempDir()
		writeGeneration(t, dir, "process-41-2", 41, 2, capability, targetRecords, "")
		writeGeneration(t, dir, "process-42-3", 42, 3, notes, []wire.Record{deviceActive, kernel}, `{"state":"collected"}`)
		return dir
	}

	t.Run("wrapper", func(t *testing.T) {
		dir := sessionDir(t, notes, nil)
		got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
		if got.Outcome != captureOutcomeCollected || !got.CaptureSuccess {
			t.Fatalf("outcome = %q, success = %v, want %q: the wrapper ran no CUDA, so the capture is what the process it started collected", got.Outcome, got.CaptureSuccess, captureOutcomeCollected)
		}
		if got.ActivityRecords != 1 || len(got.Activities) != 1 || got.ActivityNanos != 20 {
			t.Errorf("activity = %d records, %d nanos, %+v", got.ActivityRecords, got.ActivityNanos, got.Activities)
		}
		if got.RunCompleteness != runComplete || got.UnqualifiedGenerations != 0 {
			t.Errorf("run_completeness = %q, unqualified = %d: a generation with no CUDA in it had nothing to finalize", got.RunCompleteness, got.UnqualifiedGenerations)
		}
		if !got.CollectorFinalized || !got.TransportReady || !got.BootstrapSeen {
			t.Errorf("collector_finalized=%v transport_ready=%v bootstrap_seen=%v", got.CollectorFinalized, got.TransportReady, got.BootstrapSeen)
		}
		var target, child captureWorker
		for _, w := range got.WorkerGenerations {
			switch w.Process {
			case 41:
				target = w
			case 42:
				child = w
			}
		}
		if target.Disposition != captureWorkerNoCUDA || target.Failure != "" {
			t.Errorf("target generation = %+v, want disposition %q and no failure", target, captureWorkerNoCUDA)
		}
		if child.Disposition != "finalized" {
			t.Errorf("child generation = %+v", child)
		}
	})

	t.Run("wrapper that started nothing", func(t *testing.T) {
		dir := t.TempDir()
		writeGeneration(t, dir, "process-41-2", 41, 2, notes, nil, "")
		got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
		// The same answer a process that ran no CUDA and did finalize gets:
		// the agent came up, nothing ran under it, and nothing was lost.
		if got.Outcome != "initialized-no-activity" || got.CaptureSuccess {
			t.Fatalf("outcome = %q, success = %v", got.Outcome, got.CaptureSuccess)
		}
		if got.RunCompleteness != runComplete {
			t.Errorf("run_completeness = %q: a run that collected nothing because there was nothing to collect lost nothing", got.RunCompleteness)
		}
	})

	t.Run("target lost its own records", func(t *testing.T) {
		dir := sessionDir(t, notes, []wire.Record{deviceActive})
		got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
		if got.Outcome != captureOutcomeNotFinalized || got.CaptureSuccess {
			t.Fatalf("outcome = %q, success = %v, want %q: CUPTI asked this target for a buffer, so it had device work and did not publish it", got.Outcome, got.CaptureSuccess, captureOutcomeNotFinalized)
		}
	})

	t.Run("producer writes no note", func(t *testing.T) {
		dir := sessionDir(t, `{"kind":"capability"}`, nil)
		got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
		if got.Outcome != captureOutcomeNotFinalized || got.CaptureSuccess {
			t.Fatalf("outcome = %q, success = %v, want %q: an absent note means nothing from a producer that never writes one", got.Outcome, got.CaptureSuccess, captureOutcomeNotFinalized)
		}
	})
}
