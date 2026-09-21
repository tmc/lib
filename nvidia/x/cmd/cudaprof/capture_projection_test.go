package main

import (
	"encoding/hex"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// projectionSession builds a capture whose target generation runs two named
// kernels and one memcpy and finalizes cleanly. The memcpy carries a name that
// matches one of the kernel patterns on purpose: --exclude names kernels, and
// a copy that happens to share the string is not one.
func projectionSession(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	var sid [16]byte
	raw, _ := hex.DecodeString(completenessSessionID)
	copy(sid[:], raw)
	writeHandshake(t, dir, completenessSessionID, 41, 2)
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"ampere_sgemm_128x64_tn","valid":true,"start":10,"end":30}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"elementwise_add","valid":true,"start":40,"end":55}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"memcpy","name":"ampere_sgemm_128x64_tn","valid":true,"start":60,"end":70,"bytes":1024}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":3}`)},
	})
	return dir
}

func observeProjection(t *testing.T, dir string) captureResult {
	t.Helper()
	return observeCaptureSession(captureResult{TargetPID: 41}, dir, completenessSessionID, observeOptions{injectionRequested: true})
}

// TestExcludeKernelsLeavesTheTotalsAlone is the control for the trap --exclude
// invites: a capture whose counts describe one set of records and whose pprof
// holds another, with nothing in the file to say which. The totals stay
// journal totals, the exclusion is reported beside them, and the two reconcile
// exactly -- activity_samples == activity_records - excluded_records.
func TestExcludeKernelsLeavesTheTotalsAlone(t *testing.T) {
	base := observeProjection(t, projectionSession(t))
	got := excludeKernels(base, []string{"sgemm"})

	if got.ActivityRecords != base.ActivityRecords || got.ActivityNanos != base.ActivityNanos || got.CopyBytes != base.CopyBytes {
		t.Fatalf("exclusion moved the journal totals: records %d nanos %d bytes %d, want %d/%d/%d",
			got.ActivityRecords, got.ActivityNanos, got.CopyBytes, base.ActivityRecords, base.ActivityNanos, base.CopyBytes)
	}
	if got.ExcludedRecords != 1 || got.ExcludedNanos != 20 {
		t.Fatalf("excluded = %d records %d nanos, want 1/20", got.ExcludedRecords, got.ExcludedNanos)
	}
	if len(got.Activities) != 2 {
		t.Fatalf("projection = %+v", got.Activities)
	}
	for _, a := range got.Activities {
		if a.Subtype == "kernel" && strings.Contains(a.Name, "sgemm") {
			t.Fatalf("excluded kernel survived the projection: %+v", a)
		}
	}
	// The memcpy shares the excluded string and is still here: a copy has no
	// kernel name, so no --exclude can remove one.
	if got.CopyBytes != 1024 {
		t.Fatalf("copy bytes = %d, want 1024", got.CopyBytes)
	}
	status := compactCaptureStatus(got)
	if uint64(status.ActivitySamples) != got.ActivityRecords-got.ExcludedRecords {
		t.Fatalf("activity_samples %d != activity_records %d - excluded_records %d", status.ActivitySamples, got.ActivityRecords, got.ExcludedRecords)
	}
	if status.ActivityKinds["kernel"] != 1 || status.ActivityKinds["memcpy"] != 1 {
		t.Fatalf("activity kinds = %v", status.ActivityKinds)
	}
}

func TestExcludeKernelsMatching(t *testing.T) {
	base := observeProjection(t, projectionSession(t))
	for _, tc := range []struct {
		name     string
		patterns []string
		want     []string
		excluded uint64
	}{
		{"none", nil, []string{"ampere_sgemm_128x64_tn", "elementwise_add", "ampere_sgemm_128x64_tn"}, 0},
		{"substring", []string{"sgemm"}, []string{"elementwise_add", "ampere_sgemm_128x64_tn"}, 1},
		{"whole name", []string{"elementwise_add"}, []string{"ampere_sgemm_128x64_tn", "ampere_sgemm_128x64_tn"}, 1},
		{"repeated", []string{"sgemm", "elementwise"}, []string{"ampere_sgemm_128x64_tn"}, 2},
		{"no match", []string{"cutlass"}, []string{"ampere_sgemm_128x64_tn", "elementwise_add", "ampere_sgemm_128x64_tn"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := excludeKernels(base, tc.patterns)
			var names []string
			for _, a := range got.Activities {
				names = append(names, a.Name)
			}
			if strings.Join(names, ",") != strings.Join(tc.want, ",") {
				t.Errorf("projection = %v, want %v", names, tc.want)
			}
			if got.ExcludedRecords != tc.excluded {
				t.Errorf("excluded_records = %d, want %d", got.ExcludedRecords, tc.excluded)
			}
			// A pattern that matched nothing is still a policy the capture
			// applied, and a reader comparing counts needs to know it was.
			if len(got.ExcludedPatterns) != len(tc.patterns) {
				t.Errorf("excluded_patterns = %v, want %v", got.ExcludedPatterns, tc.patterns)
			}
		})
	}
}

// TestFollowChildrenDecidesWhoseRecordsCount is the pair of controls for the
// flag. Injection is inherited, so the child generation is observed either
// way; what moves is whether its records are part of this capture's totals.
func TestFollowChildrenDecidesWhoseRecordsCount(t *testing.T) {
	dir := completenessSession(t, healthyWorker(50, 60, 3))
	followed := observeCaptureSession(captureResult{TargetPID: 41}, dir, completenessSessionID, observeOptions{injectionRequested: true})
	targetOnly := observeCaptureSession(captureResult{TargetPID: 41}, dir, completenessSessionID, observeOptions{injectionRequested: true, targetOnly: true})

	if !followed.ChildrenFollowed || targetOnly.ChildrenFollowed {
		t.Fatalf("children_followed = %v / %v, want true / false", followed.ChildrenFollowed, targetOnly.ChildrenFollowed)
	}
	if followed.ActivityRecords != 4 || len(followed.Activities) != 4 {
		t.Fatalf("followed = %d records, %d samples, want 4/4", followed.ActivityRecords, len(followed.Activities))
	}
	if targetOnly.ActivityRecords != 1 || len(targetOnly.Activities) != 1 {
		t.Fatalf("target-only = %d records, %d samples, want 1/1", targetOnly.ActivityRecords, len(targetOnly.Activities))
	}
	// Not following a generation is not failing to observe it: the row, its
	// counts and its disposition are reported identically either way.
	for _, got := range []captureResult{followed, targetOnly} {
		if len(got.WorkerGenerations) != 1 {
			t.Fatalf("worker manifest = %+v", got.WorkerGenerations)
		}
		w := got.WorkerGenerations[0]
		if w.Process != 50 || w.Generation != 60 || w.ActivityRecords != 3 || w.Disposition != "finalized" {
			t.Fatalf("worker manifest = %+v", w)
		}
		if got.RunCompleteness != runComplete || !got.CaptureSuccess {
			t.Fatalf("completeness = %q success = %v", got.RunCompleteness, got.CaptureSuccess)
		}
	}
}

// TestFollowChildrenFalseStillCostsRunCompleteness is the discriminating
// control. Withholding a generation's records from the totals must not
// withhold its verdict from the run: a capture that could be made to look
// complete by not following the child that failed would be worse than one that
// merged it silently.
func TestFollowChildrenFalseStillCostsRunCompleteness(t *testing.T) {
	dir := completenessSession(t, failedWorker(50, 60, 3))
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, completenessSessionID, observeOptions{injectionRequested: true, targetOnly: true})

	if got.RunCompleteness != runPartial || got.UnqualifiedGenerations != 1 {
		t.Fatalf("completeness = %q / %d, want %q / 1", got.RunCompleteness, got.UnqualifiedGenerations, runPartial)
	}
	if got.ActivityRecords != 1 || len(got.Activities) != 1 {
		t.Fatalf("target-only totals admitted the child: %d records, %d samples", got.ActivityRecords, len(got.Activities))
	}
	if w := got.WorkerGenerations[0]; w.ActivityRecords != 3 || w.Disposition != "producer_failure" {
		t.Fatalf("worker manifest = %+v", w)
	}
}

// TestFollowChildrenFalseWithdrawsWorkerRecovery: --recover-unfinalized admits
// a generation's records as an explicit partial. With the generation not being
// followed there is nothing to admit, so the capture must not be labelled
// partial-recovered for a recovery that contributed nothing.
func TestFollowChildrenFalseWithdrawsWorkerRecovery(t *testing.T) {
	dir := recoverySession(t, unfinalizedWorker)
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, completenessSessionID, observeOptions{injectionRequested: true, recoverUnfinalized: true, targetOnly: true})

	if got.RecoveredGenerations != 0 || got.RecoveredActivityRecords != 0 {
		t.Fatalf("recovery counters = %d/%d, want 0/0", got.RecoveredGenerations, got.RecoveredActivityRecords)
	}
	if got.Outcome == captureOutcomePartialRecovered {
		t.Fatalf("capture labelled %q for a recovery that admitted nothing", got.Outcome)
	}
	if got.ActivityRecords != 0 || len(got.Activities) != 0 {
		t.Fatalf("target-only totals admitted the unfollowed worker: %+v", got)
	}
	w := got.WorkerGenerations[0]
	if w.Disposition == captureWorkerUnfinalizedPartial {
		t.Fatalf("unfollowed generation reported as recovered: %+v", w)
	}
	if w.Failure == "" || got.RunCompleteness != runPartial {
		t.Fatalf("unfollowed generation lost its verdict: %+v / %q", w, got.RunCompleteness)
	}
}

// TestCaptureHeadlineReportsTheProjectionPolicy pins that both ways of
// publishing fewer records than the totals say are stated on the terminal. A
// caller who sees "capture complete: 4 records" and finds two in the pprof has
// been told nothing by the number that was supposed to tell them.
func TestCaptureHeadlineReportsTheProjectionPolicy(t *testing.T) {
	got := captureHeadline(captureResult{
		CaptureSuccess:   true,
		RunCompleteness:  runComplete,
		ActivityRecords:  4,
		ExcludedPatterns: []string{"sgemm"},
		ExcludedRecords:  1,
		ExcludedNanos:    20,
		ChildrenFollowed: false,
		WorkerGenerations: []captureWorker{
			{Process: 50, Generation: 60, ActivityRecords: 3, Finalized: true, Disposition: "finalized"},
		},
	})
	for _, want := range []string{
		"excluded 1 kernel record(s)",
		"sgemm",
		"cudaprof gaps still sees them",
		"1 child generation(s) holding 3 activity record(s)",
		"--follow-children=false",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("headline does not say %q:\n%s", want, got)
		}
	}
	// The default policy has nothing to report, and a line that always prints
	// is a line nobody reads.
	quiet := captureHeadline(captureResult{CaptureSuccess: true, RunCompleteness: runComplete, ActivityRecords: 4, ChildrenFollowed: true})
	if strings.Contains(quiet, "excluded") || strings.Contains(quiet, "--follow-children") {
		t.Errorf("headline reports a policy nobody set:\n%s", quiet)
	}
}
