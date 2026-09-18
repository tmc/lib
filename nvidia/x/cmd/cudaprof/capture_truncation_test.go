package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// probeStatus is the capture-status.json a real ds4-min capture produced on the
// DGX: a run that finalized cleanly, self-reported success, and lost 28.5
// million records. A second capture of the same workload retained 33633 rather
// than 50702, so two captures that both announced success were truncated to
// wildly different fractions and any diff across them compares unlike samples.
func probeStatus() map[string]any {
	return map[string]any{
		"capture_success":  true,
		"outcome":          "collected",
		"run_completeness": "partial",
		"dropped_records":  28535365,
		"activity_records": 50702,
		"activity_kinds":   map[string]any{"kernel": 50401, "memcpy": 301},
	}
}

// TestGapsRefusesTheProbeCapture pins that the refusal fires on the shape that
// actually occurred, not only on a synthetic one, and that it reports both
// sides of the ratio: a drop count beside an unknown retained count does not
// say how truncated the capture is.
func TestGapsRefusesTheProbeCapture(t *testing.T) {
	cap := testCapture(map[string][]byte{"capture.journal": uniformJournal()}, probeStatus())
	_, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
	if err == nil {
		t.Fatal("analyzeGaps accepted a capture that dropped 28535365 records")
	}
	for _, want := range []string{"28535365", "retained 50702", "cannot recover an unwritten record"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal does not report %q: %v", want, err)
		}
	}
	// The one knob a reader can reach for is the wrong one. Offering it as the
	// remedy would send the user to re-run with a flag that bounds decoding and
	// cannot recover a record the producer never wrote.
	if strings.Contains(err.Error(), "re-run with --max-records") {
		t.Errorf("refusal offers --max-records as the remedy: %v", err)
	}
}

// TestGapsRefusesPartialRunWithoutDrops covers the loss a drop count cannot
// report. A generation that never finalized contributes no records at all, so
// its activities are absent rather than counted as dropped, and capture_success
// stays true through it.
func TestGapsRefusesPartialRunWithoutDrops(t *testing.T) {
	status := map[string]any{
		"capture_success":         true,
		"outcome":                 "collected",
		"run_completeness":        "partial",
		"dropped_records":         0,
		"unqualified_generations": 2,
		"activity_records":        50702,
	}
	cap := testCapture(map[string][]byte{"capture.journal": uniformJournal()}, status)
	_, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
	if err == nil {
		t.Fatal("analyzeGaps accepted a run with 2 unqualified generations and no drops")
	}
	for _, want := range []string{"run_completeness", "2 unqualified generation", "capture_success qualifies the target generation alone"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal does not report %q: %v", want, err)
		}
	}
}

// TestGapsAcceptsCompleteRun is the negative control for the two refusals
// above: a refusal that fires on every capture proves nothing.
func TestGapsAcceptsCompleteRun(t *testing.T) {
	cap := testCapture(map[string][]byte{"capture.journal": uniformJournal()}, completeStatus())
	if _, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords}); err != nil {
		t.Fatalf("analyzeGaps refused a complete capture: %v", err)
	}
}

// TestGapsRefusalNamesTheMechanism pins that the refusal reports which of the
// producer's loss mechanisms lost the records. The remedies differ and the
// counts alone do not distinguish them.
func TestGapsRefusalNamesTheMechanism(t *testing.T) {
	tests := []struct {
		name string
		loss map[string]any
		want string
	}{
		{
			name: "agent pending ring overflowed",
			loss: map[string]any{"queue_dropped": 900, "queue_capacity_dropped": 900, "offered_records": 1000, "retained_records": 100},
			want: "pending ring filling up",
		},
		{
			name: "cupti dropped before the agent",
			loss: map[string]any{"native_dropped": 900, "offered_records": 1000, "retained_records": 100},
			want: "dropped by CUPTI before the agent saw them",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status := probeStatus()
			status["record_loss"] = tt.loss
			cap := testCapture(map[string][]byte{"capture.journal": uniformJournal()}, status)
			_, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
			if err == nil {
				t.Fatal("analyzeGaps accepted a truncated capture")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("refusal does not name the mechanism %q: %v", tt.want, err)
			}
			if !strings.Contains(err.Error(), "retained 10.00%") {
				t.Errorf("refusal does not report the retained fraction: %v", err)
			}
			// Both producer-side mechanisms are reached by the same lever, and
			// the flag a reader would reach for instead is not.
			if !strings.Contains(err.Error(), "CUDAPROF_API_ACTIVITY=0") {
				t.Errorf("refusal names no reachable remedy: %v", err)
			}
			if !strings.Contains(err.Error(), "--max-records is not the remedy here") {
				t.Errorf("refusal does not rule out --max-records: %v", err)
			}
		})
	}
}

func TestCaptureLossDominant(t *testing.T) {
	tests := []struct {
		name string
		loss captureLoss
		want string
	}{
		{"nothing lost", captureLoss{}, ""},
		{"ring capacity", captureLoss{Queue: 90, QueueCapacity: 90, Native: 10}, "pending ring filling up"},
		{"cupti", captureLoss{Native: 90, Queue: 10, QueueOversize: 10}, "dropped by CUPTI"},
		{"queue but not capacity", captureLoss{Queue: 90, QueueOversize: 60, QueueContention: 30, Native: 10}, "refused by the agent queue"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.loss.dominant(nil)
			if tt.want == "" {
				if got != "" {
					t.Fatalf("dominant() = %q, want empty", got)
				}
				return
			}
			if !strings.Contains(got, tt.want) {
				t.Errorf("dominant() = %q, want it to contain %q", got, tt.want)
			}
		})
	}
}

func TestCaptureLossRetainedFraction(t *testing.T) {
	if _, ok := (captureLoss{Retained: 5}).retainedFraction(); ok {
		t.Error("retainedFraction reported a fraction with no offered total")
	}
	got, ok := captureLoss{Offered: 28586067, Retained: 50702}.retainedFraction()
	if !ok || got < 0.17 || got > 0.18 {
		t.Errorf("retainedFraction = %v (%v), want ~0.177%%", got, ok)
	}
	if (captureLoss{}).orNil() != nil {
		t.Error("orNil kept an all-zero breakdown")
	}
	if (captureLoss{Native: 1}).orNil() == nil {
		t.Error("orNil dropped a real breakdown")
	}
}

// TestCaptureHeadlineNeverCallsATruncatedRunComplete is the headline fix: the
// word a caller reads has to follow run_completeness, not the target-scoped
// capture_success boolean that stays true through a 28.5-million-record loss.
func TestCaptureHeadlineNeverCallsATruncatedRunComplete(t *testing.T) {
	tests := []struct {
		name       string
		result     captureResult
		want, deny string
	}{
		{
			name:   "complete run",
			result: captureResult{CaptureSuccess: true, RunCompleteness: runComplete, ActivityRecords: 50702},
			want:   "capture complete: 50702 records, 0 dropped",
			deny:   "TRUNCATED",
		},
		{
			name:   "the probe capture",
			result: captureResult{CaptureSuccess: true, RunCompleteness: runPartial, ActivityRecords: 50702, DroppedRecords: 28535365},
			want:   "capture TRUNCATED: 50702 records retained, 28535365 dropped",
			deny:   "capture complete",
		},
		{
			name: "loss breakdown reported when the producer supplied it",
			result: captureResult{CaptureSuccess: true, RunCompleteness: runPartial, ActivityRecords: 100, DroppedRecords: 900,
				RecordLoss: &captureLoss{Queue: 900, QueueCapacity: 900, Offered: 1000, Retained: 100}},
			want: "retained 10.00% of the 1000 records the producer offered",
			deny: "capture complete",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := captureHeadline(tt.result)
			if !strings.Contains(got, tt.want) {
				t.Errorf("headline = %q, want it to contain %q", got, tt.want)
			}
			if strings.Contains(got, tt.deny) {
				t.Errorf("headline = %q, want it not to contain %q", got, tt.deny)
			}
		})
	}
}

// TestObserveJournalKeepsLossBreakdown pins that the breakdown the agent
// already writes into its finalized record reaches the capture. Before this it
// was decoded and discarded, leaving one undifferentiated drop total.
func TestObserveJournalKeepsLossBreakdown(t *testing.T) {
	const session = "00112233445566778899aabbccddeeff"
	dir := t.TempDir()
	var sid [16]byte
	raw, _ := hex.DecodeString(session)
	copy(sid[:], raw)
	writeHandshake(t, dir, session, 41, 2)
	finalized := `{"state":"collected","dropped":900,"native_dropped":100,"queue_dropped":800,` +
		`"queue_oversize_dropped":0,"queue_contention_dropped":50,"queue_capacity_dropped":750,` +
		`"offered":1000,"retained":100}`
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
		{Kind: recordFinalized, Payload: []byte(finalized)},
	})

	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.RecordLoss == nil {
		t.Fatalf("capture kept no loss breakdown: %+v", got)
	}
	want := captureLoss{Native: 100, Queue: 800, QueueContention: 50, QueueCapacity: 750, Offered: 1000, Retained: 100}
	if *got.RecordLoss != want {
		t.Errorf("RecordLoss = %+v, want %+v", *got.RecordLoss, want)
	}
	// The run lost records, so it cannot be complete however the boolean reads.
	if got.RunCompleteness != runPartial {
		t.Errorf("run_completeness = %q, want %q", got.RunCompleteness, runPartial)
	}
	if !strings.Contains(captureHeadline(got), "TRUNCATED") {
		t.Errorf("headline called this run complete: %q", captureHeadline(got))
	}
}

// TestObserveJournalLossBreakdownAbsent is the negative control: a producer
// that reports no breakdown must publish no field, not a row of zeros that
// reads like a measurement of zero loss.
func TestObserveJournalLossBreakdownAbsent(t *testing.T) {
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
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.RecordLoss != nil {
		t.Errorf("RecordLoss = %+v, want nil for a producer that reported none", *got.RecordLoss)
	}
	if got.RunCompleteness != runComplete {
		t.Errorf("run_completeness = %q, want %q", got.RunCompleteness, runComplete)
	}
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "record_loss") {
		t.Errorf("status published an empty record_loss: %s", data)
	}
}

// TestVolumeRemedyFollowsTheCapability pins that the one lever worth naming is
// offered only where it exists. API attribution was 92% of the records in the
// journals the agent was measured against and 87% in the ds4 captures, so
// dropping it is the difference between a truncated capture and a complete one
// -- but telling a run that already declined it to decline it again is noise
// standing where an actionable sentence should be.
func TestVolumeRemedyFollowsTheCapability(t *testing.T) {
	tests := []struct {
		name       string
		capability string
		want, deny string
	}{
		{"api attribution on", `{"kind":"capability","api_records":true}`, "CUDAPROF_API_ACTIVITY=0", "already collected no API"},
		{"promises smaller inputs, not fewer drops", `{"api_records":true}`, "not fewer drops", "will fix"},
		{"api attribution off", `{"kind":"capability","api_records":false}`, "already collected no API attribution", "CUDAPROF_API_ACTIVITY=0"},
		{"no capability block", "", "CUDAPROF_API_ACTIVITY=0", "already collected no API"},
		{"capability without the field", `{"kind":"capability"}`, "CUDAPROF_API_ACTIVITY=0", "already collected no API"},
		{"unparseable capability", `not json`, "CUDAPROF_API_ACTIVITY=0", "already collected no API"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := volumeRemedy(json.RawMessage(tt.capability))
			if !strings.Contains(got, tt.want) {
				t.Errorf("volumeRemedy = %q, want it to contain %q", got, tt.want)
			}
			if strings.Contains(got, tt.deny) {
				t.Errorf("volumeRemedy = %q, want it not to contain %q", got, tt.deny)
			}
		})
	}
}

// TestTruncationMessageSeparatesTheTwoDropMechanisms pins the distinction a
// caller cannot otherwise draw. Exceeding --max-records is a reader-side
// refusal that names that flag as its remedy and fails the capture outright;
// the producer dropping records under queue pressure is a different subsystem
// the flag cannot reach. One flag name appearing in both places, meaning
// opposite things, is what makes the silent case look recoverable.
func TestTruncationMessageSeparatesTheTwoDropMechanisms(t *testing.T) {
	status := probeStatus()
	status["record_loss"] = map[string]any{"queue_dropped": 28535365, "queue_capacity_dropped": 28535365, "offered_records": 28586067, "retained_records": 50702}
	status["capability"] = json.RawMessage(`{"kind":"capability","api_records":true,"queue_capacity":2048,"drain_batch":256}`)
	cap := testCapture(map[string][]byte{"capture.journal": uniformJournal()}, status)
	_, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
	if err == nil {
		t.Fatal("analyzeGaps accepted the probe capture")
	}
	for _, want := range []string{
		"pending ring filling up",
		"CUDAPROF_API_ACTIVITY=0",
		"--max-records is not the remedy here",
		"fails outright",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal does not say %q: %v", want, err)
		}
	}
}

// TestCaptureHeadlineNamesRequireComplete pins that the default path tells a
// caller how to make truncation fail the capture itself. --require-complete
// already does this correctly; nothing on the default path said it existed,
// which is how a driver script read exit code 0 and reported success on a
// capture that retained 0.18% of its records.
func TestCaptureHeadlineNamesRequireComplete(t *testing.T) {
	got := captureHeadline(captureResult{
		CaptureSuccess:  true,
		RunCompleteness: runPartial,
		ActivityRecords: 50702,
		DroppedRecords:  28535365,
		Capability:      json.RawMessage(`{"api_records":true}`),
		RecordLoss:      &captureLoss{Queue: 28535365, QueueCapacity: 28535365, Offered: 28586067, Retained: 50702},
	})
	for _, want := range []string{
		"capture TRUNCATED",
		"retained 0.18%",
		"CUDAPROF_API_ACTIVITY=0",
		"--require-complete",
		"--max-records does not apply",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("headline does not say %q:\n%s", want, got)
		}
	}
}

// TestBudgetRemedyNamesTheFlag pins the third refusal's remedy. This is the one
// wall of the three with an available fix -- nothing was lost, the journal just
// did not fit -- so the refusal that names no flag wastes the only case where
// advice is possible.
func TestBudgetRemedyNamesTheFlag(t *testing.T) {
	journal := "worker-1-capture.journal"
	tests := []struct {
		name       string
		omitted    []captureOmittedFile
		want, deny []string
	}{
		{
			name:    "per-file budget",
			omitted: []captureOmittedFile{{Name: journal, Bytes: 8750387720, Reason: "compressed journal exceeds capture file budget of 67108864 bytes"}},
			want:    []string{"--capture-file-bytes", "67108864", "at most 8750387720"},
			deny:    []string{"--capture-total-bytes"},
		},
		{
			name:    "whole-artifact budget",
			omitted: []captureOmittedFile{{Name: journal, Bytes: 300000000, Reason: "exceeds capture total budget of 268435456 bytes"}},
			want:    []string{"--capture-total-bytes", "268435456"},
			deny:    []string{"--capture-file-bytes"},
		},
		{
			name: "both budgets",
			omitted: []captureOmittedFile{
				{Name: journal, Bytes: 8750387720, Reason: "compressed journal exceeds capture file budget of 67108864 bytes"},
				{Name: "worker-2-capture.journal", Bytes: 300000000, Reason: "exceeds capture total budget of 268435456 bytes"},
			},
			want: []string{"--capture-file-bytes", "--capture-total-bytes"},
		},
		{
			name:    "nothing omitted",
			omitted: nil,
		},
		{
			// A non-journal omission is not what makes a capture unusable for
			// gap analysis, so it must not draw a journal remedy.
			name:    "omission that is not a journal",
			omitted: []captureOmittedFile{{Name: "kernels.cubin", Bytes: 999, Reason: "exceeds capture file budget of 67108864 bytes"}},
			deny:    []string{"--capture-file-bytes", "--capture-total-bytes"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := budgetRemedy(tt.omitted)
			if len(tt.want) == 0 && len(tt.deny) == 0 && got != "" {
				t.Fatalf("budgetRemedy = %q, want empty", got)
			}
			for _, w := range tt.want {
				if !strings.Contains(got, w) {
					t.Errorf("budgetRemedy = %q, want it to contain %q", got, w)
				}
			}
			for _, d := range tt.deny {
				if strings.Contains(got, d) {
					t.Errorf("budgetRemedy = %q, want it not to contain %q", got, d)
				}
			}
		})
	}
}

// TestBudgetRemedyStatesTheSizeAsACeiling pins the distinction that makes the
// advice correct. CompressJournal abandons the encode as soon as it passes the
// limit, so the compressed size is never measured and omitted_files reports the
// raw file. Presenting that figure as the value to set would overshoot by the
// whole compression ratio, on a workload that writes 8.75 GB a run.
func TestBudgetRemedyStatesTheSizeAsACeiling(t *testing.T) {
	got := budgetRemedy([]captureOmittedFile{{
		Name:   "worker-1-capture.journal",
		Bytes:  8750387720,
		Reason: "compressed journal exceeds capture file budget of 67108864 bytes",
	}})
	if !strings.Contains(got, "at most 8750387720") {
		t.Errorf("remedy does not bound the value: %q", got)
	}
	if !strings.Contains(got, "compressed") {
		t.Errorf("remedy does not say the budget applies to the compressed member: %q", got)
	}
	if strings.Contains(got, "set --capture-file-bytes to 8750387720") {
		t.Errorf("remedy names the uncompressed size as the target: %q", got)
	}
}

// TestGapsOmissionRefusalNamesTheRemedy is the end-to-end path the lead hit:
// a capture that dropped nothing and is complete, yet is unusable for gaps
// because the journal never fit the artifact.
func TestGapsOmissionRefusalNamesTheRemedy(t *testing.T) {
	status := map[string]any{
		"capture_success":  true,
		"outcome":          "collected",
		"run_completeness": "complete",
		"dropped_records":  0,
		"detailed_records": "projected into gpu.pprof",
		"omitted_files": []any{map[string]any{
			"name":   "worker-1-capture.journal",
			"bytes":  8750387720,
			"sha256": "6f5e4d28",
			"reason": "compressed journal exceeds capture file budget of 67108864 bytes",
		}},
	}
	cap := testCapture(map[string][]byte{}, status)
	_, err := analyzeGaps(cap, gapOptions{warmupMS: 200, maxRecords: controllerMaxJournalRecords})
	if err == nil {
		t.Fatal("analyzeGaps accepted a capture with no journal member")
	}
	for _, want := range []string{"8750387720", "6f5e4d28", "not the pprof projection", "--capture-file-bytes"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal does not say %q: %v", want, err)
		}
	}
}

// TestCaptureHelpDocumentsTheBudgetFlags pins that a user who reads the help
// can find the flags that fix the refusal above. Both were parsed and neither
// was documented, so the remedy for the third wall was unreachable from
// anything the tool printed.
func TestCaptureHelpDocumentsTheBudgetFlags(t *testing.T) {
	help := captureHelp()
	for _, want := range []string{
		"--capture-file-bytes",
		"--capture-total-bytes",
		fmt.Sprintf("%d", cudacapture.MaxCaptureFileBytes),
		fmt.Sprintf("%d", cudacapture.MaxCaptureTotalBytes),
		"bounds decoding only",
	} {
		if !strings.Contains(help, want) {
			t.Errorf("capture help omits %q:\n%s", want, help)
		}
	}
	// The defaults differ; documenting one value for both would be wrong.
	if cudacapture.MaxCaptureFileBytes == cudacapture.MaxCaptureTotalBytes {
		t.Fatal("this test cannot tell the two budgets apart if they are equal")
	}
}

// TestInspectWarningNamesTheRemedy pins the other path that refuses on an
// omitted journal. gaps, gaps --diff_base and diff --explain all reach the
// refusal through analyzeGaps, so one message serves three; inspect warns on
// its own and had to be told separately.
func TestInspectWarningNamesTheRemedy(t *testing.T) {
	omitted := []captureOmittedFile{{
		Name:   "worker-1-capture.journal",
		Bytes:  8750387720,
		Reason: "compressed journal exceeds capture file budget of 67108864 bytes",
	}}
	if got := missingJournals(omitted); len(got) != 1 || got[0] != "worker-1-capture.journal" {
		t.Fatalf("missingJournals = %v, want the one journal", got)
	}
	if remedy := budgetRemedy(omitted); !strings.Contains(remedy, "--capture-file-bytes") {
		t.Errorf("inspect warning would name no remedy: %q", remedy)
	}
}

// TestCaptureHeadlineFlagsAnOmittedJournalOnACompleteRun is the third wall:
// a run that dropped nothing, finalized cleanly and reports complete, whose
// journal never fit the artifact. Every count says healthy and the capture
// cannot answer a single question about the device timeline.
func TestCaptureHeadlineFlagsAnOmittedJournalOnACompleteRun(t *testing.T) {
	result := captureResult{
		CaptureSuccess:  true,
		RunCompleteness: runComplete,
		ActivityRecords: 3726074,
		OmittedFiles: []captureOmittedFile{{
			Name:   "worker-1-capture.journal",
			Bytes:  8750387720,
			SHA256: "6f5e4d28",
			Reason: "compressed journal exceeds capture file budget of 67108864 bytes",
		}},
	}
	got := captureHeadline(result)
	for _, want := range []string{"NOT attached", "worker-1-capture.journal", "gaps will refuse it", "--capture-file-bytes"} {
		if !strings.Contains(got, want) {
			t.Errorf("headline does not say %q:\n%s", want, got)
		}
	}
	// A complete run with its journal attached must stay quiet, or the warning
	// means nothing.
	result.OmittedFiles = nil
	if quiet := captureHeadline(result); strings.Contains(quiet, "NOT attached") {
		t.Errorf("headline warns about an omission that did not happen: %s", quiet)
	}
}

// TestDetailedRecordsDispositionStaysCompact pins that the remedy is kept out
// of the embedded status. That string is generated, always written, and
// governed by the same per-file budget it would be describing, so advice
// belongs on the surfaces a person reads, not in the artifact.
func TestDetailedRecordsDispositionStaysCompact(t *testing.T) {
	got := detailedRecordsDisposition(captureResult{OmittedFiles: []captureOmittedFile{{
		Name:   "worker-1-capture.journal",
		Bytes:  8750387720,
		Reason: "compressed journal exceeds capture file budget of 67108864 bytes",
	}}})
	if !strings.Contains(got, "exceeded the artifact budget") {
		t.Errorf("disposition does not report the omission: %q", got)
	}
	if strings.Contains(got, "--capture-file-bytes") {
		t.Errorf("disposition carries the remedy, which inflates a budget-governed field: %q", got)
	}
}

// TestVolumeRemedyClaimsOnlyVolume pins the boundary between what is measured
// and what is not. API attribution really is ~87-92% of records, so naming the
// lever is sound; that cutting volume reduces drops is a separate claim, and
// six runs refute it -- an eightfold volume cut left retention between 1% and
// 77%, a spread wider than the gap between the configurations. The mechanism
// agrees: a capacity drop is an admission that waited out the deadline, and
// that wait is bounded by journal write latency, not by record count.
func TestVolumeRemedyClaimsOnlyVolume(t *testing.T) {
	got := volumeRemedy(json.RawMessage(`{"api_records":true}`))
	for _, want := range []string{"CUDAPROF_API_ACTIVITY=0", "order of magnitude smaller", "not fewer drops", "journal write latency"} {
		if !strings.Contains(got, want) {
			t.Errorf("volumeRemedy = %q, want it to contain %q", got, want)
		}
	}
	// Any phrasing that promises the drops go away is the claim we cannot make.
	for _, deny := range []string{"will fix", "resolves the drops", "eliminates", "prevents the drops"} {
		if strings.Contains(got, deny) {
			t.Errorf("volumeRemedy promises a result that is not measured (%q): %q", deny, got)
		}
	}
}

// TestWriteLatencyNoteAttributesACapacityDrop pins that a capacity drop reports
// its cause. The agent records the longest admission wait and the longest batch
// write for exactly this, and nothing read them: without the pair, a capacity
// drop is a count with no mechanism and every remedy is a guess.
func TestWriteLatencyNoteAttributesACapacityDrop(t *testing.T) {
	tests := []struct {
		name       string
		loss       captureLoss
		want, deny string
	}{
		{
			name: "wait reached the admission deadline",
			loss: captureLoss{BackpressureMaxNanos: 100 * 1000 * 1000, DrainWriteMaxNanos: 99 * 1000 * 1000},
			want: "the journal writes, not the record count, are what lost these",
		},
		{
			name: "waits well inside the deadline",
			loss: captureLoss{BackpressureMaxNanos: 5 * 1000 * 1000, DrainWriteMaxNanos: 4 * 1000 * 1000},
			want: "longest admission wait 5ms",
			deny: "not the record count",
		},
		{
			name: "producer reported no timings",
			loss: captureLoss{QueueCapacity: 900},
			want: "",
			deny: "longest admission wait",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.loss.writeLatencyNote()
			if tt.want == "" && got != "" {
				t.Fatalf("writeLatencyNote = %q, want empty", got)
			}
			if tt.want != "" && !strings.Contains(got, tt.want) {
				t.Errorf("writeLatencyNote = %q, want it to contain %q", got, tt.want)
			}
			if tt.deny != "" && strings.Contains(got, tt.deny) {
				t.Errorf("writeLatencyNote = %q, want it not to contain %q", got, tt.deny)
			}
		})
	}
}
