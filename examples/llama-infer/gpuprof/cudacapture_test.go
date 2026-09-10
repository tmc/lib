package gpuprof

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// cudaprofProfile builds a profile shaped like the one cudaprof writes: four
// sample types, a kernel record carrying a name, and a memcpy record that
// carries none.
func cudaprofProfile() *profile.Profile {
	return &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "kernel_time", Unit: "nanoseconds"},
			{Type: "copy_time", Unit: "nanoseconds"},
			{Type: "copy_bytes", Unit: "bytes"},
			{Type: "activity", Unit: "count"},
		},
		Sample: []*profile.Sample{
			{
				Value: []int64{2272, 0, 0, 1},
				Label: map[string][]string{"subtype": {"kernel"}, "name": {"RMSNorm"}, "device": {"0"}},
			},
			{
				Value: []int64{4096, 0, 0, 1},
				Label: map[string][]string{"subtype": {"kernel"}, "name": {"LinearGEMV"}, "device": {"1"}},
			},
			{
				Value: []int64{1500, 0, 0, 1},
				Label: map[string][]string{"subtype": {"kernel"}, "name": {"RMSNorm"}, "device": {"0"}},
			},
			{
				Value: []int64{0, 500, 65536, 1},
				Label: map[string][]string{"subtype": {"memcpy"}, "device": {"0"}},
			},
		},
	}
}

// unavailableCode marks every code role explicitly absent, which is what a
// capture that retained no source, PTX, CUBIN, or SASS must record.
func unavailableCode() map[string]string {
	return map[string]string{
		"source": "not captured",
		"ptx":    "not captured",
		"cubin":  "not captured",
		"sass":   "not captured",
	}
}

func testCapture(p *profile.Profile) *cudacapture.Capture {
	return &cudacapture.Capture{
		Profile:         p,
		Files:           map[string][]byte{"run.json": []byte(`{"ok":true}`)},
		CodeUnavailable: unavailableCode(),
	}
}

func writeTestCapture(t *testing.T, p *profile.Profile) string {
	t.Helper()
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, testCapture(p)); err != nil {
		t.Fatalf("write capture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "capture.pprof")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	return path
}

func TestReadCaptureSummary(t *testing.T) {
	got, err := ReadCaptureSummary(writeTestCapture(t, cudaprofProfile()))
	if err != nil {
		t.Fatalf("ReadCaptureSummary: %v", err)
	}
	if want := int64(2272 + 4096 + 1500); got.KernelTimeNS != want {
		t.Errorf("KernelTimeNS = %d, want %d", got.KernelTimeNS, want)
	}
	if got.CopyTimeNS != 500 {
		t.Errorf("CopyTimeNS = %d, want 500", got.CopyTimeNS)
	}
	if got.CopyBytes != 65536 {
		t.Errorf("CopyBytes = %d, want 65536", got.CopyBytes)
	}
	if got.Activities != 4 {
		t.Errorf("Activities = %d, want 4", got.Activities)
	}
	if got.KernelLaunches != 3 {
		t.Errorf("KernelLaunches = %d, want 3", got.KernelLaunches)
	}
	if want := []string{"LinearGEMV", "RMSNorm"}; !equalStrings(got.KernelNames, want) {
		t.Errorf("KernelNames = %v, want %v", got.KernelNames, want)
	}
	if want := []string{"0", "1"}; !equalStrings(got.Devices, want) {
		t.Errorf("Devices = %v, want %v", got.Devices, want)
	}
}

// TestReadCaptureSummaryDirectory covers the directory form, which is what an
// extracted capture looks like on disk.
func TestReadCaptureSummaryDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "extract")
	if err := cudacapture.WriteDir(dir, testCapture(cudaprofProfile())); err != nil {
		t.Fatalf("WriteDir: %v", err)
	}
	got, err := ReadCaptureSummary(dir)
	if err != nil {
		t.Fatalf("ReadCaptureSummary: %v", err)
	}
	if got.KernelLaunches != 3 {
		t.Errorf("KernelLaunches = %d, want 3", got.KernelLaunches)
	}
}

// TestSummarizeCaptureRejectsForeignProfile is the control against reporting a
// fabricated zero: a profile with none of the cudaprof sample types must fail
// rather than summarize as an idle run.
func TestSummarizeCaptureRejectsForeignProfile(t *testing.T) {
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
		Sample:     []*profile.Sample{{Value: []int64{7}}},
	}
	got, err := summarizeCapture(testCapture(p))
	if err == nil {
		t.Fatalf("summarizeCapture succeeded with %+v, want error", got)
	}
	if !strings.Contains(err.Error(), "no cudaprof sample types") {
		t.Errorf("error = %v, want it to name the missing sample types", err)
	}
}

func TestSummarizeCaptureRejectsMissingProfile(t *testing.T) {
	if _, err := summarizeCapture(nil); err == nil {
		t.Error("summarizeCapture(nil) succeeded, want error")
	}
	if _, err := summarizeCapture(&cudacapture.Capture{}); err == nil {
		t.Error("summarizeCapture with no profile succeeded, want error")
	}
}

func TestReadCaptureSummaryMissingPath(t *testing.T) {
	if _, err := ReadCaptureSummary(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Error("ReadCaptureSummary on a missing path succeeded, want error")
	}
}

// TestSummarizePartialSampleTypes covers a capture that carries only some of
// the cudaprof sample types: the present ones are summed and the absent ones
// stay zero without an error, because the capture does describe itself.
func TestSummarizePartialSampleTypes(t *testing.T) {
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "activity", Unit: "count"}},
		Sample: []*profile.Sample{
			{Value: []int64{1}, Label: map[string][]string{"subtype": {"kernel"}, "name": {"SwiGLU"}}},
		},
	}
	got, err := summarizeCapture(testCapture(p))
	if err != nil {
		t.Fatalf("summarizeCapture: %v", err)
	}
	if got.Activities != 1 || got.KernelLaunches != 1 {
		t.Errorf("Activities = %d, KernelLaunches = %d, want 1 and 1", got.Activities, got.KernelLaunches)
	}
	if got.KernelTimeNS != 0 {
		t.Errorf("KernelTimeNS = %d, want 0 for a capture with no kernel_time type", got.KernelTimeNS)
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// statusFile wraps a status body the way cudaprof writes it.
func statusFile(body string) map[string][]byte {
	return map[string][]byte{captureStatusFile: []byte(`{"status":` + body + `}`)}
}

func TestQualificationAbsentStatusIsNotComplete(t *testing.T) {
	q, err := readQualification(map[string][]byte{"other.json": []byte(`{}`)})
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.Present {
		t.Error("Present is true for a capture with no status document")
	}
	if err := q.Qualify(); err == nil {
		t.Error("Qualify succeeded with no status document, want error")
	}
}

func TestQualificationCompleteRun(t *testing.T) {
	q, err := readQualification(statusFile(`{
		"capture_success": true, "transport_ready": true, "collector_finalized": true,
		"activity_records": 1000, "dropped_records": 0,
		"worker_generations": [
			{"disposition": "finalized"},
			{"disposition": "finalized", "activity_records": 0}
		]}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.RunCompleteness != runComplete {
		t.Errorf("RunCompleteness = %q, want %q", q.RunCompleteness, runComplete)
	}
	if !q.Derived {
		t.Error("Derived is false for a status with no published verdict")
	}
	if err := q.Qualify(); err != nil {
		t.Errorf("Qualify: %v", err)
	}
}

// TestQualificationZeroRecordGenerationIsQualified is the control against
// inferring lost work from a count: a finalized generation that retained
// nothing ran no GPU work and is complete.
func TestQualificationZeroRecordGenerationIsQualified(t *testing.T) {
	q, err := readQualification(statusFile(`{
		"transport_ready": true, "collector_finalized": true,
		"worker_generations": [{"disposition": "finalized", "records": 0, "activity_records": 0}]}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.UnqualifiedGenerations != 0 {
		t.Errorf("UnqualifiedGenerations = %d, want 0", q.UnqualifiedGenerations)
	}
	if err := q.Qualify(); err != nil {
		t.Errorf("Qualify: %v", err)
	}
}

func TestQualificationFailedGenerationIsPartial(t *testing.T) {
	q, err := readQualification(statusFile(`{
		"capture_success": true, "transport_ready": true, "collector_finalized": true,
		"worker_generations": [
			{"disposition": "finalized"},
			{"disposition": "incomplete", "failure": "capture journal missing finalized record"},
			{"disposition": "producer_failure"}
		]}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.RunCompleteness != runPartial {
		t.Errorf("RunCompleteness = %q, want %q", q.RunCompleteness, runPartial)
	}
	if q.UnqualifiedGenerations != 2 {
		t.Errorf("UnqualifiedGenerations = %d, want 2", q.UnqualifiedGenerations)
	}
	err = q.Qualify()
	if err == nil {
		t.Fatal("Qualify succeeded for a partial run, want error")
	}
	if !strings.Contains(err.Error(), "2 of 3 worker generations") {
		t.Errorf("error = %v, want it to count the unqualified generations", err)
	}
}

// TestQualificationSeparatesLossFromCompleteness checks the two axes are
// independent: every generation finalized, and the producer still dropped
// records, so the totals are still not run totals.
func TestQualificationSeparatesLossFromCompleteness(t *testing.T) {
	q, err := readQualification(statusFile(`{
		"transport_ready": true, "collector_finalized": true,
		"activity_records": 4285, "dropped_records": 249418,
		"worker_generations": [{"disposition": "finalized", "records": 47810}]}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.RunCompleteness != runComplete {
		t.Errorf("RunCompleteness = %q, want %q: every generation finalized", q.RunCompleteness, runComplete)
	}
	err = q.Qualify()
	if err == nil {
		t.Fatal("Qualify succeeded for a lossy capture, want error")
	}
	if !strings.Contains(err.Error(), "249418 of 297228") {
		t.Errorf("error = %v, want it to state the offered and dropped counts", err)
	}
}

// TestQualificationLossDenominatorIsAllKinds pins the arithmetic to the
// retained SGLang run. dropped_records counts drops of every admitted kind, so
// the denominator has to be the all-kinds retained total (the sum of the
// per-generation record counts), not activity_records. Dividing 249418 by
// 4285+249418 states a 98.31% activity loss the producer never measured; the
// receipt establishes 47810 retained, 249418 dropped, 297228 offered, 83.91%.
func TestQualificationLossDenominatorIsAllKinds(t *testing.T) {
	q, err := readQualification(statusFile(`{
		"transport_ready": true, "collector_finalized": true,
		"activity_records": 4285, "dropped_records": 249418,
		"worker_generations": [
			{"disposition": "finalized", "records": 20000},
			{"disposition": "finalized", "records": 27810}]}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.RetainedRecords != 47810 {
		t.Fatalf("RetainedRecords = %d, want 47810", q.RetainedRecords)
	}
	err = q.Qualify()
	if err == nil {
		t.Fatal("Qualify succeeded for a lossy capture, want error")
	}
	for _, want := range []string{"249418 of 297228", "83.91%"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %v, want it to contain %q", err, want)
		}
	}
	if strings.Contains(err.Error(), "98.31") {
		t.Errorf("error = %v, must not divide an all-kinds drop count by an activity-only retained count", err)
	}
}

// TestQualificationLossWithoutRetainedTotal checks that a status with no
// per-generation record counts reports the drop count alone. A percentage
// there would need a denominator the receipt does not carry.
func TestQualificationLossWithoutRetainedTotal(t *testing.T) {
	q, err := readQualification(statusFile(`{
		"transport_ready": true, "collector_finalized": true,
		"activity_records": 4285, "dropped_records": 249418,
		"worker_generations": [{"disposition": "finalized"}]}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	err = q.Qualify()
	if err == nil {
		t.Fatal("Qualify succeeded for a lossy capture, want error")
	}
	if !strings.Contains(err.Error(), "dropped 249418 records") || strings.Contains(err.Error(), "%%") {
		t.Errorf("error = %v, want the bare drop count with no loss rate", err)
	}
	if strings.Contains(err.Error(), "of 253703") {
		t.Errorf("error = %v, must not invent an activity-only denominator", err)
	}
}

func TestQualificationUnknownWithoutTransport(t *testing.T) {
	q, err := readQualification(statusFile(`{"transport_ready": false, "collector_finalized": true}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.RunCompleteness != runUnknown {
		t.Errorf("RunCompleteness = %q, want %q", q.RunCompleteness, runUnknown)
	}
}

func TestQualificationFailedWithoutFinalization(t *testing.T) {
	q, err := readQualification(statusFile(`{"transport_ready": true, "collector_finalized": false}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.RunCompleteness != runFailed {
		t.Errorf("RunCompleteness = %q, want %q", q.RunCompleteness, runFailed)
	}
}

// TestQualificationKeepsPublishedVerdict is the control that the derivation
// never overrides the writer. The published verdict disagrees with what the
// dispositions alone would produce, so agreement cannot hide an override.
func TestQualificationKeepsPublishedVerdict(t *testing.T) {
	q, err := readQualification(statusFile(`{
		"transport_ready": true, "collector_finalized": true,
		"run_completeness": "partial", "unqualified_generations": 4,
		"worker_generations": [{"disposition": "finalized"}]}`))
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if q.RunCompleteness != runPartial {
		t.Errorf("RunCompleteness = %q, want the published %q", q.RunCompleteness, runPartial)
	}
	if q.Derived {
		t.Error("Derived is true for a status that published a verdict")
	}
	if q.UnqualifiedGenerations != 4 {
		t.Errorf("UnqualifiedGenerations = %d, want the published 4", q.UnqualifiedGenerations)
	}
}

// TestQualificationBareStatusDocument covers the unwrapped shape.
func TestQualificationBareStatusDocument(t *testing.T) {
	q, err := readQualification(map[string][]byte{
		captureStatusFile: []byte(`{"transport_ready": true, "collector_finalized": true, "outcome": "collected"}`),
	})
	if err != nil {
		t.Fatalf("readQualification: %v", err)
	}
	if !q.Present || q.Outcome != "collected" {
		t.Errorf("qualification = %+v, want a present status with outcome collected", q)
	}
}

func TestQualificationRejectsMalformedStatus(t *testing.T) {
	if _, err := readQualification(map[string][]byte{captureStatusFile: []byte(`not json`)}); err == nil {
		t.Error("readQualification accepted a malformed status document")
	}
}

// TestSummaryCarriesQualification checks the summary reaches its qualification
// through a real write and read.
func TestSummaryCarriesQualification(t *testing.T) {
	c := testCapture(cudaprofProfile())
	c.Files = statusFile(`{
		"transport_ready": true, "collector_finalized": true,
		"activity_records": 3, "dropped_records": 1,
		"worker_generations": [{"disposition": "finalized"}]}`)

	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, c); err != nil {
		t.Fatalf("write capture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "capture.pprof")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}

	got, err := ReadCaptureSummary(path)
	if err != nil {
		t.Fatalf("ReadCaptureSummary: %v", err)
	}
	if !got.Qualification.Present {
		t.Fatal("summary carries no qualification")
	}
	if err := got.Qualification.Qualify(); err == nil {
		t.Error("Qualify succeeded for a capture that dropped a record")
	}
}
