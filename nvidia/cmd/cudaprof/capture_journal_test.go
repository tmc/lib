package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// cdw1Journal writes a representative CDW1 journal: bootstrap, capability, n
// activity records with the payload shape the agent emits, and a finalized
// record. It returns the journal's size on disk.
func cdw1Journal(t *testing.T, path string, n int) int64 {
	t.Helper()
	var sid [16]byte
	copy(sid[:], "0123456789abcdef")
	records := []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
	}
	for i := 0; i < n; i++ {
		records = append(records, wire.Record{
			Kind: recordActivity,
			Payload: []byte(fmt.Sprintf(`{"kind":"activity","subtype":"kernel","name":"ampere_bf16_gemm_128x128_tn","valid":true,"start":%d,"end":%d,"device":0,"stream":%d,"correlation":%d,"grid":[%d,1,1],"block":[256,1,1],"registers_per_thread":64,"dynamic_shared_memory":16384}`,
				1000000+i*4200, 1000000+i*4200+3100, i%4, i, 132+i%16)),
		})
	}
	records = append(records, wire.Record{Kind: recordFinalized, Payload: []byte(fmt.Sprintf(`{"state":"collected","activity":%d}`, n))})
	appendRecords(t, path, sid, 4242, 1, records)
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return fi.Size()
}

// TestCaptureJournalIsCompressed is the F1 control for the measured defect: a
// long run's journal exceeded the per-file budget and was dropped from the
// bundle while the status still read complete. The journal is now embedded
// compressed, which is what keeps it inside the budget, and it must still read
// back byte for byte.
func TestCaptureJournalIsCompressed(t *testing.T) {
	dir := t.TempDir()
	journalPath := filepath.Join(dir, "capture.journal")
	rawSize := cdw1Journal(t, journalPath, 20000)
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	c, status := readWrittenCapture(t, out)
	if len(status.OmittedFiles) != 0 {
		t.Fatalf("omitted %+v", status.OmittedFiles)
	}
	if _, ok := c.Files["capture.journal"]; ok {
		t.Fatal("journal was embedded uncompressed")
	}
	stored, ok := c.Files["capture.journal.gz"]
	if !ok {
		t.Fatalf("no compressed journal member: %v", c.JournalNames())
	}
	ratio := float64(len(stored)) / float64(rawSize)
	t.Logf("CDW1 journal: %d bytes raw, %d bytes gzip, ratio %.4f (%.1f%% saved)", rawSize, len(stored), ratio, 100*(1-ratio))
	// Measured on this fixture, not taken on report: a CDW1 journal of JSON
	// activity payloads compresses well past an order of magnitude. The bound
	// is loose enough that ordinary variation does not trip it, and tight
	// enough that a member stored uncompressed fails it.
	if ratio > 0.15 {
		t.Fatalf("journal compressed to %.4f of its size, want at most 0.15", ratio)
	}
	raw, err := c.Journal("capture.journal.gz")
	if err != nil {
		t.Fatal(err)
	}
	onDisk, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, onDisk) {
		t.Fatalf("journal did not round-trip: %d bytes read, %d on disk", len(raw), len(onDisk))
	}
}

// TestCaptureJournalCompressionFitsWhatPlainWouldNot pins the behavior that
// motivated the change: a journal larger than the per-file budget is now
// attached rather than omitted, because the budget applies to the compressed
// member.
func TestCaptureJournalCompressionFitsWhatPlainWouldNot(t *testing.T) {
	dir := t.TempDir()
	rawSize := cdw1Journal(t, filepath.Join(dir, "capture.journal"), 20000)
	budget := captureArtifactBudget{FileBytes: rawSize / 2}
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}
	if err := writeCaptureStatus(out, result, budget); err != nil {
		t.Fatal(err)
	}
	c, status := readWrittenCapture(t, out)
	if len(status.OmittedFiles) != 0 {
		t.Fatalf("journal was omitted under a budget its compressed form fits: %+v", status.OmittedFiles)
	}
	if _, ok := c.Files["capture.journal.gz"]; !ok {
		t.Fatalf("journal members = %v", c.JournalNames())
	}
}

// inspectCapturePath runs cudaprof inspect over path and returns the report
// together with what the command wrote to stderr.
func inspectCapturePath(t *testing.T, path string) (inspectReport, string) {
	t.Helper()
	outR, outW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	errR, errW, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	origOut, origErr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = outW, errW
	runErr := runInspectCommand([]string{"--coverage", path})
	outW.Close()
	errW.Close()
	os.Stdout, os.Stderr = origOut, origErr
	var outBuf, errBuf bytes.Buffer
	io.Copy(&outBuf, outR)
	io.Copy(&errBuf, errR)
	if runErr != nil {
		t.Fatalf("inspect: %v", runErr)
	}
	var report inspectReport
	if err := json.Unmarshal(outBuf.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal inspect report: %v, raw:\n%s", err, outBuf.String())
	}
	return report, errBuf.String()
}

// TestInspectFlagsOmittedJournal is the F2 control: a capture published
// without its journal carries no raw records at all, and every other field in
// the inspection reads healthy. The verdict and the warning must move with the
// omission, and the run verdict must not: the run was observed in full.
func TestInspectFlagsOmittedJournal(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "capture.journal"), incompressibleJournal(64<<10), 0600); err != nil {
		t.Fatal(err)
	}
	result := captureResult{
		Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeCollected,
		CaptureSuccess: true, TransportReady: true, BootstrapSeen: true, CollectorFinalized: true,
		RunCompleteness: runComplete, ScratchDir: dir,
	}

	tests := []struct {
		name                string
		budget              captureArtifactBudget
		wantCompleteness    string
		wantMissingJournals int
		wantWarningOnStderr bool
	}{
		{name: "journal attached", budget: captureArtifactBudget{}, wantCompleteness: artifactComplete},
		{name: "journal omitted", budget: captureArtifactBudget{FileBytes: 4096}, wantCompleteness: artifactPartial, wantMissingJournals: 1, wantWarningOnStderr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := filepath.Join(t.TempDir(), "capture.pprof")
			if err := writeCaptureStatus(out, result, tt.budget); err != nil {
				t.Fatal(err)
			}
			report, stderr := inspectCapturePath(t, out)
			if report.Coverage == nil {
				t.Fatal("inspect published no coverage")
			}
			if got := report.Coverage.ArtifactCompleteness; got != tt.wantCompleteness {
				t.Fatalf("artifact_completeness = %q, want %q", got, tt.wantCompleteness)
			}
			if got := len(report.Coverage.MissingJournals); got != tt.wantMissingJournals {
				t.Fatalf("missing_capture_journals = %v", report.Coverage.MissingJournals)
			}
			if tt.wantMissingJournals != 0 && report.Coverage.MissingJournals[0] != "capture.journal" {
				t.Fatalf("missing journal named %q", report.Coverage.MissingJournals[0])
			}
			if warned := strings.Contains(stderr, "artifact_completeness is partial"); warned != tt.wantWarningOnStderr {
				t.Fatalf("stderr warning = %v, want %v; stderr:\n%s", warned, tt.wantWarningOnStderr, stderr)
			}
			// The run verdict states what the capture observed, and the
			// omission did not change that.
			if report.Coverage.RunCompleteness != runComplete {
				t.Fatalf("run_completeness = %q, want %q", report.Coverage.RunCompleteness, runComplete)
			}
		})
	}
}

// TestInspectOmittedJournalNamesSurviveCompression pins that an omitted
// journal is still reported under its name on disk, with the raw size and
// digest, rather than under the compressed member name it never got.
func TestInspectOmittedJournalNamesSurviveCompression(t *testing.T) {
	dir := t.TempDir()
	data := incompressibleJournal(64 << 10)
	if err := os.WriteFile(filepath.Join(dir, "capture.journal"), data, 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeCollected, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{FileBytes: 4096}); err != nil {
		t.Fatal(err)
	}
	_, status := readWrittenCapture(t, out)
	if len(status.OmittedFiles) != 1 {
		t.Fatalf("omissions = %+v", status.OmittedFiles)
	}
	got := status.OmittedFiles[0]
	if got.Name != "capture.journal" || got.Bytes != int64(len(data)) {
		t.Fatalf("omission = %+v, want the raw name and %d bytes", got, len(data))
	}
	if !strings.Contains(got.Reason, "compressed journal exceeds") {
		t.Fatalf("reason = %q, want the compressed-size reason", got.Reason)
	}
	if !cudacapture.IsJournalName(got.Name) {
		t.Fatalf("omitted name %q is not recognized as a journal", got.Name)
	}
}
