package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// budgetScratch writes a scratch directory holding a small target journal and
// one large worker journal, and returns the directory and the worker journal's
// size and digest.
func budgetScratch(t *testing.T, workerBytes int) (dir string, size int64, sum string) {
	t.Helper()
	dir = t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "capture.journal"), []byte("target journal"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "handshake.json"), []byte(`{"protocol":1}`), 0600); err != nil {
		t.Fatal(err)
	}
	workerDir := filepath.Join(dir, "process-42-3")
	if err := os.MkdirAll(workerDir, 0700); err != nil {
		t.Fatal(err)
	}
	data := incompressibleJournal(workerBytes)
	if err := os.WriteFile(filepath.Join(workerDir, "capture.journal"), data, 0600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return dir, int64(len(data)), hex.EncodeToString(digest[:])
}

// incompressibleJournal returns n deterministic bytes that gzip cannot shrink.
// Journals are embedded compressed, so a size budget only bites in these tests
// if the fixture does not compress away.
func incompressibleJournal(n int) []byte {
	b := make([]byte, n)
	r := rand.New(rand.NewSource(1))
	r.Read(b)
	return b
}

// gzipLen is the embedded size of a journal holding data.
func gzipLen(t *testing.T, data []byte) int64 {
	t.Helper()
	gz, err := cudacapture.CompressJournal(bytes.NewReader(data), 1<<30)
	if err != nil {
		t.Fatal(err)
	}
	return int64(len(gz))
}

func readWrittenCapture(t *testing.T, path string) (*cudacapture.Capture, captureStatus) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := cudacapture.Read(f)
	if err != nil {
		t.Fatal(err)
	}
	var wrapper struct {
		Status captureStatus `json:"status"`
	}
	if err := json.Unmarshal(c.Files["capture-status.json"], &wrapper); err != nil {
		t.Fatal(err)
	}
	return c, wrapper.Status
}

// TestArtifactBudgetOmitsOnlyTheOversizedAttachment is the F3 control: an
// oversized raw journal must not fail a capture whose workload already ran. It
// is dropped from the archive alone, and its identity survives in the status.
func TestArtifactBudgetOmitsOnlyTheOversizedAttachment(t *testing.T) {
	dir, size, sum := budgetScratch(t, 4096)
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}

	if err := writeCaptureStatus(out, result, captureArtifactBudget{FileBytes: 1024}); err != nil {
		t.Fatalf("write with an oversized attachment failed the capture: %v", err)
	}
	c, status := readWrittenCapture(t, out)

	for _, name := range []string{"worker-process-42-3-capture.journal", "worker-process-42-3-capture.journal.gz"} {
		if _, ok := c.Files[name]; ok {
			t.Fatalf("oversized journal was attached as %q", name)
		}
	}
	for _, name := range []string{"capture.journal.gz", "handshake.json"} {
		if _, ok := c.Files[name]; !ok {
			t.Fatalf("within-budget attachment %q was dropped", name)
		}
	}
	if len(status.OmittedFiles) != 1 {
		t.Fatalf("omitted files = %+v", status.OmittedFiles)
	}
	got := status.OmittedFiles[0]
	if got.Name != "worker-process-42-3-capture.journal" || got.Bytes != size || got.SHA256 != sum {
		t.Fatalf("omission disposition = %+v, want size %d sum %s", got, size, sum)
	}
	if got.Reason == "" {
		t.Fatal("omission recorded no reason")
	}
	if !status.CaptureSuccess {
		t.Fatal("omitting an attachment changed the capture outcome")
	}
}

// TestArtifactBudgetAttachesWithinBudget pins that a budget large enough for
// the journal attaches it and records no omission, and that the default budget
// (zero) attaches ordinary artifacts unchanged.
func TestArtifactBudgetAttachesWithinBudget(t *testing.T) {
	dir, size, _ := budgetScratch(t, 4096)
	stored := gzipLen(t, incompressibleJournal(int(size)))
	for _, budget := range []int64{0, stored, stored + 1} {
		out := filepath.Join(t.TempDir(), "capture.pprof")
		result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}
		if err := writeCaptureStatus(out, result, captureArtifactBudget{FileBytes: budget}); err != nil {
			t.Fatalf("budget %d: %v", budget, err)
		}
		c, status := readWrittenCapture(t, out)
		if len(status.OmittedFiles) != 0 {
			t.Fatalf("budget %d recorded omissions %+v", budget, status.OmittedFiles)
		}
		data, ok := c.Files["worker-process-42-3-capture.journal.gz"]
		if !ok || int64(len(data)) != stored {
			t.Fatalf("budget %d attached %d bytes, want %d (present=%v)", budget, len(data), stored, ok)
		}
		raw, err := c.Journal("worker-process-42-3-capture.journal.gz")
		if err != nil {
			t.Fatalf("budget %d: read journal: %v", budget, err)
		}
		if !bytes.Equal(raw, incompressibleJournal(int(size))) {
			t.Fatalf("budget %d: journal did not round-trip: %d bytes", budget, len(raw))
		}
	}
}

// TestArtifactBudgetFlag pins the explicit producer budget on the command line.
func TestArtifactBudgetFlag(t *testing.T) {
	o, err := parseCaptureArgs([]string{"--capture-file-bytes=1048576", "--", "true"})
	if err != nil {
		t.Fatal(err)
	}
	if o.CaptureFileBytes != 1<<20 {
		t.Fatalf("CaptureFileBytes = %d", o.CaptureFileBytes)
	}
	if o, err := parseCaptureArgs([]string{"--", "true"}); err != nil || o.CaptureFileBytes != 0 {
		t.Fatalf("default budget = %d err=%v", o.CaptureFileBytes, err)
	}
	for _, bad := range []string{"--capture-file-bytes=0", "--capture-file-bytes=-1", "--capture-file-bytes=big"} {
		if _, err := parseCaptureArgs([]string{bad, "--", "true"}); err == nil {
			t.Fatalf("%s was accepted", bad)
		}
	}
}

// manyActivities builds a result carrying n decoded activity records.
func manyActivities(n int, scratch string) captureResult {
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeCollected, CaptureSuccess: true, TransportReady: true, TargetPID: 41, ScratchDir: scratch, ActivityRecords: uint64(n)}
	for i := 0; i < n; i++ {
		result.Activities = append(result.Activities, captureActivity{Process: 42, Subtype: "kernel", Name: "kernel", Valid: true, Start: uint64(i*10 + 1), End: uint64(i*10 + 9), Device: uint32(i % 2), Correlation: uint32(i)})
		result.APIRecords = append(result.APIRecords, captureAPI{Domain: "driver", CallbackID: uint32(i), Start: uint64(i), End: uint64(i + 1)})
	}
	return result
}

// TestStatusIsIndependentOfRecordCount is the regression control for the
// measured failure shape: embedding the per-record arrays made the written
// artifact grow to about 1.66x the journal bytes it already held, so a serving
// run whose journals fit the total bound still failed at the write. The status
// must now be bounded by the capture's structure, not by its record count.
func TestStatusIsIndependentOfRecordCount(t *testing.T) {
	dir, _, _ := budgetScratch(t, 100*1024)
	var sizes []int
	for _, n := range []int{100, 20000} {
		out := filepath.Join(t.TempDir(), "capture.pprof")
		// A budget that the record arrays alone would have blown many times
		// over, while the journals themselves fit comfortably.
		budget := captureArtifactBudget{FileBytes: 256 << 10, TotalBytes: 300 << 10}
		if err := writeCaptureStatus(out, manyActivities(n, dir), budget); err != nil {
			t.Fatalf("%d records: write failed: %v", n, err)
		}
		c, status := readWrittenCapture(t, out)
		if len(status.OmittedFiles) != 0 {
			t.Fatalf("%d records: omitted %+v", n, status.OmittedFiles)
		}
		if _, ok := c.Files["worker-process-42-3-capture.journal.gz"]; !ok {
			t.Fatalf("%d records: journal was not attached", n)
		}
		sizes = append(sizes, len(c.Files["capture-status.json"]))
	}
	if sizes[1] > sizes[0]+1024 {
		t.Fatalf("status grew with record count: %d bytes at 100 records, %d at 20000", sizes[0], sizes[1])
	}
	if sizes[1] > 8<<10 {
		t.Fatalf("status is %d bytes, want a small bounded projection", sizes[1])
	}
}

// TestStatusKeepsDispositionsWithoutRecordArrays pins what the compact
// projection must retain, and that the per-record arrays are gone from it
// rather than merely shortened.
func TestStatusKeepsDispositionsWithoutRecordArrays(t *testing.T) {
	dir, _, _ := budgetScratch(t, 1024)
	result := manyActivities(64, dir)
	result.DroppedRecords = 7
	result.RecoveredGenerations, result.RecoveredActivityRecords = 1, 5
	result.WorkerGenerations = []captureWorker{{Process: 42, Generation: 3, Records: 9, ActivityRecords: 5, Disposition: captureWorkerUnfinalizedPartial, Failure: "producer_failure"}}
	result.Capability = []byte(`{"kind":"capability"}`)
	out := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	c, _ := readWrittenCapture(t, out)

	if len(c.Metadata) != 0 {
		t.Fatalf("status duplicated into Capture.Metadata: %d bytes", len(c.Metadata))
	}
	raw := c.Files["capture-status.json"]
	var probe struct {
		Status map[string]json.RawMessage `json:"status"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		t.Fatal(err)
	}
	for _, gone := range []string{"activities", "api_records"} {
		if _, ok := probe.Status[gone]; ok {
			t.Fatalf("compact status still carries %q", gone)
		}
	}
	for _, kept := range []string{"outcome", "capture_success", "session", "target_pid", "activity_records", "dropped_records", "activity_nanos", "copy_bytes", "worker_generations", "capability", "recovered_generations", "activity_samples", "api_record_count", "devices", "activity_kinds", "detailed_records"} {
		if _, ok := probe.Status[kept]; !ok {
			t.Fatalf("compact status dropped %q", kept)
		}
	}
	var envelope struct {
		Status captureStatus `json:"status"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Status.ActivitySamples != 64 || envelope.Status.APIRecordCount != 64 {
		t.Fatalf("counts = %d/%d", envelope.Status.ActivitySamples, envelope.Status.APIRecordCount)
	}
	if len(envelope.Status.Devices) != 2 || envelope.Status.ActivityKinds["kernel"] != 64 {
		t.Fatalf("aggregates = %+v / %+v", envelope.Status.Devices, envelope.Status.ActivityKinds)
	}
	if envelope.Status.DetailedRecords == "" {
		t.Fatal("no detailed-records disposition")
	}
	if len(envelope.Status.WorkerGenerations) != 1 || envelope.Status.WorkerGenerations[0].Disposition != captureWorkerUnfinalizedPartial {
		t.Fatalf("worker dispositions lost: %+v", envelope.Status.WorkerGenerations)
	}
	if len(c.Profile.Sample) != 64 {
		t.Fatalf("profile samples = %d, want 64", len(c.Profile.Sample))
	}
}

// TestLowBudgetStillWritesReadableActivityProfile is 5066's control: a per-file
// budget low enough to omit the worker journals must still produce a capture
// that is self-contained, readable under the DEFAULT reader limits, and
// carrying the activity projection.
func TestLowBudgetStillWritesReadableActivityProfile(t *testing.T) {
	dir, size, sum := budgetScratch(t, 512<<10)
	out := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(out, manyActivities(5000, dir), captureArtifactBudget{FileBytes: 4096}); err != nil {
		t.Fatal(err)
	}
	// Read back with the package defaults, not a raised reader.
	c, status := readWrittenCapture(t, out)
	if len(status.OmittedFiles) != 1 || status.OmittedFiles[0].Bytes != size || status.OmittedFiles[0].SHA256 != sum {
		t.Fatalf("omissions = %+v", status.OmittedFiles)
	}
	if !strings.Contains(status.DetailedRecords, "omitted_files") {
		t.Fatalf("disposition does not point at the omission: %q", status.DetailedRecords)
	}
	if len(c.Profile.Sample) != 5000 {
		t.Fatalf("profile samples = %d, want 5000", len(c.Profile.Sample))
	}
	if status.ActivityRecords != 5000 {
		t.Fatalf("activity_records = %d", status.ActivityRecords)
	}
}

// TestTotalBudgetOmitsLargestFirst pins that a total budget too small for every
// journal drops the largest ones, records each omission, and still writes.
func TestTotalBudgetOmitsLargestFirst(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "capture.journal"), bytes.Repeat([]byte{'t'}, 4096), 0600); err != nil {
		t.Fatal(err)
	}
	for i, size := range []int{64 << 10, 8 << 10} {
		workerDir := filepath.Join(dir, fmt.Sprintf("process-%d-1", 50+i))
		if err := os.MkdirAll(workerDir, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(workerDir, "capture.journal"), incompressibleJournal(size), 0600); err != nil {
			t.Fatal(err)
		}
	}
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 32 << 10}); err != nil {
		t.Fatalf("total budget failed the write: %v", err)
	}
	c, status := readWrittenCapture(t, out)
	if len(status.OmittedFiles) != 1 || status.OmittedFiles[0].Name != "worker-process-50-1-capture.journal" {
		t.Fatalf("omissions = %+v", status.OmittedFiles)
	}
	// The omission names the journal as it exists on disk, at its own size,
	// not the compressed member it would have been embedded as.
	if status.OmittedFiles[0].Bytes != 64<<10 {
		t.Fatalf("omission reports %d bytes, want the raw journal size %d", status.OmittedFiles[0].Bytes, 64<<10)
	}
	if !strings.Contains(status.OmittedFiles[0].Reason, "total budget") {
		t.Fatalf("reason = %q", status.OmittedFiles[0].Reason)
	}
	if _, ok := c.Files["worker-process-51-1-capture.journal.gz"]; !ok {
		t.Fatal("the smaller journal was dropped too")
	}
	var total int64
	for _, data := range c.Files {
		total += int64(len(data))
	}
	if total > 32<<10 {
		t.Fatalf("written files total %d bytes, over the budget", total)
	}
}

// TestTotalBudgetOmitsModulesBeforeJournals pins that a module image goes
// before a larger journal: the journal holds the records every analysis reads.
func TestTotalBudgetOmitsModulesBeforeJournals(t *testing.T) {
	dir := t.TempDir()
	workerDir := filepath.Join(dir, "process-50-1")
	if err := os.MkdirAll(filepath.Join(workerDir, "modules"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workerDir, "capture.journal"), incompressibleJournal(24<<10), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workerDir, "modules", "module-0000-id1.cubin"), bytes.Repeat([]byte{'m'}, 16<<10), 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "capture.pprof")
	result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}
	if err := writeCaptureStatus(out, result, captureArtifactBudget{FileBytes: 1 << 20, TotalBytes: 32 << 10}); err != nil {
		t.Fatalf("total budget failed the write: %v", err)
	}
	c, status := readWrittenCapture(t, out)
	if len(status.OmittedFiles) != 1 || status.OmittedFiles[0].Name != "worker-process-50-1-module-0000-id1.cubin" {
		t.Fatalf("omissions = %+v", status.OmittedFiles)
	}
	if _, ok := c.Files["worker-process-50-1-capture.journal.gz"]; !ok {
		t.Fatal("the journal was dropped")
	}
}

// TestReadLimitsAreExplicit pins that the read side honors an explicit bound,
// even below what the capture records. The raise direction is pinned in
// x/cudacapture/limits_test.go, where it does not cost a 64 MiB fixture; here
// the mirror image is enough: a tightened reader rejects what the default
// reader accepts, so the flag is really in force.
func TestReadLimitsAreExplicit(t *testing.T) {
	dir, _, _ := budgetScratch(t, 8192)
	out := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(out, manyActivities(16, dir), captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	if _, err := readCapturePath(out); err != nil {
		t.Fatalf("default read: %v", err)
	}
	if _, err := readCapturePathLimits(out, cudacapture.Limits{FileBytes: 128}); err == nil {
		t.Fatal("tightened read accepted an oversized file")
	}

	rest, limits, err := parseReadLimits([]string{"--capture-file-bytes=1048576", "--capture-total-bytes=536870912", "--coverage", out})
	if err != nil {
		t.Fatal(err)
	}
	if len(rest) != 2 || rest[0] != "--coverage" {
		t.Fatalf("residual args = %v", rest)
	}
	if limits.FileBytes != 1<<20 || limits.TotalBytes != 512<<20 || limits.InputBytes != 0 {
		t.Fatalf("limits = %+v", limits)
	}
	if _, limits, err := parseReadLimits([]string{"--capture-total-bytes=1073741824"}); err != nil || limits.InputBytes != 1<<30 {
		t.Fatalf("input bytes = %+v err=%v", limits, err)
	}
	for _, bad := range []string{"--capture-file-bytes=0", "--capture-total-bytes=-3", "--capture-file-bytes=x"} {
		if _, _, err := parseReadLimits([]string{bad}); err == nil {
			t.Fatalf("%s was accepted", bad)
		}
	}
}

// flagValue returns the value of the flag named in msg as name=N.
func flagValue(t *testing.T, msg, name string) int64 {
	t.Helper()
	i := strings.LastIndex(msg, name+"=")
	if i < 0 {
		t.Fatalf("%q does not name %s", msg, name)
	}
	var n int64
	if _, err := fmt.Sscanf(msg[i+len(name)+1:], "%d", &n); err != nil {
		t.Fatalf("%q: %s: %v", msg, name, err)
	}
	return n
}

// TestPartialCaptureFails pins that a capture which omitted an attachment, or
// holds a journal over --max-records, is refused with the flags that would
// capture it whole, and that rerunning with them does.
func TestPartialCaptureFails(t *testing.T) {
	dir, _, _ := budgetScratch(t, 4096)
	write := func(budget captureArtifactBudget) captureResult {
		t.Helper()
		result := captureResult{Protocol: 1, Session: "session", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, TransportReady: true, ScratchDir: dir}
		result.recordLimits(0, budget)
		omitted, err := writeCapture(filepath.Join(t.TempDir(), "capture.pprof"), result, budget)
		if err != nil {
			t.Fatal(err)
		}
		result.OmittedFiles = omitted
		return result
	}

	result := write(captureArtifactBudget{FileBytes: 1024})
	err := partialCapture(result)
	if err == nil {
		t.Fatalf("capture omitting %+v was not refused", result.OmittedFiles)
	}
	if !strings.Contains(err.Error(), "--allow-partial") {
		t.Errorf("refusal does not offer --allow-partial: %v", err)
	}
	fileBytes := flagValue(t, err.Error(), "--capture-file-bytes")
	result = write(captureArtifactBudget{FileBytes: fileBytes})
	if len(result.OmittedFiles) != 0 {
		t.Fatalf("rerun with --capture-file-bytes=%d omitted %+v", fileBytes, result.OmittedFiles)
	}
	if err := partialCapture(result); err != nil {
		t.Fatalf("whole capture refused: %v", err)
	}

	result.Limits.JournalRecords = 3 << 20
	result.Limits.MaxRecords = 1 << 20
	err = partialCapture(result)
	if err == nil {
		t.Fatal("capture over --max-records was not refused")
	}
	if n := flagValue(t, err.Error(), "--max-records"); n < 3<<20 {
		t.Errorf("refusal names --max-records=%d, below the %d records observed", n, 3<<20)
	}

	o, err := parseCaptureArgs([]string{"--allow-partial", "--", "true"})
	if err != nil || !o.AllowPartial {
		t.Fatalf("--allow-partial: AllowPartial = %v, err = %v", o.AllowPartial, err)
	}
}

// TestStoredLimitsRoundTrip pins that a capture records the bounds it was
// written under, in capture-status.json for the journal decoders and in the
// manifest for the artifact reader, and that a reader takes its defaults from
// them.
func TestStoredLimitsRoundTrip(t *testing.T) {
	dir, size, _ := budgetScratch(t, 8192)
	result := manyActivities(16, dir)
	result.Limits = &captureLimits{MaxRecords: 1 << 22, JournalRecords: 7}
	out := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(out, result, captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	c, err := readCapturePath(out)
	if err != nil {
		t.Fatal(err)
	}
	got := storedLimits(c)
	want := captureLimits{
		MaxRecords:        1 << 22,
		CaptureFileBytes:  captureFileBytes,
		CaptureTotalBytes: captureTotalBytes,
		JournalRecords:    7,
		JournalBytes:      size,
		AttachmentBytes:   got.AttachmentBytes,
	}
	if got != want {
		t.Errorf("stored limits = %+v, want %+v", got, want)
	}
	if got.AttachmentBytes < size {
		t.Errorf("attachment bytes = %d, less than the %d-byte journal", got.AttachmentBytes, size)
	}
	if want := (cudacapture.Limits{FileBytes: captureFileBytes, TotalBytes: captureTotalBytes}); c.Limits != want {
		t.Errorf("manifest limits = %+v, want %+v", c.Limits, want)
	}

	if records, bytes := decodeLimits(c, 0, 0); records != 1<<22 || bytes != cudacapture.MaxCaptureInputBytes {
		t.Errorf("decodeLimits(0, 0) = %d, %d, want %d, %d", records, bytes, 1<<22, cudacapture.MaxCaptureInputBytes)
	}
	if records, bytes := decodeLimits(c, 5, 6); records != 5 || bytes != 6 {
		t.Errorf("decodeLimits(5, 6) = %d, %d, want the explicit bounds", records, bytes)
	}
}

// TestCompareUsesStoredLimits pins that compare decodes a capture under the
// bounds it records when no flag says otherwise, and that a flag set below
// them is used as set and refused with the value that would decode it.
func TestCompareUsesStoredLimits(t *testing.T) {
	status := phaseStatus(1000, 2000)
	status["limits"] = map[string]any{"max_records": 1 << 22, "journal_bytes": 1 << 30}
	cap := compareCapture([]rec{
		{subtype: "kernel", name: "a", start: 1100, end: 1200},
		{subtype: "kernel", name: "b", start: 1300, end: 1400},
	}, status)

	opts, _, err := parseCompareArgs([]string{"a", "b"}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if records, bytes := decodeLimits(cap, opts.maxRecords, opts.journalBytes); records != 1<<22 || bytes != 1<<30 {
		t.Errorf("compare decodes under %d records, %d bytes; want the stored %d, %d", records, bytes, 1<<22, 1<<30)
	}
	if _, err := measureCompareSide(cap, nil, nil, opts); err != nil {
		t.Fatalf("compare under stored limits: %v", err)
	}

	opts.maxRecords, opts.journalBytes = 1, 16
	_, err = measureCompareSide(cap, nil, nil, opts)
	if err == nil {
		t.Fatal("compare under -max-records=1 -journal-bytes=16 accepted the capture")
	}
	opts.journalBytes = flagValue(t, err.Error(), "--journal-bytes")
	_, err = measureCompareSide(cap, nil, nil, opts)
	if err == nil {
		t.Fatal("compare under -max-records=1 accepted the capture")
	}
	opts.maxRecords = flagValue(t, err.Error(), "--max-records")
	if _, err := measureCompareSide(cap, nil, nil, opts); err != nil {
		t.Errorf("compare under the named bounds %d records, %d bytes: %v", opts.maxRecords, opts.journalBytes, err)
	}
}

// TestReadLimitErrorNamesTheFlags pins that a capture refused for its size
// names the read flags, with values that admit it.
func TestReadLimitErrorNamesTheFlags(t *testing.T) {
	dir, _, _ := budgetScratch(t, 8192)
	out := filepath.Join(t.TempDir(), "capture.pprof")
	if err := writeCaptureStatus(out, manyActivities(16, dir), captureArtifactBudget{}); err != nil {
		t.Fatal(err)
	}
	_, err := readCapturePathLimits(out, cudacapture.Limits{FileBytes: 128, TotalBytes: 256})
	if err == nil {
		t.Fatal("tightened read accepted the capture")
	}
	limits := cudacapture.Limits{
		FileBytes:  flagValue(t, err.Error(), "--capture-file-bytes"),
		TotalBytes: flagValue(t, err.Error(), "--capture-total-bytes"),
	}
	if _, err := readCapturePathLimits(out, limits); err != nil {
		t.Errorf("read under the named limits %+v: %v", limits, err)
	}
}
