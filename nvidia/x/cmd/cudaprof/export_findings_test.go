package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

func findingFixture() map[string][]byte {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 1, 10, 20, 1)
	b.traceActivity("memcpy", "copy", 0, 1, 30, 40, 2)
	return map[string][]byte{"capture.journal": b.finalize()}
}
func findingIdentity() traceExportIdentity {
	return traceExportIdentity{sha256: strings.Repeat("a", 64), path: "capture.pprof"}
}
func findingCompleteStatus() captureResult {
	return captureResult{Protocol: captureProtocolVersion, RunCompleteness: runComplete, TransportReady: true, CollectorFinalized: true, ActivityRecords: 2, RecordLoss: &captureLoss{Offered: 2, Retained: 2}}
}

func TestExportFindingsAdmission(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*captureResult)
		whole  bool
	}{
		{"complete", func(*captureResult) {}, true},
		{"missing protocol", func(s *captureResult) { s.Protocol = 0 }, false},
		{"unsupported protocol", func(s *captureResult) { s.Protocol++ }, false},
		{"unreported transport", func(s *captureResult) { s.TransportReady = false }, false},
		{"unreported finalization", func(s *captureResult) { s.CollectorFinalized = false }, false},
		{"incomplete", func(s *captureResult) { s.RunCompleteness = "partial" }, false},
		{"drop", func(s *captureResult) { s.DroppedRecords = 1 }, false},
		{"missing loss", func(s *captureResult) { s.RecordLoss = nil }, false},
		{"serialized zero on nonempty", func(s *captureResult) { s.RecordLoss = &captureLoss{} }, false},
		{"impossible loss", func(s *captureResult) { s.RecordLoss.Offered = 1 }, false},
		{"retained below raw count", func(s *captureResult) { s.RecordLoss.Offered = 1; s.RecordLoss.Retained = 1 }, false},
		{"count mismatch", func(s *captureResult) { s.ActivityRecords = 1 }, false},
		{"native loss", func(s *captureResult) { s.RecordLoss.Native = 1 }, false},
		{"queue loss", func(s *captureResult) { s.RecordLoss.Queue = 1 }, false},
		{"retention loss", func(s *captureResult) { s.RecordLoss.Offered++ }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status := findingCompleteStatus()
			tc.mutate(&status)
			got, err := exportRecordFindings(findingFixture(), findingIdentity(), status, 100, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 2 || !got[0].Admission.Admitted || got[1].Admission.Admitted != tc.whole {
				t.Fatalf("findings=%+v", got)
			}
			if got[0].Class != tracecontract.FindingObserved || *got[0].Value != 1 {
				t.Fatalf("retained=%+v", got[0])
			}
			if !tc.whole && (got[1].Class != tracecontract.FindingUnavailable || got[1].Admission.ReasonCode == "") {
				t.Fatalf("missing refusal: %+v", got[1])
			}
			if len(got[0].Intervals) != 0 {
				t.Fatal("invented clock interval")
			}
			for _, f := range got {
				if _, err := f.JSON(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestExportFindingsEmptyCaptureRefused(t *testing.T) {
	b := newJournalBuilder()
	status := findingCompleteStatus()
	status.ActivityRecords = 0
	status.RecordLoss = &captureLoss{}
	if _, err := exportRecordFindings(map[string][]byte{"empty.journal": b.finalize()}, findingIdentity(), status, 100, 1<<20); err == nil {
		t.Fatal("empty capture bypassed existing export decoder refusal")
	}
}

func TestExportFindingsDecoderControls(t *testing.T) {
	for _, tc := range []struct {
		name  string
		build func() []byte
		err   bool
	}{
		{"unknown subtype", func() []byte {
			b := newJournalBuilder()
			b.traceActivity("future", "unknown", 0, 1, 10, 20, 1)
			return b.finalize()
		}, false},
		{"invalid kernel", func() []byte {
			b := newJournalBuilder()
			b.traceActivity("kernel", "invalid", 0, 1, 0, 20, 1)
			b.traceActivity("memcpy", "valid", 0, 1, 30, 40, 2)
			return b.finalize()
		}, false},
		{"unfinished", func() []byte {
			b := newJournalBuilder()
			b.traceActivity("kernel", "k", 0, 1, 10, 20, 1)
			return b.unfinalized()
		}, true},
		{"unsupported wire version", func() []byte {
			b := newJournalBuilder()
			b.traceActivity("kernel", "k", 0, 1, 10, 20, 1)
			p := b.finalize()
			binary.BigEndian.PutUint16(p[4:6], 65535)
			return p
		}, true},
		{"truncated", func() []byte { b := newJournalBuilder(); p := b.finalize(); return p[:len(p)-1] }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := exportRecordFindings(map[string][]byte{"capture.journal": tc.build()}, findingIdentity(), findingCompleteStatus(), 100, 1<<20)
			if (err != nil) != tc.err {
				t.Fatalf("error=%v", err)
			}
			if !tc.err && (got[0].Admission.Admitted || got[1].Admission.Admitted) {
				t.Fatalf("projection mismatch admitted: %+v", got)
			}
		})
	}
	identity := findingIdentity()
	identity.sha256 = "invalid"
	if _, err := exportRecordFindings(findingFixture(), identity, findingCompleteStatus(), 100, 1<<20); err == nil {
		t.Fatal("invalid artifact accepted")
	}
	if _, err := exportRecordFindings(findingFixture(), findingIdentity(), findingCompleteStatus(), 1, 1<<20); err == nil {
		t.Fatal("record bound ignored")
	}
	if _, err := exportRecordFindings(findingFixture(), findingIdentity(), findingCompleteStatus(), 100, 1); err == nil {
		t.Fatal("byte bound ignored")
	}
}

func TestExportFindingsOrderIndependent(t *testing.T) {
	makeJournal := func(reverse bool) []byte {
		b := newJournalBuilder()
		if reverse {
			b.traceActivity("memcpy", "copy", 0, 1, 30, 40, 2)
			b.traceActivity("kernel", "k", 0, 1, 10, 20, 1)
		} else {
			b.traceActivity("kernel", "k", 0, 1, 10, 20, 1)
			b.traceActivity("memcpy", "copy", 0, 1, 30, 40, 2)
		}
		return b.finalize()
	}
	var previous []tracecontract.Finding
	for _, reverse := range []bool{false, true} {
		got, err := exportRecordFindings(map[string][]byte{"capture.journal": makeJournal(reverse)}, findingIdentity(), findingCompleteStatus(), 100, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		if previous != nil && !reflect.DeepEqual(previous, got) {
			t.Fatal("record order changed findings")
		}
		previous = got
	}
}

type findingShortWriter struct{}

func (findingShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

type findingFailWriter struct{ err error }

func (w findingFailWriter) Write([]byte) (int, error) { return 0, w.err }

type findingLateFailWriter struct {
	writer io.Writer
	calls  int
	err    error
}

func (w *findingLateFailWriter) Write(p []byte) (int, error) {
	w.calls++
	if w.calls > 2 {
		return 0, w.err
	}
	return w.writer.Write(p)
}

func TestExportFindingsOutput(t *testing.T) {
	got, err := exportRecordFindings(findingFixture(), findingIdentity(), captureResult{}, 100, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := writeRecordFindings(&out, got, findingIdentity(), 8<<48); err != nil {
		t.Fatal(err)
	}
	for _, word := range []string{"retained_kernel_records", "whole_capture_kernel_records", "withheld", "kernel_records"} {
		if !bytes.Contains(out.Bytes(), []byte(word)) {
			t.Fatalf("missing encoded %s", word)
		}
	}
	if err := writeRecordFindings(findingShortWriter{}, got, findingIdentity(), 8<<48); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("short write=%v", err)
	}
	failure := errors.New("sink failure")
	if err := writeRecordFindings(findingFailWriter{failure}, got, findingIdentity(), 8<<48); !errors.Is(err, failure) {
		t.Fatalf("write error=%v", err)
	}
	late := &findingLateFailWriter{writer: io.Discard, err: failure}
	if err := writeRecordFindings(late, got, findingIdentity(), 8<<48); !errors.Is(err, failure) || late.calls != 3 {
		t.Fatalf("finding payload error=%v calls=%d", err, late.calls)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "output.pftrace")
	os.WriteFile(path, []byte("previous"), 0600)
	if err := atomicExport(path, nil, 1, func(w io.Writer) error { return writeRecordFindings(w, got, findingIdentity(), 8<<48) }); err == nil {
		t.Fatal("budget accepted")
	}
	if err := atomicExport(path, nil, 1<<20, func(w io.Writer) error {
		return writeRecordFindings(&findingLateFailWriter{writer: w, err: failure}, got, findingIdentity(), 8<<48)
	}); !errors.Is(err, failure) {
		t.Fatalf("atomic late failure=%v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil || string(raw) != "previous" {
		t.Fatalf("atomic output changed: %q %v", raw, err)
	}
}

func TestExportFindingsCLI(t *testing.T) {
	dir := t.TempDir()
	input := filepath.Join(dir, "input.pprof")
	output := filepath.Join(dir, "output.pftrace")
	writeTestCaptureFile(t, input, compareCapture(iterationRecs(3), completeStatus()))
	export := func(args ...string) []byte {
		t.Helper()
		captureStdout(t, func() error { return runExportCommand(append(args, "-o", output, input)) })
		p, err := os.ReadFile(output)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	raw := export()
	again := export("-detail=raw")
	if !bytes.Equal(raw, again) {
		t.Fatal("default raw export changed")
	}
	findings := export("-findings")
	if !bytes.HasPrefix(findings, raw) {
		t.Fatal("optional findings changed existing raw trace bytes")
	}
	if !bytes.Contains(findings, []byte("whole_capture_kernel_records")) {
		t.Fatal("no finding track")
	}
	os.WriteFile(output, []byte("previous"), 0600)
	if err := runExportCommand([]string{"-findings", "-max-tracks=1", "-o", output, input}); err == nil {
		t.Fatal("track bound ignored")
	}
	if p, _ := os.ReadFile(output); string(p) != "previous" {
		t.Fatal("failed track bound changed output")
	}
	var envelope struct {
		Status captureResult `json:"status"`
	}
	if err := json.Unmarshal([]byte(`{"status":{"record_loss":{}}}`), &envelope); err != nil || envelope.Status.RecordLoss == nil {
		t.Fatal("explicit zero lost presence")
	}
}
