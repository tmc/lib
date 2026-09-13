package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// readAccounting decodes the receipt written beside a streamed sidecar.
func readAccounting(t *testing.T, sidecar string) streamAccounting {
	t.Helper()
	b, err := os.ReadFile(accountingPath(sidecar))
	if err != nil {
		t.Fatalf("read accounting: %v", err)
	}
	var acc streamAccounting
	if err := json.Unmarshal(b, &acc); err != nil {
		t.Fatalf("decode accounting: %v", err)
	}
	return acc
}

// readSidecar decodes the sidecar the way the converter's readHost does,
// with DisallowUnknownFields so an added field fails here rather than in a
// capture nobody re-runs.
func readSidecar(t *testing.T, path string) []hostEventRecord {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open sidecar: %v", err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var got []hostEventRecord
	for {
		var rec hostEventRecord
		err := dec.Decode(&rec)
		if errors.Is(err, io.EOF) {
			return got
		}
		if err != nil {
			t.Fatalf("decode record %d: %v", len(got), err)
		}
		got = append(got, rec)
	}
}

// withStreamRecorder installs a recorder for the duration of fn.
func withStreamRecorder(t *testing.T, level spanLevel, path string, fn func()) error {
	t.Helper()
	r, err := newStreamRecorder(level, "test-run", path)
	if err != nil {
		t.Fatalf("new stream recorder: %v", err)
	}
	prev := streamSpans
	streamSpans = r
	defer func() { streamSpans = prev }()
	fn()
	return r.Close()
}

func TestStreamRecorderWritesEverySpan(t *testing.T) {
	// The buffered recorder drops past spanRecordLimit. This one has no
	// cap, so the interesting count is one well past it.
	const n = spanRecordLimit + 1_000
	path := filepath.Join(t.TempDir(), "spans.jsonl")
	if err := withStreamRecorder(t, levelOp, path, func() {
		for i := 0; i < n; i++ {
			beginStream(levelOp, "op").end()
		}
	}); err != nil {
		t.Fatalf("close: %v", err)
	}
	got := readSidecar(t, path)
	if len(got) != n {
		t.Errorf("wrote %d records, want %d", len(got), n)
	}
	acc := readAccounting(t, path)
	if acc.Begun != n || acc.Ended != n || acc.Encoded != n {
		t.Errorf("accounting begun/ended/written = %d/%d/%d, want %d each", acc.Begun, acc.Ended, acc.Encoded, n)
	}
	if acc.Dropped != 0 || acc.Unended != 0 || !acc.Complete {
		t.Errorf("accounting dropped=%d unended=%d complete=%v, want 0/0/true", acc.Dropped, acc.Unended, acc.Complete)
	}
	if acc.BytesFile <= 0 {
		t.Errorf("accounting sidecar_bytes = %d, want positive", acc.BytesFile)
	}
	if acc.Level != "op" {
		t.Errorf("accounting level = %q, want %q", acc.Level, "op")
	}
}

func TestStreamRecorderHoldsOnlyOpenSpans(t *testing.T) {
	// What bounds memory is nesting depth, not run length: a million
	// sequential spans must peak at one open span, and a nest of three
	// at three.
	tests := []struct {
		name  string
		run   func()
		peak  int
		spans int
	}{
		{"sequential", func() {
			for i := 0; i < 5_000; i++ {
				beginStream(levelOp, "op").end()
			}
		}, 1, 5_000},
		{"nested", func() {
			a := beginStream(levelOp, "a")
			b := beginStream(levelOp, "b")
			c := beginStream(levelOp, "c")
			c.end()
			b.end()
			a.end()
		}, 3, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "spans.jsonl")
			if err := withStreamRecorder(t, levelOp, path, tt.run); err != nil {
				t.Fatalf("close: %v", err)
			}
			acc := readAccounting(t, path)
			if acc.OpenPeak != tt.peak {
				t.Errorf("open_peak = %d, want %d", acc.OpenPeak, tt.peak)
			}
			if acc.Encoded != tt.spans {
				t.Errorf("written = %d, want %d", acc.Encoded, tt.spans)
			}
		})
	}
}

func TestStreamRecorderCountsUnendedSpans(t *testing.T) {
	// A run that stops inside a span must say so in the receipt rather
	// than write a record with an invented duration.
	path := filepath.Join(t.TempDir(), "spans.jsonl")
	if err := withStreamRecorder(t, levelOp, path, func() {
		beginStream(levelOp, "closed").end()
		beginStream(levelOp, "still open")
	}); err != nil {
		t.Fatalf("close: %v", err)
	}
	got := readSidecar(t, path)
	if len(got) != 1 || got[0].Name != "closed" {
		t.Fatalf("records = %+v, want one named \"closed\"", got)
	}
	acc := readAccounting(t, path)
	if acc.Begun != 2 || acc.Ended != 1 || acc.Encoded != 1 || acc.Unended != 1 {
		t.Errorf("begun/ended/written/unended = %d/%d/%d/%d, want 2/1/1/1", acc.Begun, acc.Ended, acc.Encoded, acc.Unended)
	}
	if acc.Complete {
		t.Error("complete = true with an unended span, want false")
	}
}

func TestStreamRecorderIgnoresDeeperLevels(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spans.jsonl")
	if err := withStreamRecorder(t, levelLayer, path, func() {
		beginStream(levelLayer, "layer").end()
		beginStream(levelOp, "op").end()
	}); err != nil {
		t.Fatalf("close: %v", err)
	}
	got := readSidecar(t, path)
	if len(got) != 1 || got[0].Name != "layer" {
		t.Fatalf("records = %+v, want one named \"layer\"", got)
	}
}

// failingSink accepts n bytes and then fails every write.
type failingSink struct {
	budget int
	w      io.Writer
}

func (s *failingSink) Write(p []byte) (int, error) {
	if s.budget <= 0 {
		return 0, errors.New("sink is full")
	}
	if len(p) > s.budget {
		n, _ := s.w.Write(p[:s.budget])
		s.budget = 0
		return n, errors.New("sink is full")
	}
	s.budget -= len(p)
	return s.w.Write(p)
}

func (s *failingSink) Stat() (os.FileInfo, error) { return nil, errors.New("no stat") }
func (s *failingSink) Close() error               { return nil }

// TestStreamAccountingSeparatesEndedFromEncoded forces the encoder itself
// to fail, mid-run, and then closes more spans. It exists because a version
// that counted a span as encoded before the write was attempted passed every
// other test here: those either never fail a write, or fail only the final
// flush, by which point the buffer has already accepted every record.
//
// So this control uses an unbuffered-sized sink and records larger than the
// budget, which makes Encode return an error while spans are still being
// closed. Ended must then exceed Encoded by exactly the number of spans that
// closed after the sink went dead.
func TestStreamAccountingSeparatesEndedFromEncoded(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "spans.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// Two records fit; the write that carries the third fails.
	const accept = 2
	sink := &countingSink{limit: accept, w: f}
	r := newStreamRecorderSink(levelOp, "test-run", path, sink)
	// A one-byte buffer makes bufio pass each Encode straight through, so
	// the failure lands inside end() rather than at Flush.
	r.buf = bufio.NewWriterSize(sink, 1)
	r.enc = json.NewEncoder(r.buf)

	prev := streamSpans
	streamSpans = r
	const total = 10
	for i := 0; i < total; i++ {
		beginStream(levelOp, "op").end()
	}
	streamSpans = prev
	if err := r.Close(); err == nil {
		t.Error("Close returned nil after the encoder failed")
	}

	acc := readAccounting(t, path)
	if acc.Begun != total || acc.Ended != total {
		t.Errorf("begun/ended = %d/%d, want %d/%d", acc.Begun, acc.Ended, total, total)
	}
	if acc.Encoded != accept {
		t.Errorf("encoded = %d, want exactly %d: the sink accepted that many records", acc.Encoded, accept)
	}
	if acc.Ended <= acc.Encoded {
		t.Errorf("ended %d not greater than encoded %d; the receipt cannot show the loss", acc.Ended, acc.Encoded)
	}
	if acc.Complete {
		t.Error("complete = true after the encoder failed, want false")
	}
}

// countingSink accepts limit whole writes and then fails every one.
type countingSink struct {
	limit int
	n     int
	w     io.Writer
}

func (s *countingSink) Write(p []byte) (int, error) {
	if s.n >= s.limit {
		return 0, errors.New("sink is full")
	}
	s.n++
	return s.w.Write(p)
}

func (s *countingSink) Stat() (os.FileInfo, error) { return nil, errors.New("no stat") }
func (s *countingSink) Close() error               { return nil }

func TestStreamAccountingAdmitsAShortFile(t *testing.T) {
	// The receipt has to be readable on its own. A writer that fails
	// partway must not produce a receipt that looks like a complete run,
	// because whoever reads the sidecar later will not have the exit
	// status that reported the failure.
	dir := t.TempDir()
	path := filepath.Join(dir, "spans.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// Room for a few records, then every further write fails.
	sink := &failingSink{budget: 600, w: f}
	r := newStreamRecorderSink(levelOp, "test-run", path, sink)
	prev := streamSpans
	streamSpans = r
	for i := 0; i < 100; i++ {
		beginStream(levelOp, "op").end()
	}
	streamSpans = prev
	closeErr := r.Close()
	f.Close()
	if closeErr == nil {
		t.Error("Close returned nil after the sink failed")
	}

	acc := readAccounting(t, path)
	if acc.Complete {
		t.Error("complete = true after a failed write, want false")
	}
	if acc.Error == "" {
		t.Error("error field empty after a failed write")
	}
	if !strings.Contains(acc.Error, "stat") {
		t.Errorf("error %q does not mention the stat failure", acc.Error)
	}
	if acc.Begun != 100 || acc.Ended != 100 {
		t.Errorf("begun/ended = %d/%d, want 100/100", acc.Begun, acc.Ended)
	}
	// Encoded may equal Ended here: the sidecar is buffered, so every
	// record can reach the encoder and still not reach the disk. That is
	// exactly why the receipt cannot rest on a record count. What has to
	// hold is that the receipt refuses to call itself complete, names the
	// failure, and reports no size it cannot stat.
	if !strings.Contains(acc.Error, "flush") && !strings.Contains(acc.Error, "sink is full") {
		t.Errorf("error %q names neither the flush nor the write failure", acc.Error)
	}
	if acc.BytesFile != 0 {
		t.Errorf("sidecar_bytes = %d with no usable stat, want 0 alongside the error", acc.BytesFile)
	}
	// Count whole lines rather than decoding: a failed write can leave a
	// partial record, which is itself the short file this control is about.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(raw), "\n"); got >= acc.Ended {
		t.Errorf("sidecar holds %d records and the receipt reports %d ended; the control needs a short file", got, acc.Ended)
	}
}
