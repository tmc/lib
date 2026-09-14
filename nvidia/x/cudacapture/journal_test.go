package cudacapture

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"math/rand"
	"strings"
	"testing"
)

func TestIsJournalName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"capture.journal", true},
		{"capture.journal.gz", true},
		{"worker-process-42-3-capture.journal", true},
		{"worker-process-42-3-capture.journal.gz", true},
		{"handshake.json", false},
		{"capture-status.json", false},
		{"capture.journal.bak", false},
	}
	for _, tt := range tests {
		if got := IsJournalName(tt.name); got != tt.want {
			t.Errorf("IsJournalName(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}

// TestJournalReadsBothForms pins that a reader opens a capture written before
// journals were compressed and one written after, without knowing which it has.
func TestJournalReadsBothForms(t *testing.T) {
	raw := []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)
	var gz bytes.Buffer
	zw := gzip.NewWriter(&gz)
	if _, err := zw.Write(raw); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	c := &Capture{Files: map[string][]byte{
		"capture.journal":                        raw,
		"worker-process-42-3-capture.journal.gz": gz.Bytes(),
		"handshake.json":                         []byte(`{"protocol":1}`),
	}}
	names := c.JournalNames()
	if len(names) != 2 || names[0] != "capture.journal" || names[1] != "worker-process-42-3-capture.journal.gz" {
		t.Fatalf("JournalNames() = %v", names)
	}
	for _, name := range names {
		got, err := c.Journal(name)
		if err != nil {
			t.Fatalf("Journal(%q): %v", name, err)
		}
		if !bytes.Equal(got, raw) {
			t.Fatalf("Journal(%q) = %q, want %q", name, got, raw)
		}
	}
	if _, err := c.Journal("missing.journal"); err == nil {
		t.Fatal("Journal of an absent member returned no error")
	}
	// A member named as compressed whose bytes are not gzip is corruption, and
	// the error must say which fault it is: gzip's own "invalid header" does
	// not distinguish a damaged member from a misnamed plain one.
	c.Files["corrupt-capture.journal.gz"] = []byte("not gzip")
	_, err := c.Journal("corrupt-capture.journal.gz")
	if err == nil {
		t.Fatal("Journal of a corrupt member returned no error")
	}
	if !strings.Contains(err.Error(), "named as compressed") {
		t.Fatalf("Journal of a corrupt member = %v, want an error naming the name/content mismatch", err)
	}
}

// TestJournalDecompressesGzipUnderPlainName pins that compression is detected
// by content as well as by name. A writer outside this repo may embed gzip
// bytes under the plain journal name; trusting the suffix alone hands those
// bytes to the frame decoder, which reports a corrupt journal for a journal
// that is merely compressed.
func TestJournalDecompressesGzipUnderPlainName(t *testing.T) {
	raw := []byte("journal frames")
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	if _, err := zw.Write(raw); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	c := &Capture{Files: map[string][]byte{"capture.journal": buf.Bytes()}}
	got, err := c.Journal("capture.journal")
	if err != nil {
		t.Fatalf("Journal: %v", err)
	}
	if !bytes.Equal(got, raw) {
		t.Fatalf("Journal = %q, want %q", got, raw)
	}
}

// countingReader records how much of the stream was consumed.
type countingReader struct {
	io.Reader
	n int64
}

func (r *countingReader) Read(p []byte) (int, error) {
	n, err := r.Reader.Read(p)
	r.n += int64(n)
	return n, err
}

// TestCompressJournalStopsAtLimit pins that compression is bounded: a journal
// whose compressed form does not fit is reported, not buffered.
func TestCompressJournalStopsAtLimit(t *testing.T) {
	b := make([]byte, 8<<20)
	rand.New(rand.NewSource(7)).Read(b)
	counted := &countingReader{Reader: bytes.NewReader(b)}
	if _, err := CompressJournal(counted, 4096); !errors.Is(err, ErrJournalTooLarge) {
		t.Fatalf("CompressJournal under a small limit: err = %v, want ErrJournalTooLarge", err)
	}
	// The limit bounds memory, so it must stop the stream rather than compress
	// the whole journal and reject the result.
	if counted.n == int64(len(b)) {
		t.Fatalf("CompressJournal read all %d bytes before giving up", counted.n)
	}
	out, err := CompressJournal(bytes.NewReader(b), 0)
	if err != nil {
		t.Fatalf("CompressJournal at the default limit: %v", err)
	}
	if len(out) == 0 {
		t.Fatal("CompressJournal produced no output")
	}
}
