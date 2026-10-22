package dsv4kvstore

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

type storePayload []byte

func (p storePayload) WriteTo(w io.Writer) (int64, error) {
	n, err := w.Write(p)
	return int64(n), err
}

func storeHeader() Header {
	return Header{QuantBits: QuantQ2, Context: 8192, RawWindow: 128, ModelDigest: [32]byte{7}}
}

func TestStoreLookupLongestPrefixAndPayload(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(Options{Dir: dir, BudgetMB: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Save(storeHeader(), []byte("hello"), storePayload("short"), []byte("trailer")); err != nil {
		t.Fatal(err)
	}
	if err := s.Save(storeHeader(), []byte("hello world"), storePayload("long"), nil); err != nil {
		t.Fatal(err)
	}
	entry, err := s.Lookup([]byte("hello world!"), 8192, [32]byte{7}, QuantQ2)
	if err != nil {
		t.Fatal(err)
	}
	if string(entry.Text) != "hello world" || string(entry.Trailer) != "" {
		t.Fatalf("entry = %+v", entry)
	}
	got, err := entry.Payload()
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(got)
	got.Close()
	if err != nil || string(data) != "long" {
		t.Fatalf("payload = %q, err=%v", data, err)
	}
	if _, err := s.Lookup([]byte("no match"), 4096, [32]byte{7}, QuantQ2); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing lookup = %v, want ErrNotFound", err)
	}
}

func TestStoreLookupFiltersContextModelAndQuant(t *testing.T) {
	s, err := Open(Options{Dir: t.TempDir(), BudgetMB: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Save(storeHeader(), []byte("prefix"), storePayload("payload"), nil); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		ctx    uint32
		model  [32]byte
		quant  uint8
		wantOK bool
	}{
		{"context too small", 4096, [32]byte{7}, QuantQ2, false},
		{"context larger is compatible", 16384, [32]byte{7}, QuantQ2, true},
		{"model mismatch", 4096, [32]byte{8}, QuantQ2, false},
		{"quant mismatch", 4096, [32]byte{7}, QuantQ4, false},
		{"wildcards", 0, [32]byte{}, 0, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := s.Lookup([]byte("prefix!"), test.ctx, test.model, test.quant)
			if (err == nil) != test.wantOK {
				t.Fatalf("Lookup error = %v, want success %v", err, test.wantOK)
			}
		})
	}
}

func TestStoreKeySeparatesModelAndQuant(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(Options{Dir: dir, BudgetMB: 1})
	if err != nil {
		t.Fatal(err)
	}
	text := []byte("same rendered prompt")
	for _, test := range []struct {
		name   string
		header Header
		quant  uint8
		want   string
	}{
		{name: "q2", header: Header{QuantBits: QuantQ2, Context: 8192, RawWindow: 128, ModelDigest: [32]byte{1}}, quant: QuantQ2, want: "q2"},
		{name: "q4", header: Header{QuantBits: QuantQ4, Context: 8192, RawWindow: 128, ModelDigest: [32]byte{1}}, quant: QuantQ4, want: "q4"},
		{name: "other model", header: Header{QuantBits: QuantQ2, Context: 8192, RawWindow: 128, ModelDigest: [32]byte{2}}, quant: QuantQ2, want: "other model"},
	} {
		if err := s.Save(test.header, text, storePayload(test.want), nil); err != nil {
			t.Fatalf("Save %s: %v", test.name, err)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var records int
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) == storeSuffix {
			records++
		}
	}
	if records != 3 {
		t.Fatalf("cache records = %d, want 3", records)
	}
	for _, test := range []struct {
		name  string
		model [32]byte
		quant uint8
		want  string
	}{
		{name: "q2", model: [32]byte{1}, quant: QuantQ2, want: "q2"},
		{name: "q4", model: [32]byte{1}, quant: QuantQ4, want: "q4"},
		{name: "other model", model: [32]byte{2}, quant: QuantQ2, want: "other model"},
	} {
		entry, err := s.Lookup(text, 8192, test.model, test.quant)
		if err != nil {
			t.Fatalf("Lookup %s: %v", test.name, err)
		}
		if got := string(mustRead(t, entry)); got != test.want {
			t.Fatalf("Lookup %s payload = %q, want %q", test.name, got, test.want)
		}
	}
}

func TestStoreSaveIsAtomicAndCleansTemporaryFiles(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(Options{Dir: dir, BudgetMB: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Save(storeHeader(), []byte("atomic"), storePayload("payload"), nil); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || filepath.Ext(entries[0].Name()) != storeSuffix {
		t.Fatalf("cache directory = %v", entries)
	}
	if mode := entries[0].Type().Perm(); mode&0077 != 0 {
		t.Fatalf("cache entry permissions = %o, want private", mode)
	}

	if err := s.Save(storeHeader(), []byte("broken"), failingWriterTo{}, nil); err == nil {
		t.Fatal("failing payload unexpectedly saved")
	}
	entries, err = os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("failed save left files: %v", entries)
	}
}

type failingWriterTo struct{}

func (failingWriterTo) WriteTo(io.Writer) (int64, error) { return 0, errors.New("payload failed") }

func TestStoreTouchRejectsForeignEntry(t *testing.T) {
	first, err := Open(Options{Dir: t.TempDir(), BudgetMB: 1})
	if err != nil {
		t.Fatal(err)
	}
	second, err := Open(Options{Dir: t.TempDir(), BudgetMB: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Save(storeHeader(), []byte("x"), storePayload("y"), nil); err != nil {
		t.Fatal(err)
	}
	entry, err := first.Lookup([]byte("x"), 8192, [32]byte{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := second.Touch(entry); err == nil {
		t.Fatal("foreign entry touch unexpectedly succeeded")
	}
	if _, err := entry.Payload(); err != nil || !bytes.Equal(mustRead(t, entry), []byte("y")) {
		t.Fatalf("payload changed after foreign touch: %v", err)
	}
}

func TestStoreEvictsToBudgetAndIgnoresMalformedEntries(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(Options{Dir: dir, BudgetMB: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "broken"+storeSuffix), []byte("not a record"), 0600); err != nil {
		t.Fatal(err)
	}
	payload := storePayload(bytes.Repeat([]byte{'x'}, 700<<10))
	if err := s.Save(storeHeader(), []byte("first"), payload, nil); err != nil {
		t.Fatal(err)
	}
	if err := s.Save(storeHeader(), []byte("second"), payload, nil); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var records int
	for _, entry := range entries {
		if filepath.Ext(entry.Name()) == storeSuffix && entry.Name() != "broken"+storeSuffix {
			records++
		}
	}
	if records != 1 {
		t.Fatalf("cache records = %d, want one after eviction", records)
	}
	if _, err := s.Lookup([]byte("second suffix"), 8192, [32]byte{7}, QuantQ2); err != nil {
		t.Fatalf("newest entry was evicted: %v", err)
	}
}

func mustRead(t *testing.T, e *Entry) []byte {
	t.Helper()
	r, err := e.Payload()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
