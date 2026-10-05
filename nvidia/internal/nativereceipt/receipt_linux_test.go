//go:build linux

package nativereceipt

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"golang.org/x/sys/unix"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These are ordinary mmap fixtures, not CUDA libraries or GPU qualification.
func mapFixture(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(strings.Repeat("synthetic mapping fixture\n", 128)), 0600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	b, err := unix.Mmap(int(f.Fd()), 0, 4096, unix.PROT_READ, unix.MAP_PRIVATE)
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := unix.Munmap(b); err != nil {
			t.Error(err)
		}
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	return path
}
func testReceipt(t *testing.T) (Receipt, Process, string, string) {
	t.Helper()
	mapFixture(t, "libcuda.so.fixture")
	mapFixture(t, "libnvrtc.so.fixture")
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	source := fmt.Sprintf("%x", sha256.Sum256([]byte("source")))
	artifact := fmt.Sprintf("%x", sha256.Sum256([]byte("artifact")))
	if err := s.Capture("actual-synthetic-fixture", source, artifact); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "proof.json")
	if err := s.Write(path); err != nil {
		t.Fatal(err)
	}
	r, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Identity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	exe, err := os.ReadFile(p.Executable)
	if err != nil {
		t.Fatal(err)
	}
	p.ExecutableSHA256 = fmt.Sprintf("%x", sha256.Sum256(exe))
	if err := r.Check(p, source, artifact); err != nil {
		t.Fatal(err)
	}
	if r.BuiltinsObserved || len(r.Unobserved) == 0 {
		t.Fatal("synthetic fixtures must not claim builtins")
	}
	return r, p, source, artifact
}
func TestMappedDescriptorReceipt(t *testing.T) {
	r, p, source, artifact := testReceipt(t)
	cases := []struct {
		name   string
		mutate func(*Receipt, *Process, *string, *string)
	}{
		{"pid", func(r *Receipt, p *Process, s, a *string) { p.PID++ }},
		{"group", func(r *Receipt, p *Process, s, a *string) { p.Group++ }},
		{"start", func(r *Receipt, p *Process, s, a *string) { p.Start++ }},
		{"source", func(r *Receipt, p *Process, s, a *string) { *s = strings.Repeat("0", 64) }},
		{"artifact", func(r *Receipt, p *Process, s, a *string) { *a = strings.Repeat("0", 64) }},
		{"descriptor", func(r *Receipt, p *Process, s, a *string) { r.Files[0].Inode++ }},
		{"raw-map", func(r *Receipt, p *Process, s, a *string) { r.Stages[0].Mappings[0].Offset++ }},
		{"builtin-claim", func(r *Receipt, p *Process, s, a *string) { r.BuiltinsObserved = true }},
		{"driver-claim", func(r *Receipt, p *Process, s, a *string) { r.DriverObserved = false }},
		{"body-hash", func(r *Receipt, p *Process, s, a *string) { r.Files[0].SHA256 = "" }},
		{"absent-limitation", func(r *Receipt, p *Process, s, a *string) { r.Unobserved = nil }},
	}
	data, _ := json.Marshal(r)
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			var copy Receipt
			if err := json.Unmarshal(data, &copy); err != nil {
				t.Fatal(err)
			}
			pp, ss, aa := p, source, artifact
			tt.mutate(&copy, &pp, &ss, &aa)
			if err := copy.Check(pp, ss, aa); err == nil {
				t.Fatal("accepted changed proof")
			}
		})
	}
}
func TestMappedBodyChangeRefused(t *testing.T) {
	path := mapFixture(t, "libcuda.so.changed")
	mapFixture(t, "libnvrtc.so.changed")
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Capture("before", "", ""); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Write(filepath.Join(t.TempDir(), "proof.json")); err == nil {
		t.Fatal("accepted changed mapped body")
	}
}
func TestNonregularReceiptRefused(t *testing.T) {
	dir := t.TempDir()
	fifo := filepath.Join(dir, "fifo")
	if err := unix.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(fifo); err == nil {
		t.Fatal("accepted FIFO")
	}
	regular := filepath.Join(dir, "body")
	if err := os.WriteFile(regular, []byte("{}"), 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(regular, link); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(link); err == nil {
		t.Fatal("accepted symlink")
	}
}
func TestParseMaps(t *testing.T) {
	good := "1000-2000 r-xp 00000000 08:01 42 /path with spaces/libcuda.so.1\n"
	got, err := parseMaps([]byte(good))
	if err != nil || len(got) != 1 || got[0].Path != "/path with spaces/libcuda.so.1" {
		t.Fatalf("%+v %v", got, err)
	}
	for _, raw := range []string{"", "junk\n", strings.ReplaceAll(good, "r-xp", "r?xp"), strings.ReplaceAll(good, "1000-2000", "2000-1000"), strings.Repeat("x", maxMaps+1)} {
		if _, err := parseMaps([]byte(raw)); err == nil {
			t.Fatal("accepted malformed maps")
		}
	}
}

func TestCaptureBoundsAndDescriptorRetirement(t *testing.T) {
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Capture("before", "", ""); err != nil {
		t.Fatal(err)
	}
	var files []*os.File
	for _, b := range s.bodies {
		files = append(files, b.file)
	}
	if len(files) == 0 {
		t.Fatal("executable descriptor not retained")
	}
	if err := s.Capture("bad", "not-a-digest", ""); err == nil {
		t.Fatal("accepted bad digest")
	}
	s.receipt.Stages = make([]Stage, maxStages)
	if err := s.Capture("excess", "", ""); err == nil {
		t.Fatal("accepted excess stages")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if _, err := f.Stat(); err == nil {
			t.Fatal("descriptor retained after Close")
		}
	}
	if err := s.Capture("closed", "", ""); err == nil {
		t.Fatal("capture after close")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestReceiptExclusiveOutputAndStrictInput(t *testing.T) {
	mapFixture(t, "libcuda.so.exclusive")
	mapFixture(t, "libnvrtc.so.exclusive")
	s, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Capture("before", "", ""); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "existing")
	if err := os.WriteFile(path, []byte("retained"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := s.Write(path); err == nil {
		t.Fatal("overwrote existing receipt")
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "retained" {
		t.Fatalf("existing changed: %q %v", data, err)
	}
	for _, body := range []string{`{"unknown":1}`, `{"process":{},"process":{}}`, `{} {}`} {
		input := filepath.Join(t.TempDir(), "input")
		if err := os.WriteFile(input, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := Read(input); err == nil {
			t.Fatalf("accepted %q", body)
		}
	}
}
