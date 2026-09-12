package cudacapture_test

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// oversizeFixture returns a capture holding want files, which is more than the
// default MaxCaptureFiles bound admits.
func oversizeFixture(want int) *cudacapture.Capture {
	c := validFixture()
	for i := len(c.Files); i < want; i++ {
		c.Files[fmt.Sprintf("pad-%04d.txt", i)] = []byte("pad\n")
	}
	return c
}

// TestLimitsDefaults pins that the zero value and every unset field mean the
// exported package constants, so a caller can raise one bound alone.
func TestLimitsDefaults(t *testing.T) {
	c := oversizeFixture(cudacapture.MaxCaptureFiles + 1)
	var buf bytes.Buffer
	// A zero Limits must behave exactly like the package default.
	wantErr := cudacapture.Write(&buf, c)
	if wantErr == nil {
		t.Fatal("default Write admitted more than MaxCaptureFiles")
	}
	buf.Reset()
	if err := cudacapture.WriteLimits(&buf, c, cudacapture.Limits{}); err == nil || err.Error() != wantErr.Error() {
		t.Fatalf("zero Limits = %v, want %v", err, wantErr)
	}
	// Raising an unrelated field must not relax the file-count bound.
	buf.Reset()
	if err := cudacapture.WriteLimits(&buf, c, cudacapture.Limits{TotalBytes: 1 << 40}); err == nil || err.Error() != wantErr.Error() {
		t.Fatalf("unrelated raise = %v, want %v", err, wantErr)
	}
}

// TestWriteReadLimitsRaiseIsExplicit pins that a raised producer bound is not a
// raised consumer bound: the same bytes still need an explicit raise to be read
// back, so no default reader inherits a producer's budget.
func TestWriteReadLimitsRaiseIsExplicit(t *testing.T) {
	c := oversizeFixture(cudacapture.MaxCaptureFiles + 1)
	raised := cudacapture.Limits{Files: cudacapture.MaxCaptureFiles + 64}

	var buf bytes.Buffer
	if err := cudacapture.WriteLimits(&buf, c, raised); err != nil {
		t.Fatalf("raised Write: %v", err)
	}
	encoded := buf.Bytes()

	if _, err := cudacapture.Read(bytes.NewReader(encoded)); err == nil {
		t.Fatal("default Read accepted a capture written under a raised bound")
	} else if !strings.Contains(err.Error(), "file count exceeds limit") {
		t.Fatalf("default Read error = %v", err)
	}

	got, err := cudacapture.ReadLimits(bytes.NewReader(encoded), raised)
	if err != nil {
		t.Fatalf("raised Read: %v", err)
	}
	if len(got.Files) != len(c.Files) {
		t.Fatalf("round trip files = %d, want %d", len(got.Files), len(c.Files))
	}
}

// TestWriteReadDirLimitsRaiseIsExplicit is the directory form of the same
// contract.
func TestWriteReadDirLimitsRaiseIsExplicit(t *testing.T) {
	c := oversizeFixture(cudacapture.MaxCaptureFiles + 1)
	raised := cudacapture.Limits{Files: cudacapture.MaxCaptureFiles + 64}
	dir := t.TempDir() + "/capture"

	if err := cudacapture.WriteDir(dir, c); err == nil {
		t.Fatal("default WriteDir admitted more than MaxCaptureFiles")
	}
	if err := cudacapture.WriteDirLimits(dir, c, raised); err != nil {
		t.Fatalf("raised WriteDir: %v", err)
	}
	if _, err := cudacapture.ReadDir(dir); err == nil {
		t.Fatal("default ReadDir accepted a directory written under a raised bound")
	}
	got, err := cudacapture.ReadDirLimits(dir, raised)
	if err != nil {
		t.Fatalf("raised ReadDir: %v", err)
	}
	if len(got.Files) != len(c.Files) {
		t.Fatalf("round trip files = %d, want %d", len(got.Files), len(c.Files))
	}
}

// TestLimitsLowerBelowDefault pins that Limits is a real bound in both
// directions, not only an escape hatch: a caller may also tighten it.
func TestLimitsLowerBelowDefault(t *testing.T) {
	c := validFixture()
	tight := cudacapture.Limits{FileBytes: 4}

	var buf bytes.Buffer
	if err := cudacapture.WriteLimits(&buf, c, tight); err == nil {
		t.Fatal("tightened Write accepted an oversized file")
	}
	if err := cudacapture.Write(&buf, c); err != nil {
		t.Fatalf("default Write: %v", err)
	}
	encoded := buf.Bytes()
	if _, err := cudacapture.ReadLimits(bytes.NewReader(encoded), tight); err == nil {
		t.Fatal("tightened Read accepted an oversized embedded file")
	}
	if _, err := cudacapture.Read(bytes.NewReader(encoded)); err != nil {
		t.Fatalf("default Read: %v", err)
	}

	dir := t.TempDir() + "/capture"
	if err := cudacapture.WriteDirLimits(dir, c, tight); err == nil {
		t.Fatal("tightened WriteDir accepted an oversized file")
	}
	if err := cudacapture.WriteDir(dir, c); err != nil {
		t.Fatalf("default WriteDir: %v", err)
	}
	if _, err := cudacapture.ReadDirLimits(dir, tight); err == nil {
		t.Fatal("tightened ReadDir accepted an oversized file")
	}
	if _, err := cudacapture.ReadDir(dir); err != nil {
		t.Fatalf("default ReadDir: %v", err)
	}
}

// TestLimitsInputBytes pins that InputBytes bounds the serialized profile
// itself, independently of the per-file bound.
func TestLimitsInputBytes(t *testing.T) {
	c := validFixture()
	var buf bytes.Buffer
	if err := cudacapture.Write(&buf, c); err != nil {
		t.Fatalf("Write: %v", err)
	}
	encoded := buf.Bytes()
	tight := cudacapture.Limits{InputBytes: int64(len(encoded)) - 1}
	if _, err := cudacapture.ReadLimits(bytes.NewReader(encoded), tight); err == nil {
		t.Fatal("tightened Read accepted an oversized profile")
	}
	exact := cudacapture.Limits{InputBytes: int64(len(encoded))}
	if _, err := cudacapture.ReadLimits(bytes.NewReader(encoded), exact); err != nil {
		t.Fatalf("exact InputBytes: %v", err)
	}
}
