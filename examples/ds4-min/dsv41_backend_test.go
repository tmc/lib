package main

import (
	"errors"
	"os"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

var _ dsv4.Backend = (*v41CPUBackend)(nil)

func TestV41CPUOwnerAdmission(t *testing.T) {
	if _, err := newV41CPUBackend("missing", v41CPUOptions{}); err == nil {
		t.Fatal("missing file accepted")
	}
	var owner v41CPUBackend
	if owner.Close() != nil || owner.Close() != nil {
		t.Fatal("empty close failed")
	}
}

type failingV41File struct {
	statErr, closeErr error
	closes            int
}

func (f *failingV41File) ReadAt([]byte, int64) (int, error) { return 0, errors.New("unexpected read") }
func (f *failingV41File) Stat() (os.FileInfo, error)        { return nil, f.statErr }
func (f *failingV41File) Close() error                      { f.closes++; return f.closeErr }

func TestV41ConstructorRetainsFailedCleanup(t *testing.T) {
	statErr, closeErr := errors.New("stat failed"), errors.New("close failed")
	f := &failingV41File{statErr: statErr, closeErr: closeErr}
	b, err := openV41CPUBackend("fixture", v41CPUOptions{}, func(string) (v41ModelFile, error) { return f, nil })
	if b == nil || !errors.Is(err, statErr) || !errors.Is(err, closeErr) || f.closes != 1 {
		t.Fatalf("constructor=%v err=%v closes=%d", b, err, f.closes)
	}
	if !errors.Is(b.Close(), closeErr) || b.file != f {
		t.Fatal("failed retry lost file")
	}
	f.closeErr = nil
	if b.Close() != nil || b.file != nil || f.closes != 3 {
		t.Fatal("successful retry did not retire file")
	}
	if b.Close() != nil || f.closes != 3 {
		t.Fatal("retired file closed again")
	}
}

func TestV41ConstructorSuccessfulCleanup(t *testing.T) {
	f := &failingV41File{statErr: errors.New("stat failed")}
	b, err := openV41CPUBackend("fixture", v41CPUOptions{}, func(string) (v41ModelFile, error) { return f, nil })
	if b != nil || !errors.Is(err, f.statErr) || f.closes != 1 {
		t.Fatalf("constructor=%v err=%v closes=%d", b, err, f.closes)
	}
}

func TestV41ConstructorMissingOwner(t *testing.T) {
	b, err := openV41CPUBackend("fixture", v41CPUOptions{}, func(string) (v41ModelFile, error) { return nil, nil })
	if b != nil || err == nil {
		t.Fatalf("constructor = %v, %v", b, err)
	}
}
