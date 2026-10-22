//go:build darwin && arm64

package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestMetalRuntimeCloseFileError(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.CreateTemp(filepath.Join(home, "tmp"), "metal-close-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Remove(f.Name()) })
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	r := &metalRuntime{file: f}
	if err := r.Close(); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("close error = %v, want closed file error", err)
	}
	if r.file != nil {
		t.Fatal("close retained file ownership")
	}
	if err := r.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}
