package main

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestCollectorOutputPublication(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "profile")
	stop := errors.New("producer failed")
	if err := writeCollectorOutput(path, func(w io.Writer) error {
		io.WriteString(w, "partial")
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatal("output visible before producer completed")
		}
		return stop
	}); !errors.Is(err, stop) {
		t.Fatalf("failure: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("failed output retained: %v %v", entries, err)
	}
	write := func(w io.Writer) error { _, err := io.WriteString(w, "complete"); return err }
	if err := writeCollectorOutput(path, write); err != nil {
		t.Fatal(err)
	}
	if err := writeCollectorOutput(path, func(w io.Writer) error { _, err := io.WriteString(w, "replacement"); return err }); !errors.Is(err, os.ErrExist) {
		t.Fatalf("existing output overwritten: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "complete" {
		t.Fatalf("output: %q %v", data, err)
	}
}

func TestCollectorOutputCleanupError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile")
	err := writeCollectorOutput(path, func(w io.Writer) error {
		// Unlink the owned temporary inode while it is open. Both publication
		// and deferred cleanup must report failure, not a successful output.
		return os.Remove(w.(*os.File).Name())
	})
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("cleanup failure hidden: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("missing temporary file published: %v", err)
	}
}
