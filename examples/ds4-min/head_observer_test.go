package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOutputHeadHistory(t *testing.T) {
	var h outputHeadHistory
	done, err := h.begin(0, []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	before, err := h.hash(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(h.committed) != 0 {
		t.Fatal("pending history published early")
	}
	done(true)
	done, err = h.begin(2, []int{5})
	if err != nil {
		t.Fatal(err)
	}
	after, err := h.hash(2)
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("new token absent from history hash")
	}
	done(false)
	if _, err := h.begin(2, []int{5}); err == nil {
		t.Fatal("failed timeline admitted")
	}
	done, err = h.begin(0, []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	done(true)
	h.invalidate()
	if _, err := h.begin(2, []int{5}); err == nil {
		t.Fatal("restored cache without token history admitted")
	}
	if _, err := h.begin(0, []int{-1}); err == nil {
		t.Fatal("invalid token admitted")
	}
}

func TestHeadSnapshotRefusesOverwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "head.bin")
	if err := writeHeadSnapshot(path, []byte{1}); err != nil {
		t.Fatal(err)
	}
	if err := writeHeadSnapshot(path, []byte{2}); err == nil {
		t.Fatal("overwrite admitted")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 1 || data[0] != 1 {
		t.Fatal("sealed artifact changed")
	}
}
