//go:build linux

package main

import (
	"os"
	"testing"
)

func TestCUDAFoundation(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if len(r.model) != pinnedModelBytes {
		t.Fatalf("mapped bytes = %d, want %d", len(r.model), pinnedModelBytes)
	}
	if r.name == "" || r.major == 0 {
		t.Fatalf("invalid CUDA identity %q sm_%d%d", r.name, r.major, r.minor)
	}
}
