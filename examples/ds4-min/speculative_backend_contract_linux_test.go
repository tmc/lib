//go:build linux

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestCUDAStateAdapterSpeculationFailsClosed(t *testing.T) {
	a := &cudaStateAdapter{}
	verifier, err := a.NewSpeculativeVerifier(nil)
	if verifier != nil {
		t.Fatal("NewSpeculativeVerifier returned a verifier for an incomplete adapter")
	}
	if !errors.Is(err, dsv4.ErrUnimplemented) {
		t.Fatalf("NewSpeculativeVerifier error = %v, want dsv4.ErrUnimplemented", err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.NewSpeculativeVerifier(nil); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("NewSpeculativeVerifier after Close = %v, want dsv4.ErrClosed", err)
	}
}
