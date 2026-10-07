//go:build !linux

package main

import "testing"

func TestRuntimeUnsupportedCPU(t *testing.T) {
	if runtimeProofSupported() == nil {
		t.Fatal("non-Linux proof accepted")
	}
	if _, err := captureRuntime(); err == nil {
		t.Fatal("non-Linux mapping fabricated")
	}
}
