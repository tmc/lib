package ggml_test

import (
	"os"
	"testing"
)

// RequireCudaGate checks whether the cuda manual gate is enabled.
func RequireCudaGate(t *testing.T) {
	t.Helper()
	if os.Getenv("GGML_TEST_HARDWARE") != "1" {
		t.Skip("set GGML_TEST_HARDWARE=1 and build with -tags=ggml_cuda to run ggml CUDA integration tests")
	}
}

// RunCudaGate is set by handwritten hardware integration tests.
var RunCudaGate func(*testing.T)

func TestIntegrationCudaGate(t *testing.T) {
	RequireCudaGate(t)
	if RunCudaGate == nil {
		t.Fatal("manual integration body is not registered")
	}
	RunCudaGate(t)
}

// RequireMetalGate checks whether the metal manual gate is enabled.
func RequireMetalGate(t *testing.T) {
	t.Helper()
	if os.Getenv("GGML_TEST_HARDWARE") != "1" {
		t.Skip("set GGML_TEST_HARDWARE=1 and build with -tags=ggml_metal to run ggml Metal integration tests")
	}
}

// RunMetalGate is set by handwritten hardware integration tests.
var RunMetalGate func(*testing.T)

func TestIntegrationMetalGate(t *testing.T) {
	RequireMetalGate(t)
	if RunMetalGate == nil {
		t.Fatal("manual integration body is not registered")
	}
	RunMetalGate(t)
}
