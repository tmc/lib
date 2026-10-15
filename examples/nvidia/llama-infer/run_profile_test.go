package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cuda"
)

// TestRunProfile is a behaviour test: run with -profile on the CPU
// reference executor returns nil and leaves a pprof profile in the file.
func TestRunProfile(t *testing.T) {
	// Hide models under the user's home so run uses the generated model.
	t.Setenv("HOME", t.TempDir())
	if p := findDefaultModelPath(); p != "" {
		t.Skipf("model found at %s; this test needs the generated model", p)
	}
	if dev, err := cuda.NewDevice(0); err == nil {
		if err := dev.Close(); err != nil {
			t.Errorf("close device: %v", err)
		}
		t.Skip("CUDA device present; this test needs the CPU reference executor")
	}

	path := filepath.Join(t.TempDir(), "llama.pprof")
	oldProfile, oldSteps := *profilePath, *stepsFlag
	*profilePath, *stepsFlag = path, 1
	defer func() { *profilePath, *stepsFlag = oldProfile, oldSteps }()

	if err := run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := profile.ParseData(data); err != nil {
		t.Errorf("parse profile %s: %v", path, err)
	}
}
