package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestCapture(t *testing.T) {
	path := os.Getenv("CUDA_PROFILE_CAPTURE")
	if path == "" {
		t.Skip("set CUDA_PROFILE_CAPTURE to a real capture (.pprof or dir)")
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat capture path: %v", err)
	}

	var cap *cudacapture.Capture
	if fi.IsDir() {
		cap, err = cudacapture.ReadDir(path)
	} else {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		cap, err = cudacapture.Read(f)
		if err != nil {
			// fallback to directory containing it
			cap, err = cudacapture.ReadDir(filepath.Dir(path))
		}
	}
	if err != nil {
		t.Fatalf("load capture: %v", err)
	}

	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("load code: %v", err)
	}
	if len(code.Kernels) == 0 {
		t.Fatal("expected kernels in code")
	}

	if cap.Profile == nil {
		t.Fatal("expected profile in capture")
	}
	t.Logf("profile has %d sample types, %d samples", len(cap.Profile.SampleType), len(cap.Profile.Sample))
	for i, st := range cap.Profile.SampleType {
		t.Logf("  sample_type[%d]: %s (%s)", i, st.Type, st.Unit)
	}
	t.Logf("verified capture %s: %d kernels, %d files", path, len(code.Kernels), len(cap.Files))
}
