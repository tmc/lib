package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestWriteCaptureDoesNotLoadCUDA(t *testing.T) {
	if os.Getenv("CUDAPROF_TEST_ARTIFACT_PROVENANCE") != "1" {
		library := ""
		switch runtime.GOOS {
		case "linux":
			library = "libc.so.6"
		case "darwin":
			library = "/usr/lib/libSystem.B.dylib"
		default:
			t.Skip("no inert system library fixture")
		}
		command := exec.Command(os.Args[0], "-test.run=^TestWriteCaptureDoesNotLoadCUDA$")
		for _, entry := range os.Environ() {
			if !strings.HasPrefix(entry, "CUDA_LIB_PATH=") && !strings.HasPrefix(entry, "CODEGEN_SKIP_BINDINGS=") && !strings.HasPrefix(entry, "CUDAPROF_TEST_ARTIFACT_PROVENANCE=") {
				command.Env = append(command.Env, entry)
			}
		}
		// An accidental lazy driver call loads this inert system library, which
		// exposes no CUDA entrypoints and therefore cannot initialize a real GPU.
		command.Env = append(command.Env, "CUDAPROF_TEST_ARTIFACT_PROVENANCE=1", "CUDA_LIB_PATH="+library)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("artifact publication: %v\n%s", err, output)
		}
		return
	}
	if driver.Loaded() {
		t.Fatal("CUDA driver was loaded before artifact publication")
	}
	for _, tc := range []struct {
		name       string
		capability []byte
	}{
		{"absent", nil},
		{"retained", []byte(`{"device":"recorded remote GPU","device_id":7,"compute_capability":"8.0","producer_extension":{"retained":true}}`)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "capture.pprof")
			result := captureResult{Protocol: 1, Session: "recorded", State: captureStateFinalized, Outcome: captureOutcomeNoCUDA, CaptureSuccess: true, Capability: tc.capability}
			if err := writeCaptureStatus(path, result, captureArtifactBudget{}); err != nil {
				t.Fatal(err)
			}
			if driver.Loaded() {
				t.Fatal("artifact publication loaded the CUDA driver")
			}
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			capture, err := cudacapture.Read(f)
			f.Close()
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(capture.Files["capture-provenance.json"], tc.capability) {
				t.Fatalf("producer provenance changed: %s", capture.Files["capture-provenance.json"])
			}
		})
	}
}
