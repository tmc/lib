package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestCaptureEnvironmentExportsNVTXInjection covers the gap that made the NVTX
// tier collect nothing while reporting nvtx_markers:true. LD_PRELOAD gets CUPTI
// collecting; NVTX_INJECTION64_PATH is what gets anything emitting. A target
// using nvidia/x/nvtx returns from initInjection with enabled false and no
// error when it is unset, so the whole tier is silently empty.
//
// The assertions are about what is exported, which is a producer-side claim.
// The property that actually matters -- a capture collected markers -- needs a
// device and belongs with the injection tests, not here.
func TestCaptureEnvironmentExportsNVTXInjection(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("NVTX injection is exported on linux only")
	}
	root := t.TempDir()
	lib := filepath.Join(root, "extras", "CUPTI", "lib64")
	if err := os.MkdirAll(lib, 0o755); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(lib, "libcupti.so.13")
	if err := os.WriteFile(want, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CUDA_HOME", root)
	t.Setenv("CUDAPROF_NVTX_INJECTION", "")

	got := lookupEnv(captureEnvironment(captureOptions{Env: []string{}}, "s", "d"), "NVTX_INJECTION64_PATH")
	if got != want {
		t.Errorf("NVTX_INJECTION64_PATH = %q, want %q", got, want)
	}
}

// TestCaptureEnvironmentKeepsExplicitNVTXInjection: the caller has already
// chosen an injection library, and NVIDIA's own tooling sets this variable.
func TestCaptureEnvironmentKeepsExplicitNVTXInjection(t *testing.T) {
	env := []string{"NVTX_INJECTION64_PATH=/chosen/libnvtx.so"}
	got := lookupEnv(captureEnvironment(captureOptions{Env: env}, "s", "d"), "NVTX_INJECTION64_PATH")
	if got != "/chosen/libnvtx.so" {
		t.Errorf("NVTX_INJECTION64_PATH = %q, want the caller's value", got)
	}
}

// TestCaptureEnvironmentExportsNoUnresolvableNVTXPath is the one that keeps the
// fix honest. Exporting a name the loader cannot open reaches the same silent
// return in initInjection through a different door -- openDynamicLibrary fails,
// enabled stays false, nothing is reported -- and the capture would again be
// empty while the variable looks correctly set.
func TestCaptureEnvironmentExportsNoUnresolvableNVTXPath(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("NVTX injection is exported on linux only")
	}
	t.Setenv("CUDA_HOME", filepath.Join(t.TempDir(), "absent"))
	t.Setenv("CUDA_PATH", "")
	t.Setenv("CUDAPROF_NVTX_INJECTION", "")

	got := lookupEnv(captureEnvironment(captureOptions{Env: []string{}}, "s", "d"), "NVTX_INJECTION64_PATH")
	if got != "" && !strings.HasPrefix(got, "/usr/local/cuda") {
		t.Errorf("NVTX_INJECTION64_PATH = %q, want empty or a real toolkit path", got)
	}
	if got != "" {
		if _, err := os.Stat(got); err != nil {
			t.Errorf("exported NVTX_INJECTION64_PATH = %q, which does not stat: %v", got, err)
		}
	}
}
