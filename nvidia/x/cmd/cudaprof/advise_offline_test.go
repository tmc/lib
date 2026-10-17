package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// TestAdviseDoesNotLoadCUDA pins advise as an offline analysis. The capture
// records launch geometry but no device identity, which is exactly the case
// that once fell back to opening device 0 on the analyzing host. The child
// points CUDA_LIB_PATH at an inert system library, so any lazy driver call
// would load it and flip driver.Loaded without touching a real GPU.
func TestAdviseDoesNotLoadCUDA(t *testing.T) {
	if os.Getenv("CUDAPROF_TEST_ADVISE_OFFLINE") != "1" {
		library := ""
		switch runtime.GOOS {
		case "linux":
			library = "libc.so.6"
		case "darwin":
			library = "/usr/lib/libSystem.B.dylib"
		default:
			t.Skip("no inert system library fixture")
		}
		cmd := exec.Command(os.Args[0], "-test.run=^TestAdviseDoesNotLoadCUDA$", "-test.v")
		for _, entry := range os.Environ() {
			if !strings.HasPrefix(entry, "CUDA_LIB_PATH=") && !strings.HasPrefix(entry, "CODEGEN_SKIP_BINDINGS=") && !strings.HasPrefix(entry, "CUDAPROF_TEST_ADVISE_OFFLINE=") {
				cmd.Env = append(cmd.Env, entry)
			}
		}
		cmd.Env = append(cmd.Env, "CUDAPROF_TEST_ADVISE_OFFLINE=1", "CUDA_LIB_PATH="+library)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("advise in child: %v\n%s", err, out)
		}
		// A child that ran nothing also exits zero.
		if !strings.Contains(string(out), "--- PASS: TestAdviseDoesNotLoadCUDA") {
			t.Fatalf("child did not run the offline check:\n%s", out)
		}
		return
	}

	if driver.Loaded() {
		t.Fatal("CUDA driver was loaded before advise ran")
	}
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample: []*profile.Sample{{
				Value: []int64{100000},
				Label: map[string][]string{
					"name":    {"starved_kernel"},
					"subtype": {"kernel"},
					"grid":    {"16,1,1"},
					"block":   {"32,1,1"},
				},
			}},
		},
		// The code roles name no device: the PTX has no .target, and there is
		// no provenance or metadata.
		Code: cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"},
		Files: map[string][]byte{
			"kernels.cu":    []byte("__global__ void starved_kernel() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(".section .text.starved_kernel\n/*0000*/ EXIT;\n"),
		},
	}
	path := filepath.Join(t.TempDir(), "capture.pprof")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	for _, args := range [][]string{{path}, {"--format=json", path}} {
		out := captureStdout(t, func() error { return runAdviseCommand(args) })
		if driver.Loaded() {
			t.Fatalf("advise %v loaded the CUDA driver", args)
		}
		if !strings.Contains(out, "does not query the local host") {
			t.Errorf("advise %v output does not disclose the skipped device audits:\n%s", args, out)
		}
	}
}
