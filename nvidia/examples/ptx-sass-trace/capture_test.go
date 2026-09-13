package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatest"
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

	// Verify SIMD vs scalar opcodes via cudasass.AssertOpcodes if SASS is present.
	if sassData, ok := cap.Files[cap.Code.SASS]; ok && len(sassData) > 0 {
		disasm, err := cudasass.ParseString(string(sassData))
		if err == nil {
			if disasm.Function("ds4_iq2_simd") != nil {
				if err := disasm.AssertOpcodes("ds4_iq2_simd", []string{"IDP4A"}, []string{"SHR"}); err != nil {
					t.Errorf("AssertOpcodes(ds4_iq2_simd): %v", err)
				}
			}
			if disasm.Function("ds4_iq2_scalar") != nil {
				if err := disasm.AssertOpcodes("ds4_iq2_scalar", []string{"IMAD"}, []string{"IDP4A"}); err != nil {
					t.Errorf("AssertOpcodes(ds4_iq2_scalar): %v", err)
				}
			}
		}
	}
}

func TestIsArithmeticOpcode(t *testing.T) {
	tests := []struct {
		opcode string
		want   bool
	}{
		{"fma.rn.f32", true},
		{"add.f32", true},
		{"sub.f32", true},
		{"mul.lo.s32", true},
		{"dp4a.s32.s32", true},
		{"shfl.sync.down.b32", true},
		{"ld.global.f32", true},
		{"st.global.f32", true},
		{"rsqrt.approx.f32", true},
		{"cvt.f32.f16", true},
		{"bar.sync", false},
		{"bra.uni", false},
		{"ret", false},
		{"nop", false},
	}

	for _, tt := range tests {
		got := isArithmeticOpcode(tt.opcode)
		if got != tt.want {
			t.Errorf("isArithmeticOpcode(%q) = %v, want %v", tt.opcode, got, tt.want)
		}
	}
}

func TestKernelsNVRTCCompile(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	src, err := sources.ReadFile("kernels.cu")
	if err != nil {
		t.Fatalf("read kernels.cu from embedded sources: %v", err)
	}
	if _, err := cudatest.CompileSource(string(src), major, minor); err != nil {
		t.Fatalf("kernels.cu does not compile with NVRTC: %v", err)
	}
}
