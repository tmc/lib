package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// TestStallFixtureNamesAreReal keeps the diff fixtures inside the vocabulary
// x/cudaprof emits. A fixture is free to invent a sample type, and one that
// does makes the test exercise a shape no capture can have: that is how four
// stall narratives in diff.go stayed green while never firing on real data.
// Builder.SetSampleTypes is the producer's own validator, so it answers this
// without duplicating the list here.
func TestStallFixtureNamesAreReal(t *testing.T) {
	for _, name := range []string{"stall_mem", "stall_wait", "stall_scoreboard"} {
		if err := cudaprof.NewBuilder().SetSampleTypes(name); err != nil {
			t.Errorf("SetSampleTypes(%q) = %v, want a name the producer emits", name, err)
		}
	}
	// Negative control: if the validator ever stopped rejecting, every check
	// above would pass without checking anything.
	if err := cudaprof.NewBuilder().SetSampleTypes("stall_memory_throttle"); err == nil {
		t.Error("SetSampleTypes(\"stall_memory_throttle\") = nil, so this test proves nothing")
	}
}

func createTestCapture(t *testing.T, kernelName, sassText string) string {
	t.Helper()
	fn := &profile.Function{ID: 1, Name: kernelName, Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{100}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	cap := &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("void compute() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(sassText),
		},
	}

	dir := t.TempDir()
	capturePath := filepath.Join(dir, kernelName+".pprof")
	f, err := os.Create(capturePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	return capturePath
}

func TestRunDiffSASSFiles(t *testing.T) {
	baseSASS := `
--- Kernel: gemv ---
/*0100*/ LDG.E.F32 R4, [R2.64] ;
/*0110*/ SHR.B64 R5, R4, 0x8 ;
/*0120*/ IMAD.WIDE R0, R5, R6, R0 ;
/*0130*/ IMAD.WIDE R1, R7, R8, R1 ;
/*0140*/ EXIT ;
`
	targetSASS := `
--- Kernel: gemv ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ LOP3.LUT R6, R4, R5, 0xca ;
/*0030*/ IDP4A.U8.U8 R0, R6, R7, R0 ;
/*0040*/ IDP4A.S8.S8 R1, R8, R9, R1 ;
/*0050*/ EXIT ;
`

	dir := t.TempDir()
	basePath := filepath.Join(dir, "base.sass")
	targetPath := filepath.Join(dir, "target.sass")
	if err := os.WriteFile(basePath, []byte(baseSASS), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(targetPath, []byte(targetSASS), 0644); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--no-color", basePath, targetPath})
	if err != nil {
		t.Fatalf("runDiff failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "=== Kernel: gemv ===") {
		t.Errorf("output missing kernel header: %s", out)
	}
	if !strings.Contains(out, "+ /*0030*/ IDP4A.U8.U8 R0, R6, R7, R0") {
		t.Errorf("output missing + addition for IDP4A: %s", out)
	}
	if !strings.Contains(out, "- /*0120*/ IMAD.WIDE R0, R5, R6, R0") {
		t.Errorf("output missing - deletion for IMAD: %s", out)
	}
	if !strings.Contains(out, "IMAD -> IDP4A") {
		t.Errorf("output missing substitution summary: %s", out)
	}
	if !strings.Contains(out, "SIMD operations:   0 -> 2 (+2)") {
		t.Errorf("output missing SIMD summary: %s", out)
	}
}

func TestRunDiffPprofCaptures(t *testing.T) {
	baseSASS := `
--- Kernel: kernelA ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ FFMA R0, R4, R5, R0 ;
/*0030*/ EXIT ;
`
	targetSASS := `
--- Kernel: kernelA ---
/*0010*/ LDG.E.F32 R4, [R2.64] ;
/*0020*/ STS.32 [R1+0x10], R4 ;
/*0030*/ LDS.32 R4, [R1+0x10] ;
/*0040*/ FFMA R0, R4, R5, R0 ;
/*0050*/ EXIT ;
`

	baseCapture := createTestCapture(t, "base", baseSASS)
	targetCapture := createTestCapture(t, "target", targetSASS)

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--no-color", baseCapture, targetCapture})
	if err != nil {
		t.Fatalf("runDiff with pprof failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "=== Kernel: kernelA ===") {
		t.Errorf("output missing kernel header: %s", out)
	}
	if !strings.Contains(out, "+ /*0020*/ STS.32 [R1+0x10], R4") {
		t.Errorf("output missing + STS addition: %s", out)
	}
	if !strings.Contains(out, "+ /*0030*/ LDS.32 R4, [R1+0x10]") {
		t.Errorf("output missing + LDS addition: %s", out)
	}
	if !strings.Contains(out, "Spill operations: +2") {
		t.Errorf("output missing spill operation count: %s", out)
	}
}

func TestRunDiffKernelFilter(t *testing.T) {
	baseSASS := `
--- Kernel: k1 ---
/*0010*/ IMAD R0, R1, R2, R0 ;
/*0020*/ EXIT ;

--- Kernel: k2 ---
/*0010*/ LDG.E.F32 R0, [R1] ;
/*0020*/ EXIT ;
`
	targetSASS := `
--- Kernel: k1 ---
/*0010*/ IDP4A.U8.U8 R0, R1, R2, R0 ;
/*0020*/ EXIT ;

--- Kernel: k2 ---
/*0010*/ LDG.E.F32 R0, [R1] ;
/*0020*/ EXIT ;
`

	dir := t.TempDir()
	basePath := filepath.Join(dir, "base.sass")
	targetPath := filepath.Join(dir, "target.sass")
	os.WriteFile(basePath, []byte(baseSASS), 0644)
	os.WriteFile(targetPath, []byte(targetSASS), 0644)

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--no-color", "--kernel=k1", basePath, targetPath})
	if err != nil {
		t.Fatalf("runDiff failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "=== Kernel: k1 ===") {
		t.Errorf("missing k1: %s", out)
	}
	if strings.Contains(out, "=== Kernel: k2 ===") {
		t.Errorf("k2 should have been filtered out: %s", out)
	}

	// Filter for non-existent kernel
	buf.Reset()
	err = runDiff(&buf, []string{"--no-color", "--kernel=k999", basePath, targetPath})
	if err == nil {
		t.Error("expected error for non-existent kernel filter")
	}
}

func TestRunDiffErrors(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{
			name: "no arguments",
			args: []string{},
		},
		{
			name: "single argument",
			args: []string{"base.sass"},
		},
		{
			name: "missing file",
			args: []string{"nonexistent_base.sass", "nonexistent_target.sass"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			if err := runDiff(&buf, tt.args); err == nil {
				t.Errorf("runDiff(%v) expected error, got nil", tt.args)
			}
		})
	}
}

func TestRunDiffExplain(t *testing.T) {
	createProfilingCapture := func(t *testing.T, kernelName string, durationNs int64, memStalls, barrierStalls int64) string {
		t.Helper()
		fn := &profile.Function{ID: 1, Name: kernelName, Filename: "kernels.cu"}
		loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
		p := &profile.Profile{
			SampleType: []*profile.ValueType{
				// These must be names x/cudaprof actually emits. The fixture
				// used to declare stall_memory_throttle and stall_barrier,
				// which it does not, so this test exercised a shape no capture
				// can have and asserted a narrative only that shape reached.
				{Type: "gpu_time", Unit: "nanoseconds"},
				{Type: "stall_mem", Unit: "samples"},
				{Type: "stall_wait", Unit: "samples"},
			},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{loc},
					Value:    []int64{durationNs, memStalls, barrierStalls},
				},
			},
			Location: []*profile.Location{loc},
			Function: []*profile.Function{fn},
		}

		cap := &cudacapture.Capture{
			Profile: p,
			Code: cudacapture.CodeFiles{
				Source: "kernels.cu",
				PTX:    "kernels.ptx",
				CUBIN:  "kernels.cubin",
				SASS:   "kernels.sass",
			},
			Files: map[string][]byte{
				"kernels.cu":    []byte("void compute() {}\n"),
				"kernels.ptx":   []byte("// PTX\n"),
				"kernels.cubin": []byte("CUBIN_MOCK"),
				"kernels.sass":  []byte("--- Kernel: " + kernelName + " ---\n/*0000*/ NOP ;\n/*0010*/ EXIT ;\n"),
			},
		}

		dir := t.TempDir()
		path := filepath.Join(dir, kernelName+".pprof")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if err := cudacapture.Write(f, cap); err != nil {
			t.Fatal(err)
		}
		return path
	}

	basePath := createProfilingCapture(t, "gemv_kernel", 100000, 5000, 1000)
	targetPath := createProfilingCapture(t, "gemv_kernel", 75000, 2000, 950)

	var buf bytes.Buffer
	err := runDiff(&buf, []string{"--explain", basePath, targetPath})
	if err != nil {
		t.Fatalf("runDiff(--explain) failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "CUDA Profile Diff & Optimization Explanation") {
		t.Errorf("output missing title: %s", out)
	}
	if !strings.Contains(out, "Execution Duration Delta") {
		t.Errorf("output missing duration delta section: %s", out)
	}
	if !strings.Contains(out, "stall_mem") {
		t.Errorf("output missing stall_mem: %s", out)
	}
	// The stall table is the part that works on real captures: 5000 -> 2000
	// samples is a 60% drop, and the row has to carry it.
	if !strings.Contains(out, "-60.0%") {
		t.Errorf("output missing the stall_mem shift: %s", out)
	}
	if !strings.Contains(out, "1.33×") && !strings.Contains(out, "1.33x") {
		t.Errorf("output missing expected speedup calculation: %s", out)
	}
}
