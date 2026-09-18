package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestExplainCommand(t *testing.T) {
	out := captureStdout(t, func() error {
		return runExplainCommand([]string{"@%p1", "fma.rn.ftz.f32", "%f1,", "%f2,", "%f3,", "%f4;"})
	})

	if !strings.Contains(out, "### `fma`") {
		t.Errorf("output missing `fma` header:\n%s", out)
	}
	if !strings.Contains(out, "Fused multiply-add") {
		t.Errorf("output missing summary:\n%s", out)
	}
	if !strings.Contains(out, ".rn") || !strings.Contains(out, ".ftz") || !strings.Contains(out, ".f32") {
		t.Errorf("output missing modifiers:\n%s", out)
	}
}

func TestInspectPTX(t *testing.T) {
	ptxText := `.version 8.0
.target sm_90
.address_size 64

.visible .entry saxpy(.param .u64 a) {
	ld.global.f32 %f1, [%rd1];
	fma.rn.f32    %f2, %f1, %f0, %f1;
	st.global.f32 [%rd2], %f2;
	ret;
}`

	fn := &profile.Function{ID: 1, Name: "saxpy", Filename: "saxpy.cu"}
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
			Source: "saxpy.cu",
			PTX:    "saxpy.ptx",
			CUBIN:  "saxpy.cubin",
			SASS:   "saxpy.sass",
		},
		Files: map[string][]byte{
			"saxpy.cu":    []byte("void saxpy() {}\n"),
			"saxpy.ptx":   []byte(ptxText),
			"saxpy.cubin": []byte("CUBIN_DATA"),
			"saxpy.sass":  []byte(".section .text.saxpy\n/* 0000 */ RET;\n"),
		},
	}

	tmpDir := t.TempDir()
	capPath := filepath.Join(tmpDir, "test.pprof")
	f, err := os.Create(capPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	f.Close()

	// Test inspect --ptx
	out := captureStdout(t, func() error { return runInspectCommand([]string{"--ptx", capPath}) })

	var ptxSum inspectPTXSummary
	if err := json.NewDecoder(strings.NewReader(out)).Decode(&ptxSum); err != nil {
		t.Fatalf("decode inspectPTXSummary: %v", err)
	}
	if ptxSum.Target != "sm_90" {
		t.Errorf("Target = %q, want sm_90", ptxSum.Target)
	}
	if ptxSum.TotalInstructions != 4 {
		t.Errorf("TotalInstructions = %d, want 4", ptxSum.TotalInstructions)
	}
	if ptxSum.Opcodes["fma"] != 1 {
		t.Errorf("Opcodes[fma] = %d, want 1", ptxSum.Opcodes["fma"])
	}
	if ptxSum.Categories["FloatingPoint"] != 1 {
		t.Errorf("Categories[FloatingPoint] = %d, want 1", ptxSum.Categories["FloatingPoint"])
	}
}
