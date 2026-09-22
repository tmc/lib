package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// testCaptureWithLines is the fused_gate fixture of cudalineinfo with four
// kinds of sample: a full stack whose kernel frame names kernels.go:19 by a
// path the disassembly does not use, a bare SASS frame at the same
// instruction, a bare SASS frame in the prologue, which names no program
// line, and a prologue sample whose kernel frame names the generated file.
func testCaptureWithLines(t *testing.T) (*cudacapture.Capture, string) {
	t.Helper()
	sass, err := os.ReadFile(filepath.Join("..", "..", "cudalineinfo", "testdata", "fused_gate.sass"))
	if err != nil {
		t.Fatal(err)
	}
	// Instruction 32 of fused_gate is at offset 0x1f0, on kernels.go:19,
	// the line that both activates and multiplies in this fixture.
	fnSASS := &profile.Function{ID: 1, Name: "fused_gate.SASS", Filename: "fused_gate.sass"}
	fnKernel := &profile.Function{ID: 2, Name: "fused_gate", Filename: "/src/kernels.go"}
	fnGenerated := &profile.Function{ID: 4, Name: "fused_gate", Filename: "/tmp/fused_gate.cu"}
	fnHost := &profile.Function{ID: 3, Name: "main.gate", Filename: "kernels.go"}
	locSilu := &profile.Location{ID: 1, Line: []profile.Line{{Function: fnSASS, Line: 32}}}
	locSite := &profile.Location{ID: 2, Line: []profile.Line{{Function: fnKernel, Line: 19}}}
	locHost := &profile.Location{ID: 3, Line: []profile.Line{{Function: fnHost, Line: 14}}}
	locPrologue := &profile.Location{ID: 4, Line: []profile.Line{{Function: fnSASS, Line: 1}}}
	locGenerated := &profile.Location{ID: 5, Line: []profile.Line{{Function: fnGenerated, Line: 9}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "pc_samples", Unit: "count"},
			{Type: "stall_wait", Unit: "count"},
		},
		DefaultSampleType: "pc_samples",
		Sample: []*profile.Sample{
			{Location: []*profile.Location{locSilu, locSite, locHost}, Value: []int64{6, 2}},
			{Location: []*profile.Location{locSilu}, Value: []int64{3, 0}},
			{Location: []*profile.Location{locPrologue}, Value: []int64{1, 1}},
			{Location: []*profile.Location{locPrologue, locGenerated}, Value: []int64{2, 0}},
		},
		Location: []*profile.Location{locSilu, locSite, locHost, locPrologue, locGenerated},
		Function: []*profile.Function{fnSASS, fnKernel, fnHost, fnGenerated},
	}
	cap := &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "fused_gate.cu",
			PTX:    "fused_gate.ptx",
			CUBIN:  "fused_gate.cubin",
			SASS:   "fused_gate.sass",
		},
		Files: map[string][]byte{
			"fused_gate.cu":    []byte("// generated\n"),
			"fused_gate.ptx":   []byte("// PTX\n"),
			"fused_gate.cubin": []byte("CUBIN_MOCK"),
			"fused_gate.sass":  sass,
		},
	}
	path := filepath.Join(t.TempDir(), "capture.pprof")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	return cap, path
}

func TestPrintCaptureLines(t *testing.T) {
	cap, _ := testCaptureWithLines(t)
	for _, tt := range []struct {
		kernel, sample string
		want           []string
	}{
		// Without a kernel the generated file is a program file too, so
		// the prologue sample lands on its line rather than elsewhere.
		{"", "", []string{"12 pc_samples by program line", "kernels.go:19  ", "29    9           75.0%", "fused_gate.cu:3           1     1", "fused_gate.cu:9           11    2", "elsewhere: 0 at instructions naming no program line, 0 at unresolved PCs"}},
		{"fused_gate", "", []string{"fused_gate: 12 pc_samples", "29    9           75.0%", "elsewhere: 3 at instructions naming no program line, 0 at unresolved PCs"}},
		{"gate", "stall_wait", []string{"fused_gate: 3 stall_wait", "66.7%"}},
	} {
		var buf bytes.Buffer
		if err := printCaptureLines(&buf, cap, tt.kernel, tt.sample); err != nil {
			t.Fatalf("kernel %q sample %q: %v", tt.kernel, tt.sample, err)
		}
		for _, want := range tt.want {
			if !strings.Contains(buf.String(), want) {
				t.Errorf("kernel %q sample %q: output lacks %q:\n%s", tt.kernel, tt.sample, want, buf.String())
			}
		}
	}
	if err := printCaptureLines(&bytes.Buffer{}, cap, "", "gpu_time"); err == nil {
		t.Error("no error for a sample type the profile lacks")
	}
	if err := printCaptureLines(&bytes.Buffer{}, cap, "missing_kernel", ""); err == nil {
		t.Error("no error for a kernel the disassembly lacks")
	}
}

// TestPrintCaptureLinesModules gives the capture two modules holding the
// same kernel; each module's table counts only the samples labelled with
// its identity, and the kernel filter finds the kernel in both.
func TestPrintCaptureLinesModules(t *testing.T) {
	cap, _ := testCaptureWithLines(t)
	cap.Modules = []cudacapture.Module{
		{ID: "1", Name: "first", Code: cap.Code},
		{ID: "2", Name: "second", Code: cap.Code},
	}
	for i, s := range cap.Profile.Sample {
		s.Label = map[string][]string{"module_id": {[]string{"1", "2"}[i%2]}}
	}
	var buf bytes.Buffer
	if err := printCaptureLines(&buf, cap, "", ""); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"module first\n7 pc_samples", "module second\n5 pc_samples", "29    6           85.7%", "29    3           60.0%"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	if strings.Count(out, "kernels.go:19") != 2 {
		t.Errorf("the site is not in both tables:\n%s", out)
	}
	buf.Reset()
	if err := printCaptureLines(&buf, cap, "fused_gate", ""); err != nil {
		t.Fatal(err)
	}
	if strings.Count(buf.String(), "fused_gate: ") != 2 {
		t.Errorf("kernel filter did not report both modules:\n%s", buf.String())
	}
	cap.Profile.Sample[0].Label = nil
	buf.Reset()
	if err := printCaptureLines(&buf, cap, "", ""); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "1 profile records name no module") {
		t.Errorf("unlabelled sample not reported:\n%s", buf.String())
	}
	if err := printCaptureLines(&bytes.Buffer{}, cap, "missing_kernel", ""); err == nil {
		t.Error("no error for a sample type the profile lacks")
	}
	if err := printCaptureLines(&bytes.Buffer{}, cap, "missing_kernel", ""); err == nil {
		t.Error("no error for a kernel the disassembly lacks")
	}
}

func TestRunLinesCommand(t *testing.T) {
	_, path := testCaptureWithLines(t)
	for _, tt := range []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{"capture only", []string{path}, false},
		{"kernel and capture", []string{"fused_gate", path}, false},
		{"sample flag", []string{"-sample", "stall_wait", path}, false},
		{"unknown sample", []string{"-sample", "gpu_time", path}, true},
		{"missing file", []string{"missing.pprof"}, true},
		{"no args", nil, true},
		{"too many args", []string{"a", "b", path}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := runLinesCommand(tt.args); (err != nil) != tt.wantErr {
				t.Fatalf("runLinesCommand(%v) err = %v, wantErr %v", tt.args, err, tt.wantErr)
			}
		})
	}
}
