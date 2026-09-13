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

func testCaptureWithCFG(t *testing.T) (*cudacapture.Capture, string) {
	t.Helper()
	fn := &profile.Function{ID: 1, Name: "compute", Filename: "kernels.cu"}
	loc := &profile.Location{ID: 1, Line: []profile.Line{{Function: fn, Line: 1}}}
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		Sample:     []*profile.Sample{{Location: []*profile.Location{loc}, Value: []int64{100}}},
		Location:   []*profile.Location{loc},
		Function:   []*profile.Function{fn},
	}

	cfgDOT := `digraph compute {
	bb0 [shape=box, label="[0000] MOV R1, c[0x0][0x28];\n[0010] RET;"];
}
`
	bbcfgDOT := `digraph compute {
	bb0 [shape=box, label="[0000] MOV R1, c[0x0][0x28];"];
	bb1 [shape=box, label="[0010] RET;"];
	bb0 -> bb1;
}
`

	cap := &cudacapture.Capture{
		Profile: p,
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
			CFG:    "kernels.cfg.dot",
			BBCFG:  "kernels.bbcfg.dot",
		},
		Files: map[string][]byte{
			"kernels.cu":        []byte("void compute() {}\n"),
			"kernels.ptx":       []byte("// PTX\n"),
			"kernels.cubin":     []byte("CUBIN_MOCK"),
			"kernels.sass":      []byte(".section .text.compute\n/* 0x0000 */ MOV R1, c[0x0][0x28];\n/* 0x0010 */ RET;\n"),
			"kernels.cfg.dot":   []byte(cfgDOT),
			"kernels.bbcfg.dot": []byte(bbcfgDOT),
		},
	}

	dir := t.TempDir()
	capturePath := filepath.Join(dir, "capture.pprof")
	f, err := os.Create(capturePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := cudacapture.Write(f, cap); err != nil {
		t.Fatal(err)
	}
	return cap, capturePath
}

func TestExtractIncludesDOTFiles(t *testing.T) {
	_, capturePath := testCaptureWithCFG(t)
	extractDir := filepath.Join(t.TempDir(), "extracted")

	if err := extract(extractDir, capturePath, cudacapture.Limits{}); err != nil {
		t.Fatalf("extract failed: %v", err)
	}

	for _, name := range []string{"kernels.cu", "kernels.ptx", "kernels.cubin", "kernels.sass", "kernels.cfg.dot", "kernels.bbcfg.dot"} {
		path := filepath.Join(extractDir, name)
		if _, err := os.Stat(path); err != nil {
			t.Errorf("extracted file %q missing: %v", name, err)
		}
	}

	cfgContent, err := os.ReadFile(filepath.Join(extractDir, "kernels.cfg.dot"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(cfgContent), "digraph compute") {
		t.Errorf("unexpected kernels.cfg.dot content: %s", cfgContent)
	}
}

func TestPrintCaptureCFG(t *testing.T) {
	cap, _ := testCaptureWithCFG(t)

	var bufHyper bytes.Buffer
	if err := printCaptureCFG(&bufHyper, cap, "", false); err != nil {
		t.Fatalf("printCaptureCFG hyperblocks failed: %v", err)
	}
	if !strings.Contains(bufHyper.String(), "[0000] MOV R1") || strings.Contains(bufHyper.String(), "bb0 -> bb1") {
		t.Errorf("unexpected hyperblock CFG output: %s", bufHyper.String())
	}

	var bufBB bytes.Buffer
	if err := printCaptureCFG(&bufBB, cap, "", true); err != nil {
		t.Fatalf("printCaptureCFG basic blocks failed: %v", err)
	}
	if !strings.Contains(bufBB.String(), "bb0 -> bb1") {
		t.Errorf("expected edge bb0 -> bb1 in basic block CFG, got: %s", bufBB.String())
	}

	var bufFiltered bytes.Buffer
	if err := printCaptureCFG(&bufFiltered, cap, "compute", false); err != nil {
		t.Fatalf("printCaptureCFG filtered failed: %v", err)
	}
	if !strings.Contains(bufFiltered.String(), "digraph compute") {
		t.Errorf("expected digraph compute in filtered output, got: %s", bufFiltered.String())
	}

	var bufNotFound bytes.Buffer
	if err := printCaptureCFG(&bufNotFound, cap, "nonexistent", false); err == nil {
		t.Fatal("expected error for nonexistent kernel")
	}
}

func TestRunCFGCommand(t *testing.T) {
	_, capturePath := testCaptureWithCFG(t)

	for _, tc := range []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{"cfg command", []string{"cfg", capturePath}, false},
		{"cfg with kernel", []string{"cfg", "compute", capturePath}, false},
		{"cfg with basic blocks", []string{"cfg", "--bb", capturePath}, false},
		{"cfg with basic blocks and kernel", []string{"cfg", "--bb", "compute", capturePath}, false},
		{"flag -cfg", []string{"-cfg", capturePath}, false},
		{"flag -cfg with kernel", []string{"-cfg", "compute", capturePath}, false},
		{"flag -cfg=kernel", []string{"-cfg=compute", capturePath}, false},
		{"bbcfg command", []string{"bbcfg", capturePath}, false},
		{"flag -bbcfg", []string{"-bbcfg", capturePath}, false},
		{"flag -bbcfg=kernel", []string{"-bbcfg=compute", capturePath}, false},
		{"nonexistent file", []string{"cfg", "missing.pprof"}, true},
		{"empty args", []string{}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := runCFGCommand(tc.args)
			if (err != nil) != tc.wantErr {
				t.Fatalf("runCFGCommand(%v) err = %v, wantErr = %v", tc.args, err, tc.wantErr)
			}
		})
	}
}

func TestRunInspectCFG(t *testing.T) {
	_, capturePath := testCaptureWithCFG(t)

	for _, tc := range []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{"inspect --cfg", []string{"--cfg", capturePath}, false},
		{"inspect --cfg with kernel", []string{"--cfg", "compute", capturePath}, false},
		{"inspect --bbcfg", []string{"--bbcfg", capturePath}, false},
		{"inspect --bbcfg with kernel", []string{"--bbcfg", "compute", capturePath}, false},
		{"invalid arg count", []string{"--cfg"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := runInspectCFG(tc.args, cudacapture.Limits{})
			if (err != nil) != tc.wantErr {
				t.Fatalf("runInspectCFG(%v) err = %v, wantErr = %v", tc.args, err, tc.wantErr)
			}
		})
	}
}
