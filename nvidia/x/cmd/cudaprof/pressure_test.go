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

const sampleSASSWithPressure = `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                                                                                         // |   1   |        |        |         |
        /*0010*/                   S2R R0, SR_TID.X ;                                                                                              // |   2   |        |        |         |
        /*0020*/                   S2UR UR4, SR_CTAID.X ;                                                                                          // |   2   |        |    1   |         |
        /*0030*/                   EXIT ;                                                                                                          // |   1   |        |        |         |

	.section	.text.vector_add,"ax",@progbits
vector_add:
.text.vector_add:
        /*0000*/                   MOV R1, c[0x0][0x28] ;                                                                                          // |   1   |        |        |         |
        /*0010*/                   EXIT ;                                                                                                          // |   1   |        |        |         |
`

func testCaptureWithPressure(t *testing.T, withStalls bool) (*cudacapture.Capture, string) {
	t.Helper()

	fnCompute := &profile.Function{ID: 1, Name: "compute.SASS", Filename: "kernels.sass"}
	fnVAdd := &profile.Function{ID: 2, Name: "vector_add.SASS", Filename: "kernels.sass"}

	locCompute0 := &profile.Location{ID: 1, Address: 0x0000, Line: []profile.Line{{Function: fnCompute, Line: 4}}}
	locCompute1 := &profile.Location{ID: 2, Address: 0x0010, Line: []profile.Line{{Function: fnCompute, Line: 5}}}
	locCompute2 := &profile.Location{ID: 3, Address: 0x0020, Line: []profile.Line{{Function: fnCompute, Line: 6}}}
	locCompute3 := &profile.Location{ID: 4, Address: 0x0030, Line: []profile.Line{{Function: fnCompute, Line: 7}}}

	locVAdd0 := &profile.Location{ID: 5, Address: 0x0000, Line: []profile.Line{{Function: fnVAdd, Line: 11}}}
	locVAdd1 := &profile.Location{ID: 6, Address: 0x0010, Line: []profile.Line{{Function: fnVAdd, Line: 12}}}

	var p *profile.Profile
	if withStalls {
		p = &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_wait", Unit: "count"},
				{Type: "gpu_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{locCompute1},
					Value:    []int64{75, 5, 100}, // 80 stalls out of 100 total
				},
				{
					Location: []*profile.Location{locCompute2},
					Value:    []int64{15, 5, 50}, // 20 stalls out of 100 total
				},
			},
			Location: []*profile.Location{locCompute0, locCompute1, locCompute2, locCompute3, locVAdd0, locVAdd1},
			Function: []*profile.Function{fnCompute, fnVAdd},
		}
	} else {
		p = &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "gpu_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{locCompute0},
					Value:    []int64{100},
				},
			},
			Location: []*profile.Location{locCompute0, locVAdd0},
			Function: []*profile.Function{fnCompute, fnVAdd},
		}
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
			"kernels.cu":    []byte("void compute() {}\nvoid vector_add() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(sampleSASSWithPressure),
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

func TestRunPressureCommand(t *testing.T) {
	_, capturePath := testCaptureWithPressure(t, false)

	tests := []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{"pressure command", []string{"pressure", capturePath}, false},
		{"pressure with kernel", []string{"pressure", "compute", capturePath}, false},
		{"flag -pressure", []string{"-pressure", capturePath}, false},
		{"flag -pressure with kernel", []string{"-pressure", "compute", capturePath}, false},
		{"flag -pressure=kernel", []string{"-pressure=compute", capturePath}, false},
		{"flag --pressure", []string{"--pressure", capturePath}, false},
		{"flag --pressure=kernel", []string{"--pressure=compute", capturePath}, false},
		{"bare path without command", []string{capturePath}, false},
		{"bare kernel and path", []string{"compute", capturePath}, false},
		{"nonexistent kernel", []string{"pressure", "missing_kernel", capturePath}, true},
		{"nonexistent file", []string{"pressure", "missing.pprof"}, true},
		{"empty args", []string{}, true},
		{"duplicate kernel", []string{"-pressure=compute", "other", capturePath}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := runPressureCommand(tt.args)
			if (err != nil) != tt.wantErr {
				t.Fatalf("runPressureCommand(%v) err = %v, wantErr = %v", tt.args, err, tt.wantErr)
			}
		})
	}
}

func TestRunInspectPressure(t *testing.T) {
	_, capturePath := testCaptureWithPressure(t, false)

	tests := []struct {
		name    string
		args    []string
		wantErr bool
	}{
		{"inspect --pressure", []string{"--pressure", capturePath}, false},
		{"inspect --pressure with kernel", []string{"--pressure", "compute", capturePath}, false},
		{"inspect --pressure nonexistent kernel", []string{"--pressure", "nonexistent", capturePath}, true},
		{"invalid arg count", []string{"--pressure"}, true},
		{"inspect command router", []string{"--pressure", capturePath}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			if strings.HasPrefix(tt.name, "inspect command router") {
				err = runInspectCommand(tt.args)
			} else {
				err = runInspectPressure(tt.args, cudacapture.Limits{})
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("%s: err = %v, wantErr = %v", tt.name, err, tt.wantErr)
			}
		})
	}
}

func TestPrintCapturePressure(t *testing.T) {
	t.Run("without stalls", func(t *testing.T) {
		cap, _ := testCaptureWithPressure(t, false)
		var buf bytes.Buffer
		if err := printCapturePressure(&buf, cap, ""); err != nil {
			t.Fatalf("printCapturePressure failed: %v", err)
		}
		out := buf.String()

		// Verify table columns
		if !strings.Contains(out, "PC") || !strings.Contains(out, "GPR") || !strings.Contains(out, "PRED") ||
			!strings.Contains(out, "UGPR") || !strings.Contains(out, "UPRED") || !strings.Contains(out, "INSTRUCTION") {
			t.Errorf("missing expected headers in output:\n%s", out)
		}
		if strings.Contains(out, "STALL%") {
			t.Errorf("STALL%% header should not appear when profile has no stall metrics:\n%s", out)
		}

		// Verify kernel headers
		if !strings.Contains(out, "Kernel: compute") || !strings.Contains(out, "Kernel: vector_add") {
			t.Errorf("missing kernel headers in output:\n%s", out)
		}

		// Verify instructions and live register counts
		if !strings.Contains(out, "0x0000") || !strings.Contains(out, "LDC R1, c[0x0][0x37c];") {
			t.Errorf("missing instruction at 0x0000 in output:\n%s", out)
		}
		if !strings.Contains(out, "0x0020") || !strings.Contains(out, "S2UR UR4, SR_CTAID.X;") {
			t.Errorf("missing instruction at 0x0020 in output:\n%s", out)
		}
	})

	t.Run("with kernel filter", func(t *testing.T) {
		cap, _ := testCaptureWithPressure(t, false)
		var buf bytes.Buffer
		if err := printCapturePressure(&buf, cap, "compute"); err != nil {
			t.Fatalf("printCapturePressure failed: %v", err)
		}
		out := buf.String()

		if !strings.Contains(out, "Kernel: compute") {
			t.Errorf("expected Kernel: compute in output:\n%s", out)
		}
		if strings.Contains(out, "Kernel: vector_add") {
			t.Errorf("vector_add should be filtered out:\n%s", out)
		}
	})

	t.Run("with hardware stall metrics", func(t *testing.T) {
		cap, _ := testCaptureWithPressure(t, true)
		var buf bytes.Buffer
		if err := printCapturePressure(&buf, cap, "compute"); err != nil {
			t.Fatalf("printCapturePressure failed: %v", err)
		}
		out := buf.String()

		// Verify STALL% header appears
		if !strings.Contains(out, "STALL%") {
			t.Fatalf("expected STALL%% column when stall metrics are available:\n%s", out)
		}

		// Verify stall percentages:
		// 0x0010: 80 stalls out of 100 -> 80.0%
		// 0x0020: 20 stalls out of 100 -> 20.0%
		// 0x0000: 0 stalls -> 0.0%
		lines := strings.Split(out, "\n")
		var found0010, found0020, found0000 bool
		for _, line := range lines {
			if strings.Contains(line, "0x0010") {
				found0010 = true
				if !strings.Contains(line, "80.0%") {
					t.Errorf("line for 0x0010 want 80.0%%, got: %s", line)
				}
				// GPR count = 2
				if !strings.Contains(line, "2") {
					t.Errorf("line for 0x0010 want GPR 2, got: %s", line)
				}
			}
			if strings.Contains(line, "0x0020") {
				found0020 = true
				if !strings.Contains(line, "20.0%") {
					t.Errorf("line for 0x0020 want 20.0%%, got: %s", line)
				}
				// GPR count = 2, UGPR count = 1
				if !strings.Contains(line, "2") || !strings.Contains(line, "1") {
					t.Errorf("line for 0x0020 want GPR 2 UGPR 1, got: %s", line)
				}
			}
			if strings.Contains(line, "0x0000") {
				found0000 = true
				if !strings.Contains(line, "0.0%") {
					t.Errorf("line for 0x0000 want 0.0%%, got: %s", line)
				}
				// GPR count = 1
				if !strings.Contains(line, "1") {
					t.Errorf("line for 0x0000 want GPR 1, got: %s", line)
				}
			}
		}

		if !found0000 || !found0010 || !found0020 {
			t.Errorf("missing expected instruction lines in output:\n%s", out)
		}
	})
}

func TestPrintCapturePressure_SeparateLRMFile(t *testing.T) {
	// Standard SASS disassembly without count comments
	sassClean := `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   EXIT ;
`
	// Separate .lrm file with life-range mode count comments
	lrmCount := `
.section .text.compute,"ax",@progbits
compute:
/*0000*/ LDC R1, c[0x0][0x37c] ; // | 1 | | |
/*0010*/ S2R R0, SR_TID.X ; // | 2 | | |
/*0020*/ EXIT ; // | 1 | | |
`

	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		},
		Code: cudacapture.CodeFiles{
			SASS: "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.sass": []byte(sassClean),
			"kernels.lrm":  []byte(lrmCount),
		},
	}

	var buf bytes.Buffer
	if err := printCapturePressure(&buf, cap, "compute"); err != nil {
		t.Fatalf("printCapturePressure with separate .lrm failed: %v", err)
	}

	out := buf.String()
	if !strings.Contains(out, "0x0010") || !strings.Contains(out, "S2R R0, SR_TID.X;") {
		t.Errorf("expected instruction 0x0010 in output:\n%s", out)
	}
	// Verify GPR is 2 on 0x0010
	foundLine := false
	for _, l := range strings.Split(out, "\n") {
		if strings.Contains(l, "0x0010") {
			foundLine = true
			if !strings.Contains(l, "2") {
				t.Errorf("line for 0x0010 want GPR 2, got: %s", l)
			}
		}
	}
	if !foundLine {
		t.Errorf("instruction line 0x0010 missing:\n%s", out)
	}
}

func TestPrintCapturePressure_Errors(t *testing.T) {
	t.Run("nil capture", func(t *testing.T) {
		err := printCapturePressure(&bytes.Buffer{}, nil, "")
		if err == nil {
			t.Fatal("expected error for nil capture")
		}
	})

	t.Run("empty capture without CUBIN or SASS", func(t *testing.T) {
		cap := &cudacapture.Capture{Files: map[string][]byte{}}
		err := printCapturePressure(&bytes.Buffer{}, cap, "")
		if err == nil {
			t.Fatal("expected error for empty capture")
		}
		if !strings.Contains(err.Error(), "neither CUBIN nor SASS") {
			t.Errorf("unexpected error message: %v", err)
		}
	})

	t.Run("SASS without life ranges and no CUBIN", func(t *testing.T) {
		cap := &cudacapture.Capture{
			Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
			Files: map[string][]byte{
				"kernels.sass": []byte(".section .text.k\n/*0000*/ MOV R1, R2;\n"),
			},
		}
		err := printCapturePressure(&bytes.Buffer{}, cap, "")
		if err == nil {
			t.Fatal("expected error when SASS has no life ranges and CUBIN is absent")
		}
	})
}
