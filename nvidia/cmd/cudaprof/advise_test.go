package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
)

const sampleHighPressureSASS = `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;                                                                                         // | 140   |        |        |         |
        /*0010*/                   S2R R0, SR_TID.X ;                                                                                              // | 140   |        |        |         |
        /*0020*/                   EXIT ;                                                                                                          // |   1   |        |        |         |
`

const sampleSpillSASS = `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
        /*0000*/                   STL [R1+0x10], R2 ;                                                                                             // |  32   |        |        |         |
        /*0010*/                   LDL R3, [R1+0x10] ;                                                                                             // |  32   |        |        |         |
        /*0020*/                   EXIT ;                                                                                                          // |   1   |        |        |         |
`

const sampleHighUGPRSASS = `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
        /*0000*/                   S2UR UR4, SR_CTAID.X ;                                                                                          // |  16   |        |   70   |         |
        /*0010*/                   EXIT ;                                                                                                          // |   1   |        |        |         |
`

const sampleRemarksYAML = `--- !Passed
Pass:            Memory
Name:            RemarkMemoryLoadInstructionSelected
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 10, Column: 37 }
Function:        matrix
Args:
  - RemarkId:    4
  - Remark:      Load instruction selected
  - Instruction: TMA Load instruction
...
--- !Failure
Pass:            Memory
Name:            RemarkMemoryTmaLoadFailed
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 14, Column: 5 }
Function:        matrix
Args:
  - RemarkId:    5
  - Remark:      Failed to select TMA load instruction
  - Reason:      non-contiguous stride along dim 1
  - Shape:       '[16, 32]'
...
--- !Failure
Pass:            Tensor-core
Name:            RemarkTensorCoreMMA
DebugLoc:        { File: '/tmp/tile_matrix.cu', Line: 16, Column: 9 }
Function:        matrix
Args:
  - RemarkId:    3
  - Remark:      MMA operation failed to optimize to use Tensor Cores, it is using FMA instructions instead
  - Instruction: FMA
  - Shape:       '[1, 1, 1]'
...
`

func TestAdvice_HighRegisterPressure(t *testing.T) {
	tests := []struct {
		name         string
		sass         string
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
	}{
		{
			name:         "high_gpr_cliff",
			sass:         sampleHighPressureSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityWarning,
			wantFinding:  "140 GPRs",
		},
		{
			name:         "local_spill_points",
			sass:         sampleSpillSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityCritical,
			wantFinding:  "spill point",
		},
		{
			name:         "high_ugpr",
			sass:         sampleHighUGPRSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityWarning,
			wantFinding:  "70 UGPRs",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{
				Code: cudacapture.CodeFiles{
					SASS: "kernels.sass",
				},
				Files: map[string][]byte{
					"kernels.sass": []byte(tt.sass),
				},
			}

			report, err := AnalyzeCapture(cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && item.Severity == tt.wantSeverity &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if item.ActionableRecommendation == "" {
						t.Errorf("item recommendation is empty for %s", item.Finding)
					}
					break
				}
			}

			if !found {
				t.Errorf("did not find expected item with Category=%s, Severity=%s, Finding containing %q; got: %+v",
					tt.wantCategory, tt.wantSeverity, tt.wantFinding, report.Items)
			}
		})
	}
}

func TestAdvice_TMAFailureRemark(t *testing.T) {
	cap := &cudacapture.Capture{
		Files: map[string][]byte{
			"remarks.yaml": []byte(sampleRemarksYAML),
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	// Should contain TMA failure (Memory) and Tensor Core failure (Compute)
	var foundTMA, foundTensorCore bool
	for _, item := range report.Items {
		if item.Category == CategoryMemory && item.Severity == SeverityWarning {
			if strings.Contains(item.Finding, "non-contiguous stride along dim 1") {
				foundTMA = true
				if item.Location.Kernel != "matrix" {
					t.Errorf("TMA item Kernel = %q, want matrix", item.Location.Kernel)
				}
				if item.Location.File != "/tmp/tile_matrix.cu" || item.Location.Line != 14 {
					t.Errorf("TMA item Location = %s:%d, want /tmp/tile_matrix.cu:14", item.Location.File, item.Location.Line)
				}
				if !strings.Contains(item.ActionableRecommendation, "stride") {
					t.Errorf("TMA recommendation missing stride advice: %q", item.ActionableRecommendation)
				}
			}
		}

		if item.Category == CategoryCompute && item.Severity == SeverityWarning {
			if strings.Contains(item.Finding, "Tensor Core") || strings.Contains(item.Finding, "FMA") {
				foundTensorCore = true
				if item.Location.Kernel != "matrix" || item.Location.Line != 16 {
					t.Errorf("TensorCore item Location = %s:%d, want /tmp/tile_matrix.cu:16", item.Location.File, item.Location.Line)
				}
				if item.ActionableRecommendation == "" {
					t.Error("TensorCore recommendation should not be empty")
				}
			}
		}
	}

	if !foundTMA {
		t.Errorf("expected TMA failure warning not found; got: %+v", report.Items)
	}
	if !foundTensorCore {
		t.Errorf("expected TensorCore failure warning not found; got: %+v", report.Items)
	}
}

func TestAdvice_TargetArchitectureFallback(t *testing.T) {
	tests := []struct {
		name         string
		ptx          string
		sass         string
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
	}{
		{
			name: "scalar_global_load_sm90",
			ptx: `.version 9.0
.target sm_90
.address_size 64
.file 1 "saxpy.cu"
.visible .entry saxpy(.param .u64 in1) {
	.reg .f32 %f1;
	.reg .u64 %rd1;
	.loc 1 25 4
	ld.global.f32 %f1, [%rd1];
	ret;
}`,
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityWarning,
			wantFinding:  "Scalar global load",
		},
		{
			name: "scalar_imad_wide_ptx",
			ptx: `.version 9.0
.target sm_90
.address_size 64
.file 1 "index.cu"
.visible .entry calc_index(.param .u64 in1) {
	.reg .u32 %r1, %r2;
	.reg .u64 %rd1;
	.loc 1 12 8
	mad.wide.u32 %rd1, %r1, 4, %rd1;
	ret;
}`,
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityWarning,
			wantFinding:  "Scalar address arithmetic fallback",
		},
		{
			name: "architecture_incompatibility_sm100_on_sm90",
			ptx: `.version 9.0
.target sm_90
.address_size 64
.visible .entry blackwell_kernel() {
	.reg .b32 %r1;
	tcgen05.alloc.cta_group::1.sync.aligned.b32 [%r1], 4;
	ret;
}`,
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityCritical,
			wantFinding:  "requires architecture sm_100",
		},
		{
			name: "scalar_imad_wide_sass",
			sass: `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
        /*0000*/                   IMAD.WIDE R0, R5, 0x4, R0 ;
        /*0010*/                   EXIT ;
`,
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityWarning,
			wantFinding:  "Scalar address arithmetic fallback (IMAD.WIDE)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			files := make(map[string][]byte)
			code := cudacapture.CodeFiles{}
			if tt.ptx != "" {
				code.PTX = "kernels.ptx"
				files["kernels.ptx"] = []byte(tt.ptx)
			}
			if tt.sass != "" {
				code.SASS = "kernels.sass"
				files["kernels.sass"] = []byte(tt.sass)
			}

			cap := &cudacapture.Capture{
				Code:  code,
				Files: files,
			}

			report, err := AnalyzeCapture(cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && item.Severity == tt.wantSeverity &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					break
				}
			}

			if !found {
				t.Errorf("did not find expected item Category=%s, Severity=%s, Finding containing %q; got: %+v",
					tt.wantCategory, tt.wantSeverity, tt.wantFinding, report.Items)
			}
		})
	}
}

func TestAdvice_EmptyAndCleanCapture(t *testing.T) {
	t.Run("nil_capture", func(t *testing.T) {
		_, err := AnalyzeCapture(nil)
		if err == nil {
			t.Error("expected error for nil capture, got nil")
		}
	})

	t.Run("empty_capture", func(t *testing.T) {
		cap := &cudacapture.Capture{}
		report, err := AnalyzeCapture(cap)
		if err != nil {
			t.Fatalf("AnalyzeCapture failed: %v", err)
		}
		if len(report.Items) != 0 {
			t.Errorf("len(report.Items) = %d, want 0", len(report.Items))
		}
		md := report.Markdown()
		if !strings.Contains(md, "No optimization issues") {
			t.Errorf("Markdown() = %q, want clean message", md)
		}
		jsonStr, err := report.JSON()
		if err != nil {
			t.Fatalf("JSON() failed: %v", err)
		}
		var parsed AdviceReport
		if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
			t.Fatalf("unmarshal JSON failed: %v", err)
		}
		if len(parsed.Items) != 0 {
			t.Errorf("len(parsed.Items) = %d, want 0", len(parsed.Items))
		}
	})

	t.Run("clean_vectorized_capture", func(t *testing.T) {
		cleanPTX := `.version 9.0
.target sm_90
.address_size 64
.visible .entry clean_kernel(.param .u64 in1) {
	.reg .v4 .f32 %f<4>;
	.reg .u64 %rd1;
	ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd1];
	ret;
}`
		cleanSASS := `
	.section	.text.clean_kernel,"ax",@progbits
clean_kernel:
.text.clean_kernel:
        /*0000*/                   LDG.E.128 R0, desc[UR4][R2.64] ;                                                                               // |  24   |        |        |         |
        /*0010*/                   EXIT ;                                                                                                          // |   1   |        |        |         |
`
		passedRemarks := `--- !Passed
Pass:            Memory
Name:            RemarkMemoryLoadInstructionSelected
Function:        clean_kernel
Args:
  - Remark:      Load instruction selected
  - Instruction: TMA Load instruction
...`

		cap := &cudacapture.Capture{
			Code: cudacapture.CodeFiles{
				PTX:  "clean.ptx",
				SASS: "clean.sass",
			},
			Files: map[string][]byte{
				"clean.ptx":    []byte(cleanPTX),
				"clean.sass":   []byte(cleanSASS),
				"remarks.yaml": []byte(passedRemarks),
			},
		}

		report, err := AnalyzeCapture(cap)
		if err != nil {
			t.Fatalf("AnalyzeCapture failed: %v", err)
		}
		if len(report.Items) != 0 {
			t.Errorf("expected 0 advice items for clean capture, got %d: %+v", len(report.Items), report.Items)
		}
	})
}

func TestAdvice_MarkdownAndJSON(t *testing.T) {
	report := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityCritical,
				Category: CategoryRegisterPressure,
				Location: Location{
					Kernel: "gemm",
					PC:     0x0040,
					File:   "gemm.cu",
					Line:   42,
				},
				Finding:                  "Register spill point detected at PC 0x0040 (STL)",
				ActionableRecommendation: "Reduce live variable ranges to eliminate local memory spills.",
			},
			{
				Severity: SeverityWarning,
				Category: CategoryMemory,
				Location: Location{
					Kernel: "matrix",
					File:   "matrix.cu",
					Line:   14,
				},
				Finding:                  "TMA instruction selection failed: non-contiguous stride",
				ActionableRecommendation: "Ensure innermost tensor dimension is contiguous.",
			},
		},
	}

	md := report.Markdown()
	if !strings.Contains(md, "# CUDA Kernel Advice Report") {
		t.Errorf("Markdown missing header: %s", md)
	}
	if !strings.Contains(md, "- **Critical**: 1") {
		t.Errorf("Markdown missing Critical count: %s", md)
	}
	if !strings.Contains(md, "- **Warning**: 1") {
		t.Errorf("Markdown missing Warning count: %s", md)
	}
	if !strings.Contains(md, "gemm.cu:42") {
		t.Errorf("Markdown missing location gemm.cu:42: %s", md)
	}

	jsonStr, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON() failed: %v", err)
	}
	var decoded AdviceReport
	if err := json.Unmarshal([]byte(jsonStr), &decoded); err != nil {
		t.Fatalf("decode JSON failed: %v", err)
	}
	if len(decoded.Items) != 2 {
		t.Fatalf("len(decoded.Items) = %d, want 2", len(decoded.Items))
	}
	if decoded.Items[0].Severity != SeverityCritical {
		t.Errorf("item 0 severity = %s, want Critical", decoded.Items[0].Severity)
	}
}

func TestAdvice_CLI(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
		},
		Code: cudacapture.CodeFiles{
			Source: "kernels.cu",
			PTX:    "kernels.ptx",
			CUBIN:  "kernels.cubin",
			SASS:   "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.cu":    []byte("void matrix() {}\n"),
			"kernels.ptx":   []byte("// PTX\n"),
			"kernels.cubin": []byte("CUBIN_MOCK"),
			"kernels.sass":  []byte(".section .text.matrix\n/*0000*/ EXIT;\n"),
			"remarks.yaml":  []byte(sampleRemarksYAML),
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

	// 1. Test 'cudaprof advise capture.pprof'
	origStdout := os.Stdout
	r1, w1, _ := os.Pipe()
	os.Stdout = w1
	err = runAdviseCommand([]string{capPath})
	w1.Close()
	os.Stdout = origStdout
	if err != nil {
		t.Fatalf("runAdviseCommand failed: %v", err)
	}
	var buf1 bytes.Buffer
	buf1.ReadFrom(r1)
	out1 := buf1.String()
	if !strings.Contains(out1, "CUDA Kernel Advice Report") {
		t.Errorf("advise output missing report header: %s", out1)
	}
	if !strings.Contains(out1, "TMA") {
		t.Errorf("advise output missing TMA finding: %s", out1)
	}

	// 2. Test 'cudaprof advise --format=json capture.pprof'
	r2, w2, _ := os.Pipe()
	os.Stdout = w2
	err = runAdviseCommand([]string{"--format=json", capPath})
	w2.Close()
	os.Stdout = origStdout
	if err != nil {
		t.Fatalf("runAdviseCommand(--format=json) failed: %v", err)
	}
	var reportJSON AdviceReport
	if err := json.NewDecoder(r2).Decode(&reportJSON); err != nil {
		t.Fatalf("decode advise json failed: %v", err)
	}
	if len(reportJSON.Items) == 0 {
		t.Error("expected non-empty items in JSON output")
	}

	// 3. Test 'cudaprof inspect --advise capture.pprof'
	r3, w3, _ := os.Pipe()
	os.Stdout = w3
	err = runInspectCommand([]string{"--advise", capPath})
	w3.Close()
	os.Stdout = origStdout
	if err != nil {
		t.Fatalf("runInspectCommand(--advise) failed: %v", err)
	}
	var inspectAdv AdviceReport
	if err := json.NewDecoder(r3).Decode(&inspectAdv); err != nil {
		t.Fatalf("decode inspect --advise failed: %v", err)
	}
	if len(inspectAdv.Items) == 0 {
		t.Error("expected non-empty items in inspect --advise output")
	}
}
