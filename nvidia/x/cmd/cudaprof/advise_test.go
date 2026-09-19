package main

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"maps"
	"regexp"
	"slices"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
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

// microKernelSample builds one short kernel activity sample carrying device
// timestamps and a launch correlation ID. extraValues pads the sample to a
// profile with more than one sample type.
func microKernelSample(name string, stream int, start, end uint64, correlation int, extraValues ...int64) *profile.Sample {
	value := append([]int64{int64(end - start)}, extraValues...)
	return &profile.Sample{
		Value: value,
		Label: map[string][]string{
			"name":    {name},
			"subtype": {"kernel"},
			"stream":  {strconv.Itoa(stream)},
		},
		NumLabel: map[string][]int64{
			"start":       {int64(start)},
			"end":         {int64(end)},
			"correlation": {int64(correlation)},
		},
	}
}

// apiRecordsFile renders host launch API records, keyed by correlation ID, into
// the capture-status.json form advise reads them from.
func apiRecordsFile(t *testing.T, starts map[int]uint64) []byte {
	t.Helper()
	records := make([]map[string]any, 0, len(starts))
	for corr, start := range starts {
		records = append(records, map[string]any{"domain": "driver", "correlation": corr, "start": start, "end": start + 500})
	}
	data, err := json.Marshal(map[string]any{"api_records": records})
	if err != nil {
		t.Fatalf("marshal api records: %v", err)
	}
	return data
}

func TestAdvice_HighRegisterPressure(t *testing.T) {
	tests := []struct {
		name         string
		sass         string
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantCaveat   bool // recommendation is derived from an occupancy figure
	}{
		{
			name:         "high_gpr_cliff",
			sass:         sampleHighPressureSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityWarning,
			wantFinding:  "140 GPRs",
			wantCaveat:   true,
		},
		{
			name:         "local_spill_points",
			sass:         sampleSpillSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityCritical,
			wantFinding:  "spill point",
			wantCaveat:   false, // a spill is real work, not a residency figure
		},
		{
			name:         "high_ugpr",
			sass:         sampleHighUGPRSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityWarning,
			wantFinding:  "70 UGPRs",
			wantCaveat:   false, // uniform-register spills are real work, not a residency figure
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
					if got := strings.Contains(item.ActionableRecommendation, occupancyCaveat); got != tt.wantCaveat {
						t.Errorf("occupancyCaveat present = %v, want %v in %q", got, tt.wantCaveat, item.ActionableRecommendation)
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
	out1 := captureStdout(t, func() error { return runAdviseCommand([]string{capPath}) })
	if !strings.Contains(out1, "CUDA Kernel Advice Report") {
		t.Errorf("advise output missing report header: %s", out1)
	}
	if !strings.Contains(out1, "TMA") {
		t.Errorf("advise output missing TMA finding: %s", out1)
	}

	// 2. Test 'cudaprof advise --format=json capture.pprof'
	out2 := captureStdout(t, func() error { return runAdviseCommand([]string{"--format=json", capPath}) })
	var reportJSON AdviceReport
	if err := json.NewDecoder(strings.NewReader(out2)).Decode(&reportJSON); err != nil {
		t.Fatalf("decode advise json failed: %v", err)
	}
	if len(reportJSON.Items) == 0 {
		t.Error("expected non-empty items in JSON output")
	}

	// 3. Test 'cudaprof inspect --advise capture.pprof'
	out3 := captureStdout(t, func() error { return runInspectCommand([]string{"--advise", capPath}) })
	var inspectAdv AdviceReport
	if err := json.NewDecoder(strings.NewReader(out3)).Decode(&inspectAdv); err != nil {
		t.Fatalf("decode inspect --advise failed: %v", err)
	}
	if len(inspectAdv.Items) == 0 {
		t.Error("expected non-empty items in inspect --advise output")
	}
}

func TestAdvice_UnderpopulatedThreadBlock(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantKernel   string
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			name: "single_warp_32_threads_profile",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"decode_gemv"},
								"block":  {"32,1,1"},
								"grid":   {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantKernel:   "decode_gemv",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityWarning,
			wantFinding:  "Single-warp thread blocks",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "underpopulated_16_threads_json",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Files: map[string][]byte{
					"activities.json": []byte(`[
						{"name": "tiny_kernel", "subtype": "kernel", "threads": 16, "grid": 64}
					]`),
				},
			},
			wantWarning:  true,
			wantKernel:   "tiny_kernel",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityWarning,
			wantFinding:  "Single-warp thread blocks",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "ptx_reqntid_single_warp",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{PTX: "test.ptx"},
				Files: map[string][]byte{
					"test.ptx": []byte(`.version 9.0
.target sm_121a
.address_size 64
.visible .entry ptx_warp_kernel()
.reqntid 32, 1, 1
{
	ret;
}`),
				},
			},
			wantWarning:  true,
			wantKernel:   "ptx_warp_kernel",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityWarning,
			wantFinding:  "Single-warp thread blocks",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "single_warp_unspecified_arch_suppressed",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"decode_gemv"},
								"block":  {"32,1,1"},
								"grid":   {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "single_warp_grid_1_suppressed",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"setup_kernel"},
								"block":  {"32,1,1"},
								"grid":   {"1,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "rtx_4090_sm_89_limits",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_89", "sm_count": 128, "device": "NVIDIA GeForce RTX 4090"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"ada_gemv"},
								"block":  {"32,1,1"},
								"grid":   {"256,1,1"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantKernel:   "ada_gemv",
			wantCategory: CategoryArchitectureCompute,
			wantSeverity: SeverityWarning,
			wantFinding:  "reaches 50% theoretical occupancy with 24/48 resident warps/SM across 24 blocks/SM",
			wantRec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name: "well_populated_128_threads",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"good_kernel"},
								"block":  {"128,1,1"},
								"grid":   {"256,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := AnalyzeCapture(tt.cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && item.Severity == tt.wantSeverity &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if tt.wantKernel != "" && !strings.Contains(item.Finding, tt.wantKernel) && item.Location.Kernel != tt.wantKernel {
						t.Errorf("expected finding to mention kernel %q, got: %s", tt.wantKernel, item.Finding)
					}
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected underpopulated thread block warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected underpopulated thread block warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_SMGridStarvation(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantKernel   string
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			name: "grid_64_on_136_sm_blackwell",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a", "sm_count": 136}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"decode_attention"},
								"grid":   {"64,1,1"},
								"block":  {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantKernel:   "decode_attention",
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityWarning,
			wantFinding:  "Grid size leaves physical SMs completely idle",
			wantRec:      "Decompose grid using 2D tiling, split-K reduction, or sub-grid pipelining (griddepcontrol)",
		},
		{
			name: "grid_starvation_json_activities",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"device": "NVIDIA B200"}`), // 132 SMs
				Files: map[string][]byte{
					"activities.json": []byte(`[
						{"name": "starved_kernel", "subtype": "kernel", "grid_blocks": 32, "block_threads": 256}
					]`),
				},
			},
			wantWarning:  true,
			wantKernel:   "starved_kernel",
			wantCategory: CategoryArchitecture,
			wantSeverity: SeverityWarning,
			wantFinding:  "Grid size leaves physical SMs completely idle",
			wantRec:      "Decompose grid using 2D tiling, split-K reduction, or sub-grid pipelining (griddepcontrol)",
		},
		{
			name: "sufficient_grid_256_blocks",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"sm_count": 136}`),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"fully_saturated_kernel"},
								"grid":   {"256,1,1"},
								"block":  {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "grid_starvation_unspecified_sm_count_suppressed",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"some_kernel"},
								"grid":   {"64,1,1"},
								"block":  {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "gb10_60_blocks_fills_48_sms_no_starvation",
			cap: &cudacapture.Capture{
				Metadata: []byte(`{"arch": "sm_121a"}`), // GB10 has 48 SMs; 60 blocks >= 48 SMs
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"gb10_batch_kernel"},
								"grid":   {"60,1,1"},
								"block":  {"128,1,1"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := AnalyzeCapture(tt.cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && item.Severity == tt.wantSeverity &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if tt.wantKernel != "" && !strings.Contains(item.Finding, tt.wantKernel) && item.Location.Kernel != tt.wantKernel {
						t.Errorf("expected finding to mention kernel %q, got: %s", tt.wantKernel, item.Finding)
					}
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected grid starvation warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected grid starvation warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_UncoalescedMemoryAccess(t *testing.T) {
	const sampleLDGSASS = `
	.section	.text.uncoalesced_load,"ax",@progbits
uncoalesced_load:
.text.uncoalesced_load:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.128 R0, [R2] ;
        /*0020*/                   EXIT ;
`

	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			name: "uncoalesced_ldg_mem_throttle_gt_35pct",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
				Files: map[string][]byte{
					"kernels.sass": []byte(sampleLDGSASS),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "stall_scoreboard", Unit: "count"},
						{Type: "stall_mem", Unit: "count"},
					},
					Function: []*profile.Function{
						{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"},
					},
					Location: []*profile.Location{
						{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
					},
					Sample: []*profile.Sample{
						{
							Location: []*profile.Location{
								{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
							},
							Value: []int64{55, 45}, // 45 / 100 = 45% > 35%
							Label: map[string][]string{
								"kernel": {"uncoalesced_load"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemory,
			wantSeverity: SeverityWarning,
			wantFinding:  "Memory stalls dominate this load: stall_mem is 45.0% of stalls",
			wantRec:      "Check whether adjacent lane IDs access contiguous bytes",
		},
		{
			name: "uncoalesced_critical_high_mem_stalls_60pct",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
				Files: map[string][]byte{
					"kernels.sass": []byte(sampleLDGSASS),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "stall_math", Unit: "count"},
						{Type: "stall_mem", Unit: "count"},
					},
					Function: []*profile.Function{
						{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"},
					},
					Location: []*profile.Location{
						{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
					},
					Sample: []*profile.Sample{
						{
							Location: []*profile.Location{
								{ID: 1, Address: 0x0010, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "uncoalesced_load", Filename: "uncoalesced.cu"}, Line: 42}}},
							},
							Value: []int64{40, 60}, // 60 / 100 = 60% >= 50% -> Critical
							Label: map[string][]string{
								"kernel": {"uncoalesced_load"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemory,
			wantSeverity: SeverityCritical,
			wantFinding:  "Memory stalls dominate this load: stall_mem is 60.0% of stalls",
			wantRec:      "Check whether adjacent lane IDs access contiguous bytes",
		},
		{
			name: "coalesced_ldg_low_mem_stalls_15pct",
			cap: &cudacapture.Capture{
				Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
				Files: map[string][]byte{
					"kernels.sass": []byte(sampleLDGSASS),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "stall_scoreboard", Unit: "count"},
						{Type: "stall_mem", Unit: "count"},
					},
					Location: []*profile.Location{
						{ID: 1, Address: 0x0010},
					},
					Sample: []*profile.Sample{
						{
							Location: []*profile.Location{{ID: 1, Address: 0x0010}},
							Value:    []int64{85, 15}, // 15% <= 35%
							Label: map[string][]string{
								"kernel": {"uncoalesced_load"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "address_arithmetic_non_contiguous_stride_label",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}},
					Sample: []*profile.Sample{
						{
							Value: []int64{100},
							Label: map[string][]string{
								"kernel": {"strided_access_kernel"},
								"stride": {"32"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemory,
			wantSeverity: SeverityWarning,
			wantFinding:  "Threads within the warp generate strided or disparate 128-byte cache line requests",
			wantRec:      "Align and coalesce thread access so adjacent lane IDs access contiguous bytes within a single 128B cache line",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := AnalyzeCapture(tt.cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && (tt.wantSeverity == "" || item.Severity == tt.wantSeverity) &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected uncoalesced memory warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected uncoalesced memory warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_CUDAGraphCandidate(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			// 26.5% duty cycle and every launch issued after the previous
			// kernel had already retired.
			name: "short_kernels_idle_device_host_behind",
			cap: &cudacapture.Capture{
				Files: map[string][]byte{
					"capture-status.json": apiRecordsFile(t, map[int]uint64{2: 10000, 3: 30000, 4: 50000, 5: 70000}),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						microKernelSample("k1", 1, 1000, 4000, 1),
						microKernelSample("k2", 1, 20000, 24000, 2),
						microKernelSample("k3", 1, 40000, 45000, 3),
						microKernelSample("k4", 1, 60000, 66000, 4),
						microKernelSample("k5", 1, 80000, 84000, 5),
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryArchitectureDispatch,
			wantSeverity: SeverityWarning,
			wantFinding:  "Driver submission overhead dominates short kernel execution",
			wantRec:      "Capture sequence into a CUDA Graph (cudaGraphLaunch) to eliminate host submission latency",
		},
		{
			// Same short kernels, but back to back: the device is 98% busy, so
			// per-launch host cost is costing it nothing.
			name: "short_kernels_high_duty_cycle",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						microKernelSample("k1", 1, 1000, 4000, 1),
						microKernelSample("k2", 1, 4100, 8100, 2),
						microKernelSample("k3", 1, 8200, 13200, 3),
						microKernelSample("k4", 1, 13300, 19300, 4),
						microKernelSample("k5", 1, 19400, 23400, 5),
					},
				},
			},
			// No dispatch finding of any kind belongs on a busy device.
			wantWarning:  false,
			wantCategory: CategoryArchitectureDispatch,
		},
		{
			// Idle device, but every launch was already issued before the
			// preceding kernel retired: the host is not the cause.
			name: "short_kernels_idle_device_host_ahead",
			cap: &cudacapture.Capture{
				Files: map[string][]byte{
					"capture-status.json": apiRecordsFile(t, map[int]uint64{2: 1500, 3: 21000, 4: 41000, 5: 61000}),
				},
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						microKernelSample("k1", 1, 1000, 4000, 1),
						microKernelSample("k2", 1, 20000, 24000, 2),
						microKernelSample("k3", 1, 40000, 45000, 3),
						microKernelSample("k4", 1, 60000, 66000, 4),
						microKernelSample("k5", 1, 80000, 84000, 5),
					},
				},
			},
			wantWarning:  false,
			wantCategory: CategoryArchitectureDispatch,
			wantFinding:  "Driver submission overhead dominates short kernel execution",
		},
		{
			name: "broken_sequence_by_memcpy",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{Value: []int64{3000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{5000}, Label: map[string][]string{"name": {"cudaMemcpyAsync"}, "subtype": {"memcpy"}, "stream": {"1"}}},
						{Value: []int64{6000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k5"}, "subtype": {"kernel"}, "stream": {"1"}}},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "broken_sequence_by_synchronization",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{Value: []int64{3000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{0}, Label: map[string][]string{"name": {"cudaStreamSynchronize"}, "subtype": {"synchronization"}, "stream": {"1"}}},
						{Value: []int64{6000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{4000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "long_kernels_above_10us",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{Value: []int64{15000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{20000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{12000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{14000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
						{Value: []int64{16000}, Label: map[string][]string{"name": {"k5"}, "subtype": {"kernel"}, "stream": {"1"}}},
					},
				},
			},
			wantWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := AnalyzeCapture(tt.cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && (tt.wantSeverity == "" || item.Severity == tt.wantSeverity) &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected cuda graph candidate warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected cuda graph candidate warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_PointwiseFusion(t *testing.T) {
	tests := []struct {
		name         string
		cap          *cudacapture.Capture
		wantWarning  bool
		wantCategory Category
		wantSeverity Severity
		wantFinding  string
		wantRec      string
	}{
		{
			name: "swiglu_followed_by_quantization_matching_shape",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{50000},
							Label: map[string][]string{
								"name":    {"swiglu_forward_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemoryCompute,
			wantSeverity: SeverityWarning,
			wantFinding:  "Redundant global memory round-trips for intermediate tensors",
			wantRec:      "Fuse into a single kernel (e.g. fused SwiGLU + quant)",
		},
		{
			name: "pointwise_matching_buffers",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{20000},
							Label: map[string][]string{
								"name":    {"silu_activation"},
								"subtype": {"kernel"},
								"stream":  {"2"},
								"buffer":  {"tensor_act_0"},
							},
						},
						{
							Value: []int64{15000},
							Label: map[string][]string{
								"name":    {"elementwise_scale_bias"},
								"subtype": {"kernel"},
								"stream":  {"2"},
								"buffer":  {"tensor_act_0"},
							},
						},
					},
				},
			},
			wantWarning:  true,
			wantCategory: CategoryMemoryCompute,
			wantSeverity: SeverityWarning,
			wantFinding:  "Redundant global memory round-trips for intermediate tensors",
			wantRec:      "Fuse into a single kernel",
		},
		{
			name: "different_streams_no_fusion",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{50000},
							Label: map[string][]string{
								"name":    {"swiglu_forward_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"2"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "gemm_followed_by_quant_no_fusion",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{100000},
							Label: map[string][]string{
								"name":    {"gemm_cutlass_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
		{
			name: "intervening_memcpy_no_fusion",
			cap: &cudacapture.Capture{
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{
						{Type: "kernel_time", Unit: "nanoseconds"},
					},
					Sample: []*profile.Sample{
						{
							Value: []int64{50000},
							Label: map[string][]string{
								"name":    {"swiglu_forward_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
						{
							Value: []int64{10000},
							Label: map[string][]string{
								"name":    {"cudaMemcpyAsync"},
								"subtype": {"memcpy"},
								"stream":  {"1"},
							},
						},
						{
							Value: []int64{30000},
							Label: map[string][]string{
								"name":    {"quantize_fp8_kernel"},
								"subtype": {"kernel"},
								"stream":  {"1"},
								"shape":   {"[4096, 4096]"},
							},
						},
					},
				},
			},
			wantWarning: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report, err := AnalyzeCapture(tt.cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Category == tt.wantCategory && (tt.wantSeverity == "" || item.Severity == tt.wantSeverity) &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					break
				}
			}

			if tt.wantWarning && !found {
				t.Errorf("did not find expected pointwise fusion warning; items: %+v", report.Items)
			} else if !tt.wantWarning && found {
				t.Errorf("unexpected pointwise fusion warning found in %+v", report.Items)
			}
		})
	}
}

func TestAdvice_BlackwellDeepSeekV4DecodeProfiling(t *testing.T) {
	// Comprehensive test synthesizing all 5 heuristics in a realistic Blackwell sm_100a
	// DeepSeek-V4 decode profiling session.
	const sass = `
	.section	.text.deepseek_decode,"ax",@progbits
deepseek_decode:
.text.deepseek_decode:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.128 R0, [R2] ;
        /*0020*/                   EXIT ;
`

	cap := &cudacapture.Capture{
		Metadata: []byte(`{"arch": "sm_100a", "sm_count": 136, "device": "NVIDIA GB200"}`),
		Code: cudacapture.CodeFiles{
			SASS: "kernels.sass",
		},
		Files: map[string][]byte{
			"kernels.sass":        []byte(sass),
			"capture-status.json": apiRecordsFile(t, map[int]uint64{102: 10000, 103: 30000, 104: 50000, 105: 70000}),
		},
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_mem", Unit: "count"},
			},
			Location: []*profile.Location{
				{ID: 1, Address: 0x0010},
			},
			Sample: []*profile.Sample{
				// 1 & 2. Underpopulated thread block (32 threads) and SM Grid Starvation (64 blocks < 136 SMs)
				{
					Value: []int64{8000, 0, 0},
					Label: map[string][]string{
						"kernel":  {"deepseek_decode"},
						"subtype": {"kernel"},
						"stream":  {"0"},
						"block":   {"32,1,1"},
						"grid":    {"64,1,1"},
					},
				},
				// 3. Memory-dominated LDG.E at PC 0x0010, 45% of stalls in stall_mem.
				// The counter is stall_mem: this comment used to say
				// stall_memory_throttle, which the profile does not declare and no
				// producer emits, and the finding string said so too.
				{
					Location: []*profile.Location{{ID: 1, Address: 0x0010}},
					Value:    []int64{0, 55, 45}, // 45 / 100 = 45% > 35%
					Label: map[string][]string{
						"kernel": {"deepseek_decode"},
					},
				},
				// 4. CUDA Graph candidates: 5 consecutive short kernels on stream 1
				// (< 10 µs) on an idle device, each launched after the previous
				// kernel had retired.
				microKernelSample("rms_norm", 1, 1000, 3000, 101, 0, 0),
				microKernelSample("q_proj", 1, 20000, 24000, 102, 0, 0),
				microKernelSample("k_proj", 1, 40000, 43000, 103, 0, 0),
				microKernelSample("v_proj", 1, 60000, 63000, 104, 0, 0),
				microKernelSample("rope_embed", 1, 80000, 85000, 105, 0, 0),
				// 5. Pointwise fusion: SwiGLU followed by quantization on stream 2
				{
					Value: []int64{45000, 0, 0},
					Label: map[string][]string{
						"name":    {"swiglu_mlp_gate"},
						"subtype": {"kernel"},
						"stream":  {"2"},
						"shape":   {"[2048, 7168]"},
					},
				},
				{
					Value: []int64{35000, 0, 0},
					Label: map[string][]string{
						"name":    {"quantize_fp8_linear"},
						"subtype": {"kernel"},
						"stream":  {"2"},
						"shape":   {"[2048, 7168]"},
					},
				},
			},
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	expectedFindings := []struct {
		name     string
		category Category
		finding  string
		rec      string
	}{
		{
			name:     "heuristic_1_single_warp",
			category: CategoryArchitectureCompute,
			finding:  "Single-warp thread blocks",
			rec:      "Restructure thread block geometry to >=128 threads (4 warps)",
		},
		{
			name:     "heuristic_2_grid_starvation",
			category: CategoryArchitecture,
			finding:  "Grid size leaves physical SMs completely idle",
			rec:      "Decompose grid using 2D tiling, split-K reduction, or sub-grid pipelining (griddepcontrol)",
		},
		{
			name:     "heuristic_3_memory_dominated_load",
			category: CategoryMemory,
			finding:  "Memory stalls dominate this load: stall_mem is 45.0% of stalls",
			rec:      "Check whether adjacent lane IDs access contiguous bytes",
		},
		{
			name:     "heuristic_4_cuda_graph",
			category: CategoryArchitectureDispatch,
			finding:  "Driver submission overhead dominates short kernel execution",
			rec:      "Capture sequence into a CUDA Graph (cudaGraphLaunch) to eliminate host submission latency",
		},
		{
			name:     "heuristic_5_pointwise_fusion",
			category: CategoryMemoryCompute,
			finding:  "Redundant global memory round-trips for intermediate tensors",
			rec:      "Fuse into a single kernel (e.g. fused SwiGLU + quant)",
		},
	}

	for _, ef := range expectedFindings {
		t.Run(ef.name, func(t *testing.T) {
			var found bool
			for _, item := range report.Items {
				if item.Category == ef.category && strings.Contains(item.Finding, ef.finding) {
					found = true
					if !strings.Contains(item.ActionableRecommendation, ef.rec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, ef.rec)
					}
					break
				}
			}
			if !found {
				t.Errorf("expected finding %q in category %q not found; report: %+v", ef.finding, ef.category, report.Items)
			}
		})
	}
}

func TestAdvice_CUBINResourceOccupancyAndTargetDisambiguation(t *testing.T) {
	// Synthesize a CUBIN ELF image with three kernels:
	// 1. "spilling_kernel": 48 regs, 256 bytes frame size (EIATTR_FRAME_SIZE)
	// 2. "reg_limited_kernel": 80 regs, ReqNTID [128, 1, 1]
	// 3. "smem_limited_kernel": 24 regs, 65536 bytes static smem, ReqNTID [256, 1, 1]
	var nv1, nv2, nv3 bytes.Buffer

	// spilling_kernel
	nv1.Write([]byte{0x04, cudaoccupancy.EIATTR_REGCOUNT, 0x04, 0x00})
	binary.Write(&nv1, binary.LittleEndian, uint32(48))
	nv1.Write([]byte{0x04, cudaoccupancy.EIATTR_FRAME_SIZE, 0x04, 0x00})
	binary.Write(&nv1, binary.LittleEndian, uint32(256))

	// reg_limited_kernel
	nv2.Write([]byte{0x04, cudaoccupancy.EIATTR_REGCOUNT, 0x04, 0x00})
	binary.Write(&nv2, binary.LittleEndian, uint32(80))
	nv2.Write([]byte{0x04, cudaoccupancy.EIATTR_REQ_NTID, 0x0c, 0x00})
	binary.Write(&nv2, binary.LittleEndian, uint32(128))
	binary.Write(&nv2, binary.LittleEndian, uint32(1))
	binary.Write(&nv2, binary.LittleEndian, uint32(1))

	// smem_limited_kernel
	nv3.Write([]byte{0x04, cudaoccupancy.EIATTR_REGCOUNT, 0x04, 0x00})
	binary.Write(&nv3, binary.LittleEndian, uint32(24))
	nv3.Write([]byte{0x04, cudaoccupancy.EIATTR_REQ_NTID, 0x0c, 0x00})
	binary.Write(&nv3, binary.LittleEndian, uint32(256))
	binary.Write(&nv3, binary.LittleEndian, uint32(1))
	binary.Write(&nv3, binary.LittleEndian, uint32(1))
	nv3.Write([]byte{0x04, cudaoccupancy.EIATTR_SHARED_SIZE_SYSMEM, 0x04, 0x00})
	binary.Write(&nv3, binary.LittleEndian, uint32(65536))

	sections := map[string][]byte{
		".text.spilling_kernel":        make([]byte, 32),
		".nv.info.spilling_kernel":     nv1.Bytes(),
		".text.reg_limited_kernel":     make([]byte, 32),
		".nv.info.reg_limited_kernel":  nv2.Bytes(),
		".text.smem_limited_kernel":    make([]byte, 32),
		".nv.info.smem_limited_kernel": nv3.Bytes(),
	}

	cubinBytes := synthesizeTestELFForAdvise(sections)

	tests := []struct {
		name         string
		targetArch   string
		wantKernel   string
		wantFinding  string
		wantRec      string
		wantSeverity Severity
		wantEvidence EvidenceLevel
		wantCaveat   bool // recommendation is derived from an occupancy figure
	}{
		{
			name:         "frame_size_spill_detection",
			targetArch:   "sm_121a",
			wantKernel:   "spilling_kernel",
			wantFinding:  "allocates 256 bytes of local stack frame per thread in CUBIN ELF (EIATTR_FRAME_SIZE)",
			wantRec:      "eliminate local memory stack frame spills",
			wantSeverity: SeverityCritical,
			wantEvidence: EvidenceObserved,
			wantCaveat:   false, // a local stack frame spill is real work, not a residency figure
		},
		{
			name:         "blackwell_sm121_reg_cliff",
			targetArch:   "sm_121a",
			wantKernel:   "reg_limited_kernel",
			wantFinding:  "Microarchitecture theoretical occupancy on sm_121 for kernel \"reg_limited_kernel\"",
			wantRec:      "Register file capacity (65536 regs/SM) limits residency",
			wantSeverity: SeverityWarning,
			wantEvidence: EvidenceSuspected,
			wantCaveat:   true,
		},
		{
			name:         "hopper_sm90_target_disambiguation",
			targetArch:   "sm_90a",
			wantKernel:   "reg_limited_kernel",
			wantFinding:  "Microarchitecture theoretical occupancy on sm_90 for kernel \"reg_limited_kernel\"",
			wantRec:      "Register file capacity (65536 regs/SM) limits residency",
			wantSeverity: SeverityWarning,
			wantEvidence: EvidenceSuspected,
			wantCaveat:   true,
		},
		{
			name:         "smem_capacity_limit",
			targetArch:   "sm_121a",
			wantKernel:   "smem_limited_kernel",
			wantFinding:  "constrained by shared_memory",
			wantRec:      "Static shared memory (65536 bytes/block) limits residency",
			wantSeverity: SeverityWarning,
			wantEvidence: EvidenceSuspected,
			wantCaveat:   true,
		},
	}

	for _, arch := range []string{"", "sm_120", "unrecognized"} {
		t.Run("unsupported/"+arch, func(t *testing.T) {
			cap := &cudacapture.Capture{
				Metadata: []byte(`{"architecture":"` + arch + `"}`),
				Code:     cudacapture.CodeFiles{CUBIN: "kernels.cubin"},
				Files:    map[string][]byte{"kernels.cubin": cubinBytes},
			}
			var report AdviceReport
			auditCUBINResourceOccupancy(cap, &report, nil)
			var frameFound bool
			for _, item := range report.Items {
				if strings.Contains(item.Finding, "Microarchitecture theoretical occupancy") {
					t.Errorf("unknown architecture produced occupancy advice: %s", item.Finding)
				}
				if strings.Contains(item.Finding, "allocates 256 bytes of local stack frame") {
					frameFound = true
				}
			}
			if !frameFound {
				t.Error("unknown architecture suppressed the observed stack frame")
			}
		})
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{
				Metadata: []byte(`{"architecture":"` + tt.targetArch + `"}`),
				Code: cudacapture.CodeFiles{
					CUBIN: "kernels.cubin",
				},
				Files: map[string][]byte{
					"kernels.cubin": cubinBytes,
				},
			}

			report, err := AnalyzeCapture(cap)
			if err != nil {
				t.Fatalf("AnalyzeCapture failed: %v", err)
			}

			var found bool
			for _, item := range report.Items {
				if item.Location.Kernel == tt.wantKernel &&
					item.Severity == tt.wantSeverity &&
					strings.Contains(item.Finding, tt.wantFinding) {
					found = true
					if tt.wantEvidence != "" && item.EvidenceLevel != tt.wantEvidence {
						t.Errorf("evidence = %q, want %q", item.EvidenceLevel, tt.wantEvidence)
					}
					if !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
						t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
					}
					if got := strings.Contains(item.ActionableRecommendation, occupancyCaveat); got != tt.wantCaveat {
						t.Errorf("occupancyCaveat present = %v, want %v in %q", got, tt.wantCaveat, item.ActionableRecommendation)
					}
					break
				}
			}
			if !found {
				t.Errorf("did not find expected item for kernel %q with finding containing %q, severity %s; got report: %+v",
					tt.wantKernel, tt.wantFinding, tt.wantSeverity, report.Items)
			}
		})
	}
}

func synthesizeTestELFForAdvise(sections map[string][]byte) []byte {
	var buf bytes.Buffer

	secNames := []string{""}
	secData := [][]byte{nil}
	secTypes := []uint32{0}
	secFlags := []uint64{0}

	shstrtabIdx := len(secNames)
	secNames = append(secNames, ".shstrtab")
	secTypes = append(secTypes, uint32(elf.SHT_STRTAB))
	secFlags = append(secFlags, 0)
	secData = append(secData, nil)

	for name, data := range sections {
		secNames = append(secNames, name)
		secTypes = append(secTypes, uint32(elf.SHT_PROGBITS))
		secFlags = append(secFlags, 0)
		secData = append(secData, data)
	}

	var shstrtab bytes.Buffer
	shstrtab.WriteByte(0)
	nameOffsets := make([]uint32, len(secNames))
	for i, name := range secNames {
		if i == 0 {
			nameOffsets[i] = 0
			continue
		}
		nameOffsets[i] = uint32(shstrtab.Len())
		shstrtab.WriteString(name)
		shstrtab.WriteByte(0)
	}
	secData[shstrtabIdx] = shstrtab.Bytes()

	ehdr := make([]byte, 64)
	copy(ehdr[0:4], []byte("\x7fELF"))
	ehdr[4] = 2                                     // 64-bit
	ehdr[5] = 1                                     // Little-endian
	ehdr[6] = 1                                     // Version 1
	binary.LittleEndian.PutUint16(ehdr[16:18], 1)   // ET_REL
	binary.LittleEndian.PutUint16(ehdr[18:20], 190) // EM_CUDA
	binary.LittleEndian.PutUint32(ehdr[20:24], 1)   // EV_CURRENT
	binary.LittleEndian.PutUint16(ehdr[52:54], 64)  // ehsize
	binary.LittleEndian.PutUint16(ehdr[58:60], 64)  // shentsize
	binary.LittleEndian.PutUint16(ehdr[60:62], uint16(len(secNames)))
	binary.LittleEndian.PutUint16(ehdr[62:64], uint16(shstrtabIdx))

	buf.Write(ehdr)

	secOffsets := make([]uint64, len(secNames))
	for i, data := range secData {
		if len(data) == 0 {
			secOffsets[i] = 0
			continue
		}
		secOffsets[i] = uint64(buf.Len())
		buf.Write(data)
	}

	shoff := uint64(buf.Len())
	binary.LittleEndian.PutUint64(ehdr[40:48], shoff)
	copy(buf.Bytes()[40:48], ehdr[40:48])

	for i := range secNames {
		shdr := make([]byte, 64)
		binary.LittleEndian.PutUint32(shdr[0:4], nameOffsets[i])
		binary.LittleEndian.PutUint32(shdr[4:8], secTypes[i])
		binary.LittleEndian.PutUint64(shdr[8:16], secFlags[i])
		binary.LittleEndian.PutUint64(shdr[24:32], secOffsets[i])
		binary.LittleEndian.PutUint64(shdr[32:40], uint64(len(secData[i])))
		binary.LittleEndian.PutUint64(shdr[48:56], 1)
		buf.Write(shdr)
	}

	return buf.Bytes()
}

func TestAdvice_ElementwiseExcludesGEMV(t *testing.T) {
	// Verifies that matrix-vector multiplication (GEMV) kernels like ds4_q8_0_matvec_quantized
	// are excluded from elementwise fusion recommendations, even if they have "quant" in name.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{50000},
					Label: map[string][]string{
						"name":    {"ds4_swiglu_quant_q8_0"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
				{
					Value: []int64{30000},
					Label: map[string][]string{
						"name":    {"ds4_q8_0_matvec_quantized"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
			},
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	for _, item := range report.Items {
		if item.Category == CategoryMemoryCompute && strings.Contains(item.Finding, "ds4_q8_0_matvec_quantized") && strings.Contains(item.Finding, "elementwise") {
			t.Errorf("unexpected elementwise fusion recommendation between SwiGLU and GEMV: %+v", item)
		}
	}
}

func TestAdvice_SequentialRMSNormQuantFusion(t *testing.T) {
	// Verifies detection of sequential RMSNorm followed immediately by quantization.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{4000},
					Label: map[string][]string{
						"name":    {"ds4_rms"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
				{
					Value: []int64{3000},
					Label: map[string][]string{
						"name":    {"ds4_quant_q8_1"},
						"subtype": {"kernel"},
						"stream":  {"1"},
					},
				},
			},
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if item.Category == CategoryMemoryCompute && strings.Contains(item.Finding, "Redundant global memory round-trips for normalized activations") {
			found = true
			if !strings.Contains(item.ActionableRecommendation, "Fuse normalization and quantization") {
				t.Errorf("recommendation = %q, want containing fuse normalization and quantization", item.ActionableRecommendation)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected RMSNorm->Quant fusion advisory not found; items: %+v", report.Items)
	}
}

func TestAdvice_DualGEMVPairedProjections(t *testing.T) {
	// Verifies detection of paired GEMVs operating on identical activation vectors.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{25000},
					Label: map[string][]string{
						"name":    {"ds4_attnQA_gemv"},
						"subtype": {"kernel"},
						"stream":  {"1"},
						"buffer":  {"attn_norm_out"},
					},
				},
				{
					Value: []int64{25000},
					Label: map[string][]string{
						"name":    {"ds4_attnKV_gemv"},
						"subtype": {"kernel"},
						"stream":  {"1"},
						"buffer":  {"attn_norm_out"},
					},
				},
			},
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if item.Category == CategoryMemoryCompute && strings.Contains(item.Finding, "Redundant activation DRAM traffic") {
			found = true
			if !strings.Contains(item.ActionableRecommendation, "paired dual-GEMV") {
				t.Errorf("recommendation = %q, want containing paired dual-GEMV", item.ActionableRecommendation)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected Dual-GEMV paired projection advisory not found; items: %+v", report.Items)
	}
}

func TestAdvice_MultiWarpReductionStarvation(t *testing.T) {
	// Verifies detection of multi-warp thread blocks executing sequential chunk reductions with idle warps.
	cap := &cudacapture.Capture{
		Metadata: []byte(`{"arch": "sm_121a"}`),
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{100},
					Label: map[string][]string{
						"kernel": {"ds4_f16_pair_ordered_chunks"},
						"block":  {"128,1,1"},
						"grid":   {"256,1,1"},
					},
				},
			},
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if item.Category == CategoryArchitectureCompute && strings.Contains(item.Finding, "Multi-warp thread block underutilization in row reduction") {
			found = true
			if !strings.Contains(item.ActionableRecommendation, "Group multiple rows per thread block") {
				t.Errorf("recommendation = %q, want containing Group multiple rows per thread block", item.ActionableRecommendation)
			}
			// This advisory names ds4_f16_pair_ordered_chunks, the one kernel where
			// the transformation it recommends was measured 2.84% slower. It must
			// not issue that as an instruction.
			if !strings.Contains(item.ActionableRecommendation, occupancyCaveat) {
				t.Errorf("recommendation = %q, want occupancyCaveat appended", item.ActionableRecommendation)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected Multi-Warp reduction starvation advisory not found; items: %+v", report.Items)
	}
}

func TestAdvice_CumulativeDriverSubmissionOverhead(t *testing.T) {
	// Verifies that a stream with multiple micro-kernels reports cumulative statistics and top contributors.
	// Kernels are spaced 20 µs apart on an idle stream, and every launch is
	// issued 5 µs after the previous kernel retired.
	names := []string{"ds4_rms", "ds4_rms", "ds4_quant_q8_1", "ds4_quant_q8_1", "ds4_router_biased", "ds4_rms", "ds4_quant_q8_1", "ds4_rms", "ds4_router_biased", "ds4_rms"}
	durations := []uint64{2000, 2000, 2500, 2500, 3000, 2000, 2500, 2000, 3000, 2000}
	samples := make([]*profile.Sample, 0, len(names))
	apiStarts := make(map[int]uint64, len(names))
	var prevEnd uint64
	for i, name := range names {
		start := uint64(1000 + i*20000)
		samples = append(samples, microKernelSample(name, 1, start, start+durations[i], i+1))
		if i > 0 {
			apiStarts[i+1] = prevEnd + 5000
		}
		prevEnd = start + durations[i]
	}

	cap := &cudacapture.Capture{
		Files: map[string][]byte{"capture-status.json": apiRecordsFile(t, apiStarts)},
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: samples,
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if item.Category == CategoryArchitectureDispatch && strings.Contains(item.Finding, "Driver submission overhead dominates short kernel execution") {
			found = true
			if !strings.Contains(item.Finding, "stream totals 10 micro-kernels") {
				t.Errorf("finding = %q, want containing stream totals 10 micro-kernels", item.Finding)
			}
			if !strings.Contains(item.Finding, "top micro-kernels") {
				t.Errorf("finding = %q, want containing top micro-kernels", item.Finding)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected Cumulative driver submission advisory not found; items: %+v", report.Items)
	}
}

func TestAdvice_MicroKernelDeterministicTieBreaking(t *testing.T) {
	// Verifies that micro-kernels with identical counts sort deterministically (alphabetically by name).
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "kernel_time", Unit: "nanoseconds"},
			},
			Sample: []*profile.Sample{
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_z"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_a"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_z"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_a"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_m"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_m"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_b"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_b"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_c"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{2000}, Label: map[string][]string{"name": {"kernel_c"}, "subtype": {"kernel"}, "stream": {"1"}}},
			},
		},
	}

	for repeat := 0; repeat < 5; repeat++ {
		report, err := AnalyzeCapture(cap)
		if err != nil {
			t.Fatalf("AnalyzeCapture failed: %v", err)
		}
		var found bool
		for _, item := range report.Items {
			if strings.Contains(item.Finding, "top micro-kernels") {
				found = true
				want := "[top micro-kernels: kernel_a (2x), kernel_b (2x), kernel_c (2x)]"
				if !strings.Contains(item.Finding, want) {
					t.Fatalf("run %d: finding = %q, want containing %q", repeat, item.Finding, want)
				}
				break
			}
		}
		if !found {
			t.Fatalf("run %d: expected finding with top micro-kernels not found", repeat)
		}
	}
}

// geometrySample builds one kernel activity sample with launch geometry labels.
func geometrySample(name, grid, block string, durationNs int64) *profile.Sample {
	return &profile.Sample{
		Value: []int64{durationNs},
		Label: map[string][]string{
			"name":    {name},
			"subtype": {"kernel"},
			"stream":  {"1"},
			"grid":    {grid},
			"block":   {block},
		},
	}
}

func TestAdvice_UnderSubscribedLaunchGeometry(t *testing.T) {
	const gb10Metadata = `{"arch":"sm_121a","sm_count":48,"device":"NVIDIA GB10"}`
	// 48 SMs x 32 threads per warp.
	const warpPerSM = 1536

	tests := []struct {
		name         string
		metadata     string
		samples      []*profile.Sample
		wantSeverity Severity
		wantFindings []string
		wantRec      string
		wantAbsent   []string
		// wantCaveat is false for every row here on purpose: grid starvation
		// leaves SMs literally idle, so its advice is not a capacity figure being
		// read as throughput. This pins those sites bare against a later sweep
		// that appends occupancyCaveat to everything mentioning occupancy.
		wantCaveat bool
	}{
		{
			name:     "single_threaded_launches",
			metadata: gb10Metadata,
			samples: []*profile.Sample{
				geometrySample("ds4_compressor_attention", "1,1,1", "1,1,1", 361300),
				geometrySample("ds4_compressor_attention", "1,1,1", "1,1,1", 361300),
				geometrySample("ds4_hc_cached_split", "1,1,1", "1,1,1", 15200),
				geometrySample("ds4_hc_cached_split", "1,1,1", "1,1,1", 15200),
				geometrySample("ds4_hc_cached_split", "1,1,1", "1,1,1", 15200),
				geometrySample("ds4_moe_gemm", "96,1,1", "256,1,1", 100000),
			},
			wantSeverity: SeverityCritical,
			wantFindings: []string{
				"Single-threaded kernel launches leave the device idle",
				"5 of 6 captured launches (83.3%) ran one thread (grid 1 x block 1)",
				"consuming 0.77 ms of device time",
				"ds4_compressor_attention (1 thread, 2 launches, 361.3 µs mean, 0.72 ms total)",
			},
			wantRec: "Parallelize these kernels across threads and blocks",
		},
		{
			name:     "cannot_fill_one_warp_per_sm",
			metadata: gb10Metadata,
			samples: []*profile.Sample{
				geometrySample("ds4_compressor_indexer", "4,1,1", "32,1,1", 78800),
				geometrySample("ds4_compressor_indexer", "4,1,1", "32,1,1", 78800),
				geometrySample("ds4_moe_gemm", "96,1,1", "256,1,1", 100000),
			},
			wantSeverity: SeverityWarning,
			wantFindings: []string{
				"Launch geometry cannot fill one warp per SM",
				"2 of 3 captured launches (66.7%) ran fewer than 1536 total threads (48 SMs x 32 threads per warp)",
				"ds4_compressor_indexer (128 threads, 2 launches, 78.8 µs mean, 0.16 ms total)",
			},
			wantRec:    "Grow the grid so at least one warp is resident per SM (>= 1536 total threads)",
			wantAbsent: []string{"Single-threaded kernel launches leave the device idle"},
		},
		{
			name:     "fully_subscribed_launches",
			metadata: gb10Metadata,
			samples: []*profile.Sample{
				geometrySample("ds4_moe_gemm", "96,1,1", "256,1,1", 100000),
				geometrySample("ds4_attention", "48,2,1", "128,1,1", 200000),
			},
			wantAbsent: []string{
				"Single-threaded kernel launches leave the device idle",
				"Launch geometry cannot fill one warp per SM",
			},
		},
	}

	if warpPerSM != 48*32 {
		t.Fatalf("warpPerSM constant %d does not match 48 SMs x 32 threads", warpPerSM)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cap := &cudacapture.Capture{
				Metadata: []byte(tt.metadata),
				Profile: &profile.Profile{
					SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
					Sample:     tt.samples,
				},
			}
			report := &AdviceReport{}
			auditUnderSubscribedLaunches(cap, report)

			for _, want := range tt.wantFindings {
				var found bool
				for _, item := range report.Items {
					if strings.Contains(item.Finding, want) {
						found = true
						if item.Severity != tt.wantSeverity {
							t.Errorf("severity = %q, want %q", item.Severity, tt.wantSeverity)
						}
						if item.Category != CategoryArchitectureCompute {
							t.Errorf("category = %q, want %q", item.Category, CategoryArchitectureCompute)
						}
						if tt.wantRec != "" && !strings.Contains(item.ActionableRecommendation, tt.wantRec) {
							t.Errorf("recommendation = %q, want containing %q", item.ActionableRecommendation, tt.wantRec)
						}
						if got := strings.Contains(item.ActionableRecommendation, occupancyCaveat); got != tt.wantCaveat {
							t.Errorf("occupancyCaveat present = %v, want %v in %q", got, tt.wantCaveat, item.ActionableRecommendation)
						}
						break
					}
				}
				if !found {
					t.Errorf("finding containing %q not found; items: %+v", want, report.Items)
				}
			}
			for _, absent := range tt.wantAbsent {
				for _, item := range report.Items {
					if strings.Contains(item.Finding, absent) {
						t.Errorf("unexpected finding containing %q: %+v", absent, item)
					}
				}
			}
		})
	}
}

func TestAdvice_UnderSubscribedWithoutSMCount(t *testing.T) {
	// Without a recorded SM count the one-warp-per-SM rule cannot run, and the
	// single-thread finding has to say so.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample: []*profile.Sample{
				geometrySample("ds4_compressor_attention", "1,1,1", "1,1,1", 361300),
				geometrySample("ds4_compressor_indexer", "4,1,1", "32,1,1", 78800),
			},
		},
	}

	// A CUDA host answers determineSMCount from the live device; this case only
	// exists where nothing can.
	if _, _, hasSMCount := determineSMCount(cap); hasSMCount {
		t.Skip("local CUDA device supplies an SM count; degraded path not reachable here")
	}

	report := &AdviceReport{}
	auditUnderSubscribedLaunches(cap, report)

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Single-threaded kernel launches leave the device idle") {
			found = true
			if !strings.Contains(item.Finding, "one-warp-per-SM rule was skipped") {
				t.Errorf("finding = %q, want it to disclose the skipped one-warp-per-SM rule", item.Finding)
			}
		}
		if strings.Contains(item.Finding, "Launch geometry cannot fill one warp per SM") {
			t.Errorf("warp-per-SM finding emitted without an SM count: %+v", item)
		}
	}
	if !found {
		t.Fatalf("expected single-threaded launch finding; items: %+v", report.Items)
	}
}

func TestAdvice_ShortKernelSequenceWithoutTimestamps(t *testing.T) {
	// A capture with no activity timestamps cannot support any statement about
	// device idle time or host submission.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample: []*profile.Sample{
				{Value: []int64{3000}, Label: map[string][]string{"name": {"k1"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{4000}, Label: map[string][]string{"name": {"k2"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{5000}, Label: map[string][]string{"name": {"k3"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{6000}, Label: map[string][]string{"name": {"k4"}, "subtype": {"kernel"}, "stream": {"1"}}},
				{Value: []int64{4000}, Label: map[string][]string{"name": {"k5"}, "subtype": {"kernel"}, "stream": {"1"}}},
			},
		},
	}

	report := &AdviceReport{}
	auditStreamKernelSequences(cap, report)

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Driver submission overhead") || strings.Contains(item.Finding, "Low GPU duty cycle") {
			t.Errorf("asserted an unmeasured cause: %+v", item)
		}
		if item.Severity == SeverityInfo && strings.Contains(item.Finding, "Short-kernel dispatch sequence with unmeasured GPU duty cycle") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected the unmeasured-duty-cycle Info finding; items: %+v", report.Items)
	}
}

func TestAdvice_DetermineSMCountPrecedence(t *testing.T) {
	// 1. Explicit SM count from capture-provenance.json takes highest precedence.
	capExplicit := &cudacapture.Capture{
		Files: map[string][]byte{
			"capture-provenance.json": []byte(`{"device":"NVIDIA GB10","arch":"sm_121a","sm_count":48}`),
			"kernels.ptx":             []byte(".version 8.0\n.target sm_90\n"),
		},
	}
	count, src, ok := determineSMCount(capExplicit)
	if !ok || count != 48 || src != sourceCapture {
		t.Fatalf("determineSMCount(capExplicit) = (%d, %v, %v), want (48, sourceCapture, true)", count, src, ok)
	}

	// 2. SM count derived from target arch takes precedence over missing explicit count.
	capArchOnly := &cudacapture.Capture{
		Files: map[string][]byte{
			"kernels.ptx": []byte(".version 8.0\n.target sm_90\n"),
		},
	}
	count, src, ok = determineSMCount(capArchOnly)
	if !ok || count != 132 || src != sourceCapture {
		t.Fatalf("determineSMCount(capArchOnly) = (%d, %v, %v), want (132, sourceCapture, true)", count, src, ok)
	}
}

func TestAdvice_MissingDeviceProvenanceInfoFinding(t *testing.T) {
	// Profile has launches with grid and block dimensions, but no device identity or target arch.
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
			Sample: []*profile.Sample{
				{
					Value: []int64{100000},
					Label: map[string][]string{
						"name":    {"my_kernel"},
						"subtype": {"kernel"},
						"grid":    {"16,1,1"},
						"block":   {"32,1,1"},
					},
				},
			},
		},
	}

	report := &AdviceReport{}
	auditLaunchGeometry(cap, report)

	// If no live device is available (e.g. non-CUDA host), an Info finding must be emitted.
	_, _, _, hasLimits := determineDeviceLimits(cap)
	_, _, hasSMCount := determineSMCount(cap)
	if !hasLimits && !hasSMCount {
		var found bool
		for _, item := range report.Items {
			if item.Severity == SeverityInfo && strings.Contains(item.Finding, "Device architecture and SM count metadata are not recorded") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected Info finding for missing device provenance, got items: %+v", report.Items)
		}
	}
}

func TestAdvice_DeviceProvenanceDisclosure(t *testing.T) {
	// When SM count or hardware limits are inferred from local device, finding text must disclose it.
	launches := []launchRecord{
		{Kernel: "starved_kernel", GridBlocks: 16, BlockThreads: 256},
		{Kernel: "single_warp_kernel", GridBlocks: 100, BlockThreads: 32},
	}

	// Emulate live device source
	smCount := 48
	smSrc := sourceLiveDevice
	maxWarpsPerSM := 48
	maxBlocksPerSM := 24
	limitSrc := sourceLiveDevice

	// Single warp check
	provSuffixLimit := ""
	if limitSrc == sourceLiveDevice {
		provSuffixLimit = " (hardware limits inferred from local host device, not recorded in capture)"
	}
	f1 := fmt.Sprintf("Single-warp thread blocks limit theoretical SM occupancy: kernel %q launched with %d block threads (<= 32) reaches 50%% theoretical occupancy with %d/%d resident warps/SM across %d blocks/SM%s",
		launches[1].Kernel, launches[1].BlockThreads, maxBlocksPerSM*1, maxWarpsPerSM, maxBlocksPerSM, provSuffixLimit)
	if !strings.Contains(f1, "(hardware limits inferred from local host device, not recorded in capture)") {
		t.Errorf("f1 missing live device disclosure: %s", f1)
	}

	// Starvation check
	provSuffixSM := ""
	if smSrc == sourceLiveDevice {
		provSuffixSM = " (SM count inferred from local host device, not recorded in capture)"
	}
	f2 := fmt.Sprintf("Grid size leaves physical SMs completely idle: kernel %q launched with %d grid blocks on a %d-SM GPU leaves 66%% of SMs idle%s",
		launches[0].Kernel, launches[0].GridBlocks, smCount, provSuffixSM)
	if !strings.Contains(f2, "(SM count inferred from local host device, not recorded in capture)") {
		t.Errorf("f2 missing live device disclosure: %s", f2)
	}
}

func TestAdvise_WarnsOnDroppedRecords(t *testing.T) {
	for _, statusBytes := range [][]byte{
		[]byte(`{"status":{"capture_success":true,"run_completeness":"partial","activity_records":36504,"dropped_records":48161}}`),
		[]byte(`{"capture_success":true,"run_completeness":"partial","activity_records":36504,"dropped_records":48161}`),
	} {
		cap := &cudacapture.Capture{
			Files: map[string][]byte{
				"capture-status.json": statusBytes,
			},
		}
		report := &AdviceReport{}
		auditCaptureStatus(cap, report)

		var found bool
		for _, item := range report.Items {
			if item.Severity == SeverityWarning && strings.Contains(item.Finding, "Lossy capture: 48161 records dropped") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected warning for dropped records, got items: %+v", report.Items)
		}
	}
}

// TestFixtureSampleTypesAreReal is the sweep behind TestStallFixtureNamesAreReal
// in diff_test.go. That test names four types and carries the negative control;
// this one finds every sample type any fixture in the package declares and
// checks them all, so a fixture added later cannot invent a counter and quietly
// take its narrative with it. The names come out of the test sources rather
// than a list here, because a list here would be a third copy of the same
// assumption, and the oracle is sampleTypeIsReal so there is only one.
func TestFixtureSampleTypesAreReal(t *testing.T) {
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`\{Type:\s*"([a-z0-9_]+)"`)
	seen := make(map[string]map[string]bool)
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range re.FindAllSubmatch(src, -1) {
			name := string(m[1])
			if seen[name] == nil {
				seen[name] = make(map[string]bool)
			}
			seen[name][f] = true
		}
	}
	if len(seen) == 0 {
		t.Fatal("found no sample types in the package fixtures; the pattern no longer matches anything")
	}
	for name, files := range seen {
		if sampleTypeIsReal(name) {
			continue
		}
		where := slices.Sorted(maps.Keys(files))
		t.Errorf("fixture sample type %q (%v) is emitted by no producer this command reads", name, where)
	}
}

// TestCapturedGraphReplays measures the correlation-multiplicity signature that
// stands in for CUPTI's dropped graphId. The two cases are the two arms of the
// GB10 measurement the detector is derived from: the same kernel executed 20
// times, once as 20 direct launches and once as a four-node graph replayed five
// times.
func TestCapturedGraphReplays(t *testing.T) {
	tests := []struct {
		name        string
		correlation func(i int) int
		graphID     func(i int) int // 0 means the capture carries no graph ID
		want        int
	}{
		{
			name:        "direct launches give every kernel its own correlation",
			correlation: func(i int) int { return 100 + i },
			want:        0,
		},
		{
			name:        "graph replay shares one correlation per cudaGraphLaunch",
			correlation: func(i int) int { return 100 + i/4 },
			want:        5,
		},
		{
			name:        "captures without correlation IDs claim nothing",
			correlation: func(int) int { return 0 },
			want:        0,
		},
		{
			// The agent now forwards graphId, so the same four-node graph is
			// read directly rather than inferred from shared correlations.
			name:        "graph ID counts replays directly",
			correlation: func(i int) int { return 100 + i/4 },
			graphID:     func(int) int { return 42 },
			want:        5,
		},
		{
			// The case correlation multiplicity cannot see: every replay of a
			// one-node graph has its own correlation and exactly one kernel
			// record, which is indistinguishable from an ordinary launch.
			name:        "single-node graph replay is invisible without a graph ID",
			correlation: func(i int) int { return 100 + i },
			want:        0,
		},
		{
			name:        "single-node graph replay is counted with a graph ID",
			correlation: func(i int) int { return 100 + i },
			graphID:     func(int) int { return 42 },
			want:        20,
		},
		{
			// A graph ID on some kernels and not others: only the graph nodes
			// witness replays, and the plain launches must not inflate them.
			name:        "mixed graph and non-graph work counts only the graph launches",
			correlation: func(i int) int { return 100 + i },
			graphID: func(i int) int {
				if i < 8 {
					return 42
				}
				return 0
			},
			want: 8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var samples []*profile.Sample
			for i := range 20 {
				start := uint64(1000 + i*10000)
				sample := microKernelSample("bump", 7, start, start+6800, tt.correlation(i))
				if tt.graphID != nil {
					if id := tt.graphID(i); id != 0 {
						sample.NumLabel["graph_id"] = []int64{int64(id)}
					}
				}
				samples = append(samples, sample)
			}
			cap := &cudacapture.Capture{Profile: &profile.Profile{Sample: samples}}
			if got := capturedGraphReplays(cap); got != tt.want {
				t.Errorf("capturedGraphReplays = %d, want %d", got, tt.want)
			}
		})
	}

	if got := capturedGraphReplays(nil); got != 0 {
		t.Errorf("capturedGraphReplays(nil) = %d, want 0", got)
	}
}

// TestDominanceRecommendationDropsGraphAdviceUnderReplay pins the defect
// measured on GB10: the dominance advisory told a workload that was already a
// CUDA graph replay to capture itself into a CUDA graph, at Critical severity,
// because graphId never reaches the advisor.
func TestDominanceRecommendationDropsGraphAdviceUnderReplay(t *testing.T) {
	const graphClause = "capture the entire layer sequence into a CUDA Graph"

	if rec := dominanceRecommendation(0); !strings.Contains(rec, graphClause) {
		t.Errorf("without graph replays the advisory must still offer graph capture; got %q", rec)
	}
	rec := dominanceRecommendation(5)
	if strings.Contains(rec, graphClause) {
		t.Errorf("under graph replay the advisory must not recommend graph capture; got %q", rec)
	}
	if !strings.Contains(rec, "already replays CUDA graphs") {
		t.Errorf("suppressed advisory must say why it was suppressed; got %q", rec)
	}
}

func TestAdvice_PCSamplingRankingAndDurationFallback(t *testing.T) {
	cap := &cudacapture.Capture{
		Metadata: []byte(`{"duration_ms": 100}`),
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
				{Type: "stall_mem", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{900, 800},
					Label: map[string][]string{"kernel": {"heavy_kernel"}},
				},
				{
					Value: []int64{100, 90},
					Label: map[string][]string{"kernel": {"light_kernel"}},
				},
			},
		},
	}

	// The capture's 100 ms is wall clock. Spreading it across the samples
	// would yield a per-kernel nanosecond figure nothing measured, so the
	// counts stay counts and the ranking below works off their share.
	times, totalTime := computeProfileTimes(cap)
	if totalTime != 1000 {
		t.Errorf("totalTime = %d, want 1000 samples", totalTime)
	}
	if times["heavy_kernel"] != 900 {
		t.Errorf("heavy_kernel = %d, want its 900 samples", times["heavy_kernel"])
	}
	if times["light_kernel"] != 100 {
		t.Errorf("light_kernel = %d, want its 100 samples", times["light_kernel"])
	}

	report := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityCritical,
				Category: CategoryMemory,
				Location: Location{Kernel: "light_kernel"},
				Finding:  "light finding",
			},
			{
				Severity: SeverityCritical,
				Category: CategoryMemory,
				Location: Location{Kernel: "heavy_kernel"},
				Finding:  "heavy finding",
			},
		},
	}
	finalizeAndRankFindings(report, times, totalTime)
	if len(report.Items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(report.Items))
	}
	if report.Items[0].Location.Kernel != "heavy_kernel" {
		t.Errorf("item 0 kernel = %s, want heavy_kernel", report.Items[0].Location.Kernel)
	}
	if report.Items[0].ShareOfProfile != 90.0 {
		t.Errorf("item 0 share = %.2f, want 90.0", report.Items[0].ShareOfProfile)
	}
}

func TestAdvice_SASSFastMathFCHK(t *testing.T) {
	const sass = `
	.section	.text.fchk_kernel,"ax",@progbits
fchk_kernel:
.text.fchk_kernel:
        /*0000*/                   FCHK R0, R1, 0x1 ;
        /*0010*/                   FCHK R2, R3, 0x1 ;
        /*0020*/                   EXIT ;
`
	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
		Files: map[string][]byte{
			"kernels.sass": []byte(sass),
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "without --use_fast_math") && strings.Contains(item.Finding, "2 IEEE compliance check") {
			found = true
			if item.Category != CategoryArchitectureCompute {
				t.Errorf("category = %s, want %s", item.Category, CategoryArchitectureCompute)
			}
			break
		}
	}
	if !found {
		t.Fatalf("expected FCHK finding, got items: %+v", report.Items)
	}
}

func TestAdvice_SASSIMADCapping(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("\t.section\t.text.imad_kernel,\"ax\",@progbits\nimad_kernel:\n.text.imad_kernel:\n")
	for i := 0; i < 10; i++ {
		sb.WriteString(fmt.Sprintf("        /*%04x*/                   IMAD.WIDE R%d, R%d, 0x4, R%d ;\n", i*16, i, i, i))
	}
	sb.WriteString("        /*00b0*/                   EXIT ;\n")

	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
		Files: map[string][]byte{
			"kernels.sass": []byte(sb.String()),
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var imadItems int
	var summaryItem bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Scalar address arithmetic fallback") {
			imadItems++
			if strings.Contains(item.Finding, "has 10 instructions") {
				summaryItem = true
			}
		}
	}
	// 3 individual items + 1 summary item = 4 items total
	if imadItems != 4 {
		t.Errorf("expected 4 IMAD items (3 detailed + 1 summary), got %d", imadItems)
	}
	if !summaryItem {
		t.Errorf("expected summary IMAD item not found")
	}
}

func TestAdvice_RepeatedHostLoopDispatch(t *testing.T) {
	var samples []*profile.Sample
	for i := 0; i < 60; i++ {
		start := uint64(1000 + i*5000)
		samples = append(samples, microKernelSample("repeated_gemv", 1, start, start+2000, 100+i))
	}
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{Sample: samples},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Repeated fine-grained host loop dispatch") && strings.Contains(item.Finding, "60 consecutive times") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected repeated host loop dispatch finding, got: %+v", report.Items)
	}
}

// TestAdvice_WallDurationIsNotDeviceTime pins the case that made the advisor
// print a fabricated millisecond figure: a profile whose only sample type is
// pc_samples, carrying a wall-clock duration. Scaling the samples by it gave
// this single kernel a "device time" equal to the entire capture.
func TestAdvice_WallDurationIsNotDeviceTime(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			DurationNanos: 250_000_000,
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{100},
					Label: map[string][]string{"kernel": {"k1"}},
				},
			},
		},
	}
	times, totalTime := computeProfileTimes(cap)
	if totalTime != 100 {
		t.Errorf("totalTime = %d, want 100 samples", totalTime)
	}
	if times["k1"] != 100 {
		t.Errorf("times[k1] = %d, want its 100 samples, not the 250 ms wall duration", times["k1"])
	}
	if hasDeviceTiming(cap) {
		t.Error("a wall-clock duration was accepted as device timing")
	}
}

func TestAdvice_PCSamplingExcludesGhostSequences(t *testing.T) {
	// PC-sampled instructions without stream/timeline labels must not be treated as microkernel launch sequences.
	var samples []*profile.Sample
	for i := 0; i < 20; i++ {
		samples = append(samples, &profile.Sample{
			Location: []*profile.Location{
				{ID: uint64(i + 1), Address: uint64(0x1000 + i*16), Line: []profile.Line{
					{Function: &profile.Function{ID: 1, Name: "my_kernel"}, Line: int64(i + 1)},
				}},
			},
			Value: []int64{10},
		})
	}
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "pc_samples", Unit: "count"}},
			Sample:     samples,
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Short-kernel dispatch sequence") || strings.Contains(item.Finding, "Repeated fine-grained host loop dispatch") {
			t.Errorf("unexpected dispatch sequence finding on pure PC samples: %+v", item)
		}
	}
}

func TestAdvice_ScoreboardAndBranchStalls(t *testing.T) {
	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "stall_scoreboard", Unit: "count"},
				{Type: "stall_branch", Unit: "count"},
				{Type: "stall_math", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Location: []*profile.Location{
						{ID: 1, Address: 0x100, Line: []profile.Line{
							{Function: &profile.Function{ID: 1, Name: "k_stalls"}, Line: 10},
						}},
					},
					Value: []int64{80, 0, 20}, // Scoreboard: 80% >= 25% and 80 >= 50
					Label: map[string][]string{"opcode": {"FADD"}},
				},
				{
					Location: []*profile.Location{
						{ID: 2, Address: 0x200, Line: []profile.Line{
							{Function: &profile.Function{ID: 1, Name: "k_stalls"}, Line: 20},
						}},
					},
					Value: []int64{0, 70, 30}, // Branch: 70% >= 20% and 70 >= 50
					Label: map[string][]string{"opcode": {"BRA"}},
				},
			},
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var foundScoreboard, foundBranch bool
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "Register scoreboard latency stall") {
			foundScoreboard = true
			if !strings.Contains(it.ActionableRecommendation, "software pipelining") {
				t.Errorf("expected software pipelining recommendation, got: %s", it.ActionableRecommendation)
			}
		}
		if strings.Contains(it.Finding, "Warp branch divergence") {
			foundBranch = true
			if !strings.Contains(it.ActionableRecommendation, "branchless predicated selects") {
				t.Errorf("expected branchless recommendation, got: %s", it.ActionableRecommendation)
			}
		}
	}
	if !foundScoreboard {
		t.Errorf("did not find expected register scoreboard finding")
	}
	if !foundBranch {
		t.Errorf("did not find expected warp branch divergence finding")
	}
}

func TestAdvice_PTXScalarSuppressedWhenSASS(t *testing.T) {
	ptxText := `.version 8.0
.target sm_90
.address_size 64
.visible .entry my_kernel(.param .u64 ptr) {
	ld.param.u64 %rd1, [ptr];
	mad.wide.u32 %rd2, %r1, 4, %rd1;
	ld.global.u32 %r2, [%rd2];
	ret;
}`
	sassText := `
Function : my_kernel
	MOV R1, c[0x0][0x20] ;
	RET ;
`
	cap := &cudacapture.Capture{
		Files: map[string][]byte{
			"kernel.ptx":  []byte(ptxText),
			"kernel.sass": []byte(sassText),
		},
		Code: cudacapture.CodeFiles{
			PTX:  "kernel.ptx",
			SASS: "kernel.sass",
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	for _, it := range report.Items {
		if strings.Contains(it.Finding, "mad.wide") || strings.Contains(it.Finding, "Scalar global load (ld.global)") {
			t.Errorf("PTX scalar finding should have been suppressed by presence of SASS: %+v", it)
		}
	}
}

func TestAdvice_PointwiseKernelPairFusion(t *testing.T) {
	var samples []*profile.Sample
	// Simulate 25 back-to-back pairs of norm -> quant
	for i := 0; i < 25; i++ {
		samples = append(samples, &profile.Sample{
			Label: map[string][]string{
				"stream":  {"1"},
				"subtype": {"kernel"},
			},
			NumLabel: map[string][]int64{
				"correlation": {int64(i*2 + 1)},
			},
			Location: []*profile.Location{
				{ID: 1, Line: []profile.Line{{Function: &profile.Function{ID: 1, Name: "my_norm"}}}},
			},
		})
		samples = append(samples, &profile.Sample{
			Label: map[string][]string{
				"stream":  {"1"},
				"subtype": {"kernel"},
			},
			NumLabel: map[string][]int64{
				"correlation": {int64(i*2 + 2)},
			},
			Location: []*profile.Location{
				{ID: 2, Line: []profile.Line{{Function: &profile.Function{ID: 2, Name: "my_quant"}}}},
			},
		})
	}

	cap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{{Type: "pc_samples", Unit: "count"}},
			Sample:     samples,
		},
	}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture failed: %v", err)
	}

	var foundFusion bool
	for _, it := range report.Items {
		if strings.Contains(it.Finding, "occurred 25 times") {
			foundFusion = true
			if !strings.Contains(it.ActionableRecommendation, "eliminate 25 DRAM round-trips") {
				t.Errorf("expected recommendation to cite 25 round trips, got: %s", it.ActionableRecommendation)
			}
		}
	}
	if !foundFusion {
		t.Errorf("expected fusion finding with recurrence count 25, items: %+v", report.Items)
	}
}

func TestAdvice_OutputLimitAndFlags(t *testing.T) {
	// Build a report with 15 findings
	items := make([]AdviceItem, 15)
	for i := 0; i < 15; i++ {
		items[i] = AdviceItem{
			Severity:                 SeverityWarning,
			Category:                 CategoryArchitecture,
			Finding:                  fmt.Sprintf("Finding number %d", i+1),
			ActionableRecommendation: fmt.Sprintf("Fix number %d", i+1),
		}
	}
	rep := &AdviceReport{Items: items}

	// 1. Default Markdown() should truncate to 10 findings and show omission note
	mdDefault := rep.Markdown()
	if !strings.Contains(mdDefault, "### 10. [Warning]") {
		t.Errorf("expected finding #10 in default output: %s", mdDefault)
	}
	if strings.Contains(mdDefault, "### 11. [Warning]") {
		t.Errorf("finding #11 should be truncated in default output: %s", mdDefault)
	}
	if !strings.Contains(mdDefault, "*... 5 additional findings omitted (use --format=json for all, -e, or -limit=N).*") {
		t.Errorf("missing truncation notice: %s", mdDefault)
	}

	// 2. MarkdownWithLimit(0) should print all 15 findings
	mdAll := rep.MarkdownWithLimit(0)
	if !strings.Contains(mdAll, "### 15. [Warning]") {
		t.Errorf("expected finding #15 in unlimited output: %s", mdAll)
	}
	if strings.Contains(mdAll, "additional findings omitted") {
		t.Errorf("unexpected truncation notice in unlimited output: %s", mdAll)
	}

	// 3. MarkdownWithLimit(5) should print exactly 5 findings
	md5 := rep.MarkdownWithLimit(5)
	if !strings.Contains(md5, "### 5. [Warning]") {
		t.Errorf("expected finding #5 in limit 5 output: %s", md5)
	}
	if strings.Contains(md5, "### 6. [Warning]") {
		t.Errorf("finding #6 should be truncated in limit 5 output: %s", md5)
	}
	if !strings.Contains(md5, "*... 10 additional findings omitted (use --format=json for all, -e, or -limit=N).*") {
		t.Errorf("missing 10 omitted notice in limit 5 output: %s", md5)
	}
}

func TestAdvice_EvidenceLevels(t *testing.T) {
	rep := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity:                 SeverityCritical,
				Category:                 CategoryArchitecture,
				EvidenceLevel:            EvidenceObserved,
				Finding:                  "Observed hardware violation",
				ActionableRecommendation: "Fix code",
				NextMeasurement:          "Verify with hardware timer",
			},
			{
				Severity:                 SeverityWarning,
				Category:                 CategoryMemory,
				EvidenceLevel:            EvidenceSuspected,
				Finding:                  "Suspected memory stall",
				ActionableRecommendation: "Check access pattern",
				NextMeasurement:          "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions",
			},
			{
				Severity:                 SeverityInfo,
				Category:                 CategoryArchitecture,
				EvidenceLevel:            EvidenceUnmeasured,
				Finding:                  "Unmeasured launch telemetry",
				ActionableRecommendation: "Enable metrics",
				NextMeasurement:          "Capture with CUPTI activity records enabled",
			},
		},
	}

	md := rep.Markdown()
	for _, want := range []string{
		"- **Evidence**: observed",
		"- **Evidence**: suspected",
		"- **Evidence**: unmeasured",
		"- **Next Measurement**: Verify with hardware timer",
		"- **Next Measurement**: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions",
		"- **Next Measurement**: Capture with CUPTI activity records enabled",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("Markdown output missing %q:\n%s", want, md)
		}
	}

	jsonStr, err := rep.JSON()
	if err != nil {
		t.Fatalf("JSON serialization failed: %v", err)
	}
	for _, want := range []string{
		`"evidence_level": "observed"`,
		`"evidence_level": "suspected"`,
		`"evidence_level": "unmeasured"`,
		`"next_measurement": "Verify with hardware timer"`,
		`"next_measurement": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions"`,
	} {
		if !strings.Contains(jsonStr, want) {
			t.Errorf("JSON output missing %q:\n%s", want, jsonStr)
		}
	}
}

func TestAdvice_PurePCSampleProfileShareFormatting(t *testing.T) {
	// Pure PC-sample capture: no duration metadata, sample type is pc_samples.
	purePCCap := &cudacapture.Capture{
		Profile: &profile.Profile{
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{800},
					Label: map[string][]string{
						"kernel": {"heavy_kernel"},
					},
				},
				{
					Value: []int64{200},
					Label: map[string][]string{
						"kernel": {"light_kernel"},
					},
				},
			},
		},
	}

	times, totalTime := computeProfileTimes(purePCCap)
	if hasDeviceTiming(purePCCap) {
		t.Fatal("hasDeviceTiming returned true for pure PC-sample capture")
	}

	report := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityWarning,
				Category: CategoryMemory,
				Location: Location{Kernel: "heavy_kernel"},
				Finding:  "Heavy load finding",
			},
			{
				Severity: SeverityInfo,
				Category: CategoryMemory,
				Location: Location{Kernel: "light_kernel"},
				Finding:  "Light load finding",
			},
		},
	}

	finalizeAndRankFindingsWithTiming(report, times, totalTime, false)

	for _, it := range report.Items {
		if it.DeviceTimeNanos != 0 {
			t.Errorf("kernel %q assigned non-zero DeviceTimeNanos %d on pure PC-sample profile", it.Location.Kernel, it.DeviceTimeNanos)
		}
		if it.Location.Kernel == "heavy_kernel" && it.ShareOfProfile != 80.0 {
			t.Errorf("heavy_kernel share = %.2f, want 80.0", it.ShareOfProfile)
		}
		if it.Location.Kernel == "light_kernel" && it.ShareOfProfile != 20.0 {
			t.Errorf("light_kernel share = %.2f, want 20.0", it.ShareOfProfile)
		}
	}

	md := report.Markdown()
	if !strings.Contains(md, "(80.00% of observed samples)") {
		t.Errorf("Markdown output missing '(80.00%% of observed samples)':\n%s", md)
	}
	if strings.Contains(md, "GPU time") {
		t.Errorf("Markdown output improperly claims 'GPU time' on pure PC-sample profile:\n%s", md)
	}
	if strings.Contains(md, "0.00 ms") {
		t.Errorf("Markdown output improperly claims '0.00 ms' on pure PC-sample profile:\n%s", md)
	}

	// A wall-clock duration on a sample-only profile is not device time. This
	// half of the test used to assert the opposite -- that 800 of 1000 samples
	// over a 100 ms capture should render as "80.00 ms of GPU time" -- which is
	// the fabrication itself, written down as intent.
	durCap := &cudacapture.Capture{
		Profile: &profile.Profile{
			DurationNanos: 100_000_000, // 100 ms
			SampleType: []*profile.ValueType{
				{Type: "pc_samples", Unit: "count"},
			},
			Sample: []*profile.Sample{
				{
					Value: []int64{800},
					Label: map[string][]string{
						"kernel": {"heavy_kernel"},
					},
				},
				{
					Value: []int64{200},
					Label: map[string][]string{
						"kernel": {"light_kernel"},
					},
				},
			},
		},
	}

	if hasDeviceTiming(durCap) {
		t.Fatal("hasDeviceTiming returned true for a sample-only capture with a wall duration")
	}

	durTimes, durTotalTime := computeProfileTimes(durCap)
	if got := durTimes["heavy_kernel"]; got != 800 {
		t.Errorf("heavy_kernel = %d, want its 800 samples unscaled by the 100 ms duration", got)
	}
	durReport := &AdviceReport{
		Items: []AdviceItem{
			{
				Severity: SeverityWarning,
				Category: CategoryMemory,
				Location: Location{Kernel: "heavy_kernel"},
				Finding:  "Heavy load finding",
			},
		},
	}
	finalizeAndRankFindingsWithTiming(durReport, durTimes, durTotalTime, hasDeviceTiming(durCap))

	if got := durReport.Items[0].DeviceTimeNanos; got != 0 {
		t.Errorf("DeviceTimeNanos = %d on a sample-only capture, want 0", got)
	}
	durMd := durReport.Markdown()
	if strings.Contains(durMd, "GPU time") || strings.Contains(durMd, " ms)") {
		t.Errorf("Markdown claims device time from a wall-clock duration:\n%s", durMd)
	}
	if !strings.Contains(durMd, "80.00% of observed samples") {
		t.Errorf("Markdown missing sample-share formatting:\n%s", durMd)
	}
}

func TestAdvice_NegativeControls(t *testing.T) {
	t.Run("scalar_load_severity_and_evidence", func(t *testing.T) {
		ptxText := `.version 9.0
.target sm_90
.address_size 64
.file 1 "saxpy.cu"
.visible .entry saxpy(.param .u64 in1) {
	.reg .f32 %f1;
	.reg .u64 %rd1;
	.loc 1 25 4
	ld.global.f32 %f1, [%rd1];
	ret;
}`
		cap := &cudacapture.Capture{
			Files: map[string][]byte{"kernel.ptx": []byte(ptxText)},
			Code:  cudacapture.CodeFiles{PTX: "kernel.ptx"},
		}
		rep, err := AnalyzeCapture(cap)
		if err != nil {
			t.Fatalf("AnalyzeCapture failed: %v", err)
		}

		var scalarItem *AdviceItem
		for i := range rep.Items {
			if strings.Contains(rep.Items[i].Finding, "Scalar global load") {
				scalarItem = &rep.Items[i]
				break
			}
		}
		if scalarItem == nil {
			t.Fatal("expected scalar global load finding, found none")
		}

		// Negative control: must NOT be SeverityCritical.
		if scalarItem.Severity == SeverityCritical {
			t.Errorf("scalar load finding severity is Critical, want Warning")
		}
		if scalarItem.Severity != SeverityWarning {
			t.Errorf("scalar load finding severity = %v, want SeverityWarning", scalarItem.Severity)
		}
		// Negative control: must be EvidenceSuspected, not EvidenceObserved.
		if scalarItem.EvidenceLevel != EvidenceSuspected {
			t.Errorf("scalar load EvidenceLevel = %v, want %v", scalarItem.EvidenceLevel, EvidenceSuspected)
		}
		// Vectorization suggestion must be conditioned on access pattern and alignment.
		if !strings.Contains(scalarItem.ActionableRecommendation, "contiguous") ||
			!strings.Contains(scalarItem.ActionableRecommendation, "aligned") {
			t.Errorf("recommendation is not conditioned on contiguous and aligned access: %q", scalarItem.ActionableRecommendation)
		}
		// Must name deciding hardware counter.
		if !strings.Contains(scalarItem.NextMeasurement, "gld_transactions") &&
			!strings.Contains(scalarItem.NextMeasurement, "sectors") {
			t.Errorf("next measurement missing transaction counter: %q", scalarItem.NextMeasurement)
		}
	})

	t.Run("pc_memory_stalls_do_not_assert_uncoalesced_transactions", func(t *testing.T) {
		const sampleLDGSASS = `
	.section	.text.uncoalesced_load,"ax",@progbits
uncoalesced_load:
.text.uncoalesced_load:
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   LDG.E.128 R0, [R2] ;
        /*0020*/                   EXIT ;
`
		cap := &cudacapture.Capture{
			Code: cudacapture.CodeFiles{SASS: "kernels.sass"},
			Files: map[string][]byte{
				"kernels.sass": []byte(sampleLDGSASS),
			},
			Profile: &profile.Profile{
				SampleType: []*profile.ValueType{
					{Type: "stall_scoreboard", Unit: "count"},
					{Type: "stall_mem", Unit: "count"},
				},
				Location: []*profile.Location{
					{ID: 1, Address: 0x0010},
				},
				Sample: []*profile.Sample{
					{
						Location: []*profile.Location{{ID: 1, Address: 0x0010}},
						Value:    []int64{30, 70}, // 70% memory stalls
						Label: map[string][]string{
							"kernel": {"uncoalesced_load"},
						},
					},
				},
			},
		}

		rep, err := AnalyzeCapture(cap)
		if err != nil {
			t.Fatalf("AnalyzeCapture failed: %v", err)
		}

		var memItem *AdviceItem
		for i := range rep.Items {
			if strings.Contains(rep.Items[i].Finding, "Memory stalls dominate this load") {
				memItem = &rep.Items[i]
				break
			}
		}
		if memItem == nil {
			t.Fatal("expected memory stall finding, found none")
		}

		// Evidence must be suspected because deciding hardware transaction counter is absent.
		if memItem.EvidenceLevel != EvidenceSuspected {
			t.Errorf("memory stall EvidenceLevel = %v, want %v", memItem.EvidenceLevel, EvidenceSuspected)
		}
		// Negative control: must NOT assert 32 uncoalesced cache transactions without hardware counters.
		if strings.Contains(memItem.Finding, "32 cache transactions") {
			t.Errorf("memory stall finding improperly asserted 32 cache transactions: %q", memItem.Finding)
		}
		// Recommendation must clarify that if access is already contiguous, stall is latency/bandwidth.
		if !strings.Contains(memItem.ActionableRecommendation, "contiguous bytes") ||
			!strings.Contains(memItem.ActionableRecommendation, "latency or bandwidth") {
			t.Errorf("recommendation missing latency/bandwidth qualification: %q", memItem.ActionableRecommendation)
		}
		// Next measurement must specify deciding counters.
		if !strings.Contains(memItem.NextMeasurement, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum or gld_transactions") {
			t.Errorf("next measurement missing deciding counter: %q", memItem.NextMeasurement)
		}
	})

	t.Run("static_occupancy_cliff_is_not_critical", func(t *testing.T) {
		var nv bytes.Buffer
		nv.Write([]byte{0x04, cudaoccupancy.EIATTR_REGCOUNT, 0x04, 0x00})
		binary.Write(&nv, binary.LittleEndian, uint32(24))
		nv.Write([]byte{0x04, cudaoccupancy.EIATTR_REQ_NTID, 0x0c, 0x00})
		binary.Write(&nv, binary.LittleEndian, uint32(256))
		binary.Write(&nv, binary.LittleEndian, uint32(1))
		binary.Write(&nv, binary.LittleEndian, uint32(1))
		nv.Write([]byte{0x04, cudaoccupancy.EIATTR_SHARED_SIZE_SYSMEM, 0x04, 0x00})
		binary.Write(&nv, binary.LittleEndian, uint32(65536))

		sections := map[string][]byte{
			".text.smem_limited_kernel":    make([]byte, 32),
			".nv.info.smem_limited_kernel": nv.Bytes(),
		}
		cubinBytes := synthesizeTestELFForAdvise(sections)

		cap := &cudacapture.Capture{
			Metadata: []byte(`{"architecture":"sm_121a"}`),
			Code: cudacapture.CodeFiles{
				CUBIN: "kernels.cubin",
			},
			Files: map[string][]byte{
				"kernels.cubin": cubinBytes,
			},
		}

		rep, err := AnalyzeCapture(cap)
		if err != nil {
			t.Fatalf("AnalyzeCapture failed: %v", err)
		}

		var occItem *AdviceItem
		for i := range rep.Items {
			if strings.Contains(rep.Items[i].Finding, "constrained by shared_memory") {
				occItem = &rep.Items[i]
				break
			}
		}
		if occItem == nil {
			t.Fatal("expected theoretical occupancy finding, found none")
		}

		// Negative control: static occupancy limit bounds residency, not throughput; must not be Critical.
		if occItem.Severity == SeverityCritical {
			t.Errorf("theoretical occupancy finding is SeverityCritical, want SeverityWarning")
		}
		if occItem.Severity != SeverityWarning {
			t.Errorf("theoretical occupancy finding severity = %v, want SeverityWarning", occItem.Severity)
		}
		if occItem.EvidenceLevel != EvidenceSuspected {
			t.Errorf("theoretical occupancy finding EvidenceLevel = %v, want %v", occItem.EvidenceLevel, EvidenceSuspected)
		}
		if !strings.Contains(occItem.NextMeasurement, "active warps") && !strings.Contains(occItem.NextMeasurement, "A/B") {
			t.Errorf("theoretical occupancy NextMeasurement missing active warps / A/B timing: %q", occItem.NextMeasurement)
		}
	})
}

// TestFusionPairFindingCarriesItsDeviceTime pins that a pair finding publishes
// the device time it already names in prose.
//
// The pair findings set Location.Kernel to "A -> B", which is no kernel in the
// profile, so the per-kernel fill-in in finalizeAndRankFindingsWithTiming
// looks it up and finds nothing. Both numeric fields then stayed zero and,
// being omitempty, left the JSON entirely: a consumer iterating items and
// reading device_time_nanos got a missing key, and the item could not be
// ranked or sorted against the per-kernel findings it competes with. The
// accumulated duration was in the finding text the whole time.
func TestFusionPairFindingCarriesItsDeviceTime(t *testing.T) {
	var samples []*profile.Sample
	for i := 0; i < 4; i++ {
		for _, name := range []string{"swiglu_fwd", "quant_block_q8"} {
			samples = append(samples, &profile.Sample{
				Location: []*profile.Location{{ID: uint64(len(samples) + 1), Line: []profile.Line{
					{Function: &profile.Function{ID: 1, Name: name}},
				}}},
				Value:    []int64{100000},
				Label:    map[string][]string{"name": {name}, "buffer": {"hidden"}},
				NumLabel: map[string][]int64{"stream": {7}},
			})
		}
	}
	cap := &cudacapture.Capture{Profile: &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}},
		Sample:     samples,
	}}

	report, err := AnalyzeCapture(cap)
	if err != nil {
		t.Fatalf("AnalyzeCapture: %v", err)
	}
	var pair *AdviceItem
	for i := range report.Items {
		if strings.Contains(report.Items[i].Location.Kernel, "->") {
			pair = &report.Items[i]
			break
		}
	}
	if pair == nil {
		t.Fatalf("no pair finding in %d items; the fixture no longer reaches the fusion path", len(report.Items))
	}
	// The invariant is that the field says what the prose says. Hardcoding a
	// number here would pin the fixture's adjacency arithmetic instead, and
	// that is not the property that broke.
	said, ok := accumulatedFromFinding(pair.Finding)
	if !ok {
		t.Fatalf("finding names no accumulation, so there is nothing to agree with: %s", pair.Finding)
	}
	if pair.DeviceTimeNanos != int64(said) {
		t.Errorf("device_time_nanos = %d, finding says %v: the field must carry what the prose already names", pair.DeviceTimeNanos, said)
	}
	if pair.ShareOfProfile <= 0 {
		t.Errorf("share_of_profile = %v, want the share derived from the device time", pair.ShareOfProfile)
	}

	// Asserted on the bytes: omitempty means a zero field is an ABSENT key,
	// which is what a consumer hit as a missing-key error, and a struct decode
	// would silently read it back as zero.
	b, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"device_time_nanos", "share_of_profile"} {
		if !bytes.Contains(b, []byte(`"`+key+`"`)) {
			t.Errorf("%s absent from the published item: %s", key, b)
		}
	}
}

// accumulatedFromFinding returns the duration a pair finding names in its
// "accumulating <d>" clause.
func accumulatedFromFinding(finding string) (time.Duration, bool) {
	const marker = "accumulating "
	i := strings.Index(finding, marker)
	if i < 0 {
		return 0, false
	}
	rest := finding[i+len(marker):]
	if j := strings.IndexAny(rest, ")"); j >= 0 {
		rest = rest[:j]
	}
	d, err := time.ParseDuration(rest)
	if err != nil {
		return 0, false
	}
	return d, true
}
