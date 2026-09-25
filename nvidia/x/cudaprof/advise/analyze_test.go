package advise

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

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
			wantSeverity: SeverityCost,
			wantFinding:  "140 GPRs",
			wantCaveat:   true,
		},
		{
			name:         "local_spill_points",
			sass:         sampleSpillSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityCost,
			wantFinding:  "spill point",
			wantCaveat:   false, // a spill is real work, not a residency figure
		},
		{
			name:         "high_ugpr",
			sass:         sampleHighUGPRSASS,
			wantCategory: CategoryRegisterPressure,
			wantSeverity: SeverityCost,
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

			report, err := Analyze(cap)
			if err != nil {
				t.Fatalf("Analyze failed: %v", err)
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

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	// Should contain TMA failure (Memory) and Tensor Core failure (Compute)
	var foundTMA, foundTensorCore bool
	for _, item := range findings(report) {
		if item.Category == CategoryMemory && item.Severity == SeverityCost {
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

		if item.Category == CategoryCompute && item.Severity == SeverityCost {
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
			wantSeverity: SeverityCost,
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
			wantSeverity: SeverityCost,
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
			wantSeverity: SeverityReadFirst,
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
			wantSeverity: SeverityCost,
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

			report, err := Analyze(cap)
			if err != nil {
				t.Fatalf("Analyze failed: %v", err)
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
		_, err := Analyze(nil)
		if err == nil {
			t.Error("expected error for nil capture, got nil")
		}
	})

	t.Run("empty_capture", func(t *testing.T) {
		cap := &cudacapture.Capture{}
		report, err := Analyze(cap)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
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
		var parsed Report
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

		report, err := Analyze(cap)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}
		if len(report.Items) != 0 {
			t.Errorf("expected 0 advice items for clean capture, got %d: %+v", len(report.Items), report.Items)
		}
	})
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

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var found bool
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "without --use_fast_math") && strings.Contains(item.Finding, "2 IEEE compliance check") {
			found = true
			if item.Category != CategoryArchitectureCompute {
				t.Errorf("category = %s, want %s", item.Category, CategoryArchitectureCompute)
			}
			if !strings.Contains(item.ActionableRecommendation, "changes division and transcendental rounding") ||
				!strings.Contains(item.ActionableRecommendation, "flushes denormals") ||
				!strings.Contains(item.ActionableRecommendation, "bit-exact logit invariants") {
				t.Errorf("fast-math recommendation lacks bit-exactness warning: %q", item.ActionableRecommendation)
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

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	var imad []string
	for _, item := range report.Items {
		if strings.Contains(item.Finding, "Scalar address arithmetic fallback") {
			imad = append(imad, item.Finding)
		}
	}
	// One finding for the kernel's pattern, however many instructions show it.
	const want = "Scalar address arithmetic fallback (IMAD.WIDE) in kernel \"imad_kernel\": 10 instructions, at 0x0000, 0x0010, 0x0020, +7 more"
	if len(imad) != 1 || imad[0] != want {
		t.Errorf("IMAD findings = %q, want [%q]", imad, want)
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

	report, err := Analyze(cap)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	for _, it := range report.Items {
		if strings.Contains(it.Finding, "mad.wide") || strings.Contains(it.Finding, "Scalar global load (ld.global)") {
			t.Errorf("PTX scalar finding should have been suppressed by presence of SASS: %+v", it)
		}
	}
}

func TestAdvice_OutputLimitAndFlags(t *testing.T) {
	// Build a report with 15 findings
	items := make([]Item, 15)
	for i := 0; i < 15; i++ {
		items[i] = Item{
			Severity:                 SeverityCost,
			Category:                 CategoryArchitecture,
			Finding:                  fmt.Sprintf("Finding number %d", i+1),
			ActionableRecommendation: fmt.Sprintf("Fix number %d", i+1),
		}
	}
	rep := &Report{Items: items}

	// 1. Default Markdown() should truncate to 10 findings and show omission note
	mdDefault := rep.Markdown()
	if !strings.Contains(mdDefault, "### 10. [Cost]") {
		t.Errorf("expected finding #10 in default output: %s", mdDefault)
	}
	if strings.Contains(mdDefault, "### 11. [Cost]") {
		t.Errorf("finding #11 should be truncated in default output: %s", mdDefault)
	}
	if !strings.Contains(mdDefault, "*... 5 additional findings omitted (use --format=json for all, -e, or -limit=N).*") {
		t.Errorf("missing truncation notice: %s", mdDefault)
	}

	// 2. MarkdownWithLimit(0) should print all 15 findings
	mdAll := rep.MarkdownWithLimit(0)
	if !strings.Contains(mdAll, "### 15. [Cost]") {
		t.Errorf("expected finding #15 in unlimited output: %s", mdAll)
	}
	if strings.Contains(mdAll, "additional findings omitted") {
		t.Errorf("unexpected truncation notice in unlimited output: %s", mdAll)
	}

	// 3. MarkdownWithLimit(5) should print exactly 5 findings
	md5 := rep.MarkdownWithLimit(5)
	if !strings.Contains(md5, "### 5. [Cost]") {
		t.Errorf("expected finding #5 in limit 5 output: %s", md5)
	}
	if strings.Contains(md5, "### 6. [Cost]") {
		t.Errorf("finding #6 should be truncated in limit 5 output: %s", md5)
	}
	if !strings.Contains(md5, "*... 10 additional findings omitted (use --format=json for all, -e, or -limit=N).*") {
		t.Errorf("missing 10 omitted notice in limit 5 output: %s", md5)
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
		rep, err := Analyze(cap)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}

		var scalarItem *Item
		for i := range rep.Items {
			if strings.Contains(rep.Items[i].Finding, "Scalar global load") {
				scalarItem = &rep.Items[i]
				break
			}
		}
		if scalarItem == nil {
			t.Fatal("expected scalar global load finding, found none")
		}

		// Negative control: must NOT be SeverityReadFirst.
		if scalarItem.Severity == SeverityReadFirst {
			t.Errorf("scalar load finding severity is Critical, want Warning")
		}
		if scalarItem.Severity != SeverityCost {
			t.Errorf("scalar load finding severity = %v, want SeverityCost", scalarItem.Severity)
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

		rep, err := Analyze(cap)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}

		var memItem *Item
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

		rep, err := Analyze(cap)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}

		var occItem *Item
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
		if occItem.Severity == SeverityReadFirst {
			t.Errorf("theoretical occupancy finding is SeverityReadFirst, want SeverityCost")
		}
		if occItem.Severity != SeverityCost {
			t.Errorf("theoretical occupancy finding severity = %v, want SeverityCost", occItem.Severity)
		}
		if occItem.EvidenceLevel != EvidenceSuspected {
			t.Errorf("theoretical occupancy finding EvidenceLevel = %v, want %v", occItem.EvidenceLevel, EvidenceSuspected)
		}
		if !strings.Contains(occItem.NextMeasurement, "active warps") && !strings.Contains(occItem.NextMeasurement, "A/B") {
			t.Errorf("theoretical occupancy NextMeasurement missing active warps / A/B timing: %q", occItem.NextMeasurement)
		}
	})
}
