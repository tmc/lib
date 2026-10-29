package advise

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaoccupancy"
)

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
			wantSeverity: SeverityCost,
			wantEvidence: EvidenceObserved,
			wantCaveat:   false, // a local stack frame spill is real work, not a residency figure
		},
		{
			name:         "blackwell_sm121_reg_cliff",
			targetArch:   "sm_121a",
			wantKernel:   "reg_limited_kernel",
			wantFinding:  "Microarchitecture theoretical occupancy on sm_121 for kernel \"reg_limited_kernel\"",
			wantRec:      "Register file capacity (65536 regs/SM) limits residency",
			wantSeverity: SeverityCost,
			wantEvidence: EvidenceSuspected,
			wantCaveat:   true,
		},
		{
			name:         "hopper_sm90_target_disambiguation",
			targetArch:   "sm_90a",
			wantKernel:   "reg_limited_kernel",
			wantFinding:  "Microarchitecture theoretical occupancy on sm_90 for kernel \"reg_limited_kernel\"",
			wantRec:      "Register file capacity (65536 regs/SM) limits residency",
			wantSeverity: SeverityCost,
			wantEvidence: EvidenceSuspected,
			wantCaveat:   true,
		},
		{
			name:         "smem_capacity_limit",
			targetArch:   "sm_121a",
			wantKernel:   "smem_limited_kernel",
			wantFinding:  "constrained by shared_memory",
			wantRec:      "Static shared memory (65536 bytes/block) limits residency",
			wantSeverity: SeverityCost,
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
			var report Report
			auditCUBINResourceOccupancy(cap, &report, kernelResources([]kernelImage{{cubin: cubinBytes}}), nil, nil)
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

			report, err := Analyze(cap)
			if err != nil {
				t.Fatalf("Analyze failed: %v", err)
			}

			var found bool
			for _, item := range findings(report) {
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

// These are CPU model controls, not admitted hardware measurements.
func TestOccupancyJointLimitsAndUnknownInputs(t *testing.T) {
	tests := []struct {
		name                                              string
		registers, shared, requiredThreads, launchThreads int
		want                                              []string
		absent                                            []string
	}{
		{
			name: "joint register and shared limits", registers: 128, shared: 112656, requiredThreads: 256,
			want:   []string{"<=80 GPRs still models 2 blocks/SM", "other known resource limits prevent 3 blocks/SM", "Register reduction alone is insufficient", "dynamic shared memory is unknown and excluded", "CUBIN required dimensions"},
			absent: []string{"would allow 3 blocks/SM", "could model 3 blocks/SM", "<=85 GPRs"},
		},
		{
			name: "register-only static counterfactual is conditional", registers: 128, requiredThreads: 256,
			want:   []string{"<=80 GPRs could model 3 blocks/SM", "same known static inputs", "dynamic shared memory is unknown and excluded", "sufficient grid work"},
			absent: []string{"would allow 3 blocks/SM", "Register reduction alone is insufficient"},
		},
		{
			name: "allocation granularity", registers: 85, shared: 77824, requiredThreads: 256,
			want:   []string{"from 85 to <=80 GPRs could model 3 blocks/SM"},
			absent: []string{"to <=85 GPRs"},
		},
		{
			name: "recorded block overrides required metadata", registers: 128, requiredThreads: 128, launchThreads: 256,
			want:   []string{"256 threads/block (recorded launch)", "<=80 GPRs could model 3 blocks/SM"},
			absent: []string{"128 threads/block", "assumed default"},
		},
		{
			name: "missing block metadata", registers: 128,
			want: []string{"128 threads/block (assumed default; actual block geometry unknown)", "Conditional static-resource model", "dynamic shared memory is unknown and excluded", "not achieved or complete launch residency"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := kernelResource{KernelResourceInfo: cudaoccupancy.KernelResourceInfo{Name: "cpu_model", Registers: tt.registers, SharedMemBytes: tt.shared}}
			if tt.requiredThreads > 0 {
				k.ReqNTID = [3]int{tt.requiredThreads, 1, 1}
			}
			var launches map[string]launchRecord
			if tt.launchThreads > 0 {
				launches = map[string]launchRecord{"cpu_model": {BlockThreads: tt.launchThreads}}
			}
			var report Report
			auditCUBINResourceOccupancy(&cudacapture.Capture{Metadata: []byte(`{"architecture":"sm_90"}`)}, &report, []kernelResource{k}, launches, nil)
			var item *Item
			for i := range report.Items {
				if report.Items[i].rule == "occupancy" {
					item = &report.Items[i]
				}
			}
			if item == nil {
				t.Fatal("missing conditional occupancy finding")
			}
			text := item.Finding + " " + item.ActionableRecommendation
			for _, want := range tt.want {
				if !strings.Contains(text, want) {
					t.Errorf("missing %q in %s", want, text)
				}
			}
			for _, absent := range tt.absent {
				if strings.Contains(text, absent) {
					t.Errorf("unqualified claim %q in %s", absent, text)
				}
			}
			if !strings.Contains(item.NextMeasurement, "dynamic shared bytes") {
				t.Errorf("next measurement omits missing dynamic input: %s", item.NextMeasurement)
			}
			if item.EvidenceLevel != EvidenceSuspected {
				t.Errorf("evidence %s, want suspected", item.EvidenceLevel)
			}
		})
	}
}
