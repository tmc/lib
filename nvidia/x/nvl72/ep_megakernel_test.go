package nvl72

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/expertparallel"
)

func TestRackEPMegakernelSingleAndMultiRackBlackwell(t *testing.T) {
	cases := []struct {
		name           string
		cfg            RackEPMegakernelConfig
		wantGPUs       int
		wantRDMAGBps   float64
		wantInterRack  bool
		wantRunOnArm   bool
		wantTargetArch string
	}{
		{
			name: "SingleRack_GB200_NVL72_C2C",
			cfg: RackEPMegakernelConfig{
				Name:            "gb200_1rack_ep",
				Generation:      GenGB200,
				NumRacks:        1,
				TokensPerRank:   64,
				HiddenSize:      4096,
				TopK:            8,
				TotalExperts:    288,
				TotalCTAs:       148,
				ThreadsPerCTA:   256,
				Microbatches:    2,
				EnableC2CHybrid: true,
				ColdTokenCount:  1,
			},
			wantGPUs:       72,
			wantRDMAGBps:   50.0,
			wantInterRack:  false,
			wantRunOnArm:   true,
			wantTargetArch: "sm_100a",
		},
		{
			name: "SingleRack_GB300_NVL72_C2C",
			cfg: RackEPMegakernelConfig{
				Name:            "gb300_1rack_ep",
				Generation:      GenGB300,
				NumRacks:        1,
				TokensPerRank:   64,
				HiddenSize:      7168,
				TopK:            8,
				TotalExperts:    288,
				TotalCTAs:       148,
				ThreadsPerCTA:   256,
				Microbatches:    4,
				EnableC2CHybrid: true,
				ColdTokenCount:  1,
			},
			wantGPUs:       72,
			wantRDMAGBps:   100.0,
			wantInterRack:  false,
			wantRunOnArm:   true,
			wantTargetArch: "sm_103a",
		},
		{
			name: "FourRack_GB200_Pod_RDMA",
			cfg: RackEPMegakernelConfig{
				Name:            "gb200_4rack_ep",
				Generation:      GenGB200,
				NumRacks:        4,
				TokensPerRank:   64,
				HiddenSize:      7168,
				TopK:            8,
				TotalExperts:    288,
				TotalCTAs:       148,
				ThreadsPerCTA:   256,
				Microbatches:    2,
				EnableC2CHybrid: true,
				ColdTokenCount:  1,
			},
			wantGPUs:       288,
			wantRDMAGBps:   50.0,
			wantInterRack:  true,
			wantRunOnArm:   true,
			wantTargetArch: "sm_100a",
		},
		{
			name: "FourRack_GB300_Pod_RDMA",
			cfg: RackEPMegakernelConfig{
				Name:            "gb300_4rack_ep",
				Generation:      GenGB300,
				NumRacks:        4,
				TokensPerRank:   64,
				HiddenSize:      7168,
				TopK:            8,
				TotalExperts:    288,
				TotalCTAs:       148,
				ThreadsPerCTA:   256,
				Microbatches:    2,
				EnableC2CHybrid: true,
				ColdTokenCount:  64,
			},
			wantGPUs:       288,
			wantRDMAGBps:   100.0,
			wantInterRack:  true,
			wantRunOnArm:   false,
			wantTargetArch: "sm_103a",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := CompileRackEPMegakernel(tc.cfg)
			if err != nil {
				t.Fatalf("CompileRackEPMegakernel: %v", err)
			}
			if plan.TotalGPUs != tc.wantGPUs {
				t.Fatalf("TotalGPUs = %d, want %d", plan.TotalGPUs, tc.wantGPUs)
			}
			if plan.TargetArch != tc.wantTargetArch {
				t.Fatalf("TargetArch = %q, want %q", plan.TargetArch, tc.wantTargetArch)
			}
			if plan.PodEstimate.Tier4RDMAUniGBps != tc.wantRDMAGBps {
				t.Fatalf("Tier4RDMAUniGBps = %v, want %v", plan.PodEstimate.Tier4RDMAUniGBps, tc.wantRDMAGBps)
			}
			if tc.wantInterRack {
				if plan.PodEstimate.HierarchicalInterRackDispatchBytes == 0 ||
					plan.PodEstimate.HierarchicalInterRackDispatchBytes >= plan.PodEstimate.FlatInterRackDispatchBytes {
					t.Fatalf("expected hierarchical rack-deduped RDMA bytes (%d) < flat (%d)",
						plan.PodEstimate.HierarchicalInterRackDispatchBytes, plan.PodEstimate.FlatInterRackDispatchBytes)
				}
				if plan.PodEstimate.RackDedupSavingsRatio <= 0 {
					t.Fatalf("expected positive RackDedupSavingsRatio, got %v", plan.PodEstimate.RackDedupSavingsRatio)
				}
			} else if plan.PodEstimate.HierarchicalInterRackDispatchBytes != 0 {
				t.Fatalf("expected 0 inter-rack RDMA bytes for single rack, got %d",
					plan.PodEstimate.HierarchicalInterRackDispatchBytes)
			}
			if plan.ColdPlacement.RunOnArmInPlace != tc.wantRunOnArm {
				t.Fatalf("ColdPlacement.RunOnArmInPlace = %v, want %v", plan.ColdPlacement.RunOnArmInPlace, tc.wantRunOnArm)
			}
			if plan.ColdTokenCrossover < 2 {
				t.Fatalf("ColdTokenCrossover = %d, want >= 2", plan.ColdTokenCrossover)
			}
			if plan.ComputeCTAs+plan.DispatchCTAs+plan.CombineCTAs != plan.TotalCTAs {
				t.Fatalf("CTA partition %d+%d+%d != %d",
					plan.ComputeCTAs, plan.DispatchCTAs, plan.CombineCTAs, plan.TotalCTAs)
			}
			if plan.C2CDoorbellCTAs != 1 {
				t.Fatalf("C2CDoorbellCTAs = %d, want 1", plan.C2CDoorbellCTAs)
			}
			if got := plan.RoleForCTA(0); got != expertparallel.WarpgroupRoleCompute {
				t.Fatalf("RoleForCTA(0) = %v, want compute", got)
			}
			if got := plan.RoleForCTA(plan.ComputeCTAs + plan.DispatchCTAs - 1); got != expertparallel.WarpgroupRoleC2CDoorbell {
				t.Fatalf("RoleForCTA(doorbell) = %v, want c2c_doorbell", got)
			}
			if got := plan.RoleForCTA(plan.TotalCTAs - 1); got != expertparallel.WarpgroupRoleFabricCombine {
				t.Fatalf("RoleForCTA(last) = %v, want fabric_combine", got)
			}
			if plan.TBOSpeedup <= 1.0 || plan.TBOTwoBatchUs >= plan.SequentialTwoBatchUs {
				t.Fatalf("expected TBO speedup > 1.0: seq=%v us tbo=%v us speedup=%v",
					plan.SequentialTwoBatchUs, plan.TBOTwoBatchUs, plan.TBOSpeedup)
			}
			if err := AuditC2CCoherence(plan.DoorbellPTX); err != nil {
				t.Fatalf("AuditC2CCoherence(DoorbellPTX): %v", err)
			}
			if err := AuditC2CCoherence(plan.Source); err != nil {
				t.Fatalf("AuditC2CCoherence(Source): %v", err)
			}
			for _, want := range []string{
				plan.Name,
				"const int slot = step & 1;",
				"stage1_rdma_ready",
				"stage2_nvlink_ready",
				"st.release.sys.global.u32",
				"ld.acquire.sys.global.u32",
				"st.release.sys.global.u64",
				"ld.acquire.sys.global.u64",
				"eg_rack_ep_c2c_doorbell_handoff",
			} {
				if !strings.Contains(plan.Source, want) {
					t.Fatalf("Source missing %q", want)
				}
			}
		})
	}
}

func TestRackEPMegakernelRubinNVL144AndNVL576(t *testing.T) {
	cases := []struct {
		name          string
		cfg           RackEPMegakernelConfig
		wantDies      int
		wantInterRack bool
	}{
		{
			name: "VeraRubin_NVL144_SingleRack",
			cfg: RackEPMegakernelConfig{
				Name:            "rubin_nvl144_1rack",
				RubinTier:       TierVeraRubinNVL144,
				NumRacks:        1,
				TokensPerRank:   64,
				HiddenSize:      7168,
				TopK:            8,
				TotalExperts:    288,
				EnableC2CHybrid: true,
				ColdTokenCount:  1,
				UseFP4Dispatch:  true,
			},
			wantDies:      144,
			wantInterRack: false,
		},
		{
			name: "VeraRubin_NVL144_FourRackPod",
			cfg: RackEPMegakernelConfig{
				Name:            "rubin_nvl144_4rack",
				Generation:      "VeraRubin-NVL144",
				NumRacks:        4,
				TokensPerRank:   64,
				HiddenSize:      7168,
				TopK:            8,
				TotalExperts:    576,
				EnableC2CHybrid: true,
				ColdTokenCount:  1,
				UseFP4Dispatch:  true,
			},
			wantDies:      576,
			wantInterRack: true,
		},
		{
			name: "RubinUltra_NVL576_SingleRack",
			cfg: RackEPMegakernelConfig{
				Name:            "rubin_ultra_nvl576_1rack",
				RubinTier:       TierRubinUltraNVL576,
				NumRacks:        1,
				TokensPerRank:   64,
				HiddenSize:      7168,
				TopK:            8,
				TotalExperts:    576,
				EnableC2CHybrid: true,
				ColdTokenCount:  1,
				UseFP4Dispatch:  true,
			},
			wantDies:      576,
			wantInterRack: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := CompileRackEPMegakernel(tc.cfg)
			if err != nil {
				t.Fatalf("CompileRackEPMegakernel: %v", err)
			}
			if !plan.IsRubin || plan.TargetArch != RubinTargetArch {
				t.Fatalf("IsRubin=%v TargetArch=%q, want true, %q", plan.IsRubin, plan.TargetArch, RubinTargetArch)
			}
			if plan.TotalGPUs != tc.wantDies || plan.RubinEstimate.TotalDies != tc.wantDies {
				t.Fatalf("TotalGPUs=%d RubinEstimate.TotalDies=%d, want %d",
					plan.TotalGPUs, plan.RubinEstimate.TotalDies, tc.wantDies)
			}
			if plan.TotalCTAs != 160 {
				t.Fatalf("default Rubin TotalCTAs = %d, want 160", plan.TotalCTAs)
			}
			if tc.wantInterRack {
				if plan.RubinEstimate.InterRackDispatchUS <= 0 {
					t.Fatalf("expected positive InterRackDispatchUS for 4-rack Rubin pod, got %v",
						plan.RubinEstimate.InterRackDispatchUS)
				}
			} else if plan.RubinEstimate.InterRackDispatchUS != 0 {
				t.Fatalf("expected 0 InterRackDispatchUS for 1-rack Rubin, got %v",
					plan.RubinEstimate.InterRackDispatchUS)
			}
			if plan.RubinEstimate.SpeedupVsGB300Pod <= 1.8 {
				t.Fatalf("SpeedupVsGB300Pod = %.3f, want > 1.8", plan.RubinEstimate.SpeedupVsGB300Pod)
			}
			if !plan.ColdPlacement.RunOnArmInPlace {
				t.Fatalf("expected 1-token cold expert on Vera CPU in place: %+v", plan.ColdPlacement)
			}
			if err := AuditC2CCoherence(plan.DoorbellPTX); err != nil {
				t.Fatalf("AuditC2CCoherence(DoorbellPTX): %v", err)
			}
			if err := AuditC2CCoherence(plan.Source); err != nil {
				t.Fatalf("AuditC2CCoherence(Source): %v", err)
			}
			for _, want := range []string{
				"multimem.ld_reduce.relaxed.sys.global.add.f32",
				"multimem.st.relaxed.sys.global.f32",
				"eg_rack_ep_c2c_doorbell_handoff",
			} {
				if !strings.Contains(plan.Source, want) {
					t.Fatalf("Rubin Source missing %q", want)
				}
			}
		})
	}
}

func TestRackEPMegakernelValidationErrors(t *testing.T) {
	badConfigs := []struct {
		name string
		cfg  RackEPMegakernelConfig
	}{
		{
			name: "BadName",
			cfg:  RackEPMegakernelConfig{Name: "bad name", TokensPerRank: 16, HiddenSize: 128, TopK: 2},
		},
		{
			name: "BadGeneration",
			cfg:  RackEPMegakernelConfig{Generation: "H100", TokensPerRank: 16, HiddenSize: 128, TopK: 2},
		},
		{
			name: "BadTokensPerRank",
			cfg:  RackEPMegakernelConfig{TokensPerRank: 0, HiddenSize: 128, TopK: 2},
		},
		{
			name: "BadHiddenSize",
			cfg:  RackEPMegakernelConfig{TokensPerRank: 16, HiddenSize: 127, TopK: 2},
		},
		{
			name: "BadTopK",
			cfg:  RackEPMegakernelConfig{TokensPerRank: 16, HiddenSize: 128, TopK: 0},
		},
		{
			name: "TooFewCTAs",
			cfg:  RackEPMegakernelConfig{TokensPerRank: 16, HiddenSize: 128, TopK: 2, TotalCTAs: 2},
		},
		{
			name: "TooFewMicrobatches",
			cfg:  RackEPMegakernelConfig{TokensPerRank: 16, HiddenSize: 128, TopK: 2, Microbatches: 1},
		},
	}

	for _, tc := range badConfigs {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := CompileRackEPMegakernel(tc.cfg); err == nil {
				t.Fatalf("CompileRackEPMegakernel(%s) succeeded, want error", tc.name)
			}
		})
	}
}
