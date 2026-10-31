package tcgen05

import (
	"strings"
	"testing"
)

func TestBuildTMEMCrossLayerMegakernel(t *testing.T) {
	tests := []struct {
		name            string
		spec            TMEMCrossLayerSpec
		wantClusterSize int
		wantNativeTMEM  bool
		wantCols        int
		wantStages      int
		wantAllocs      int
	}{
		{
			name: "sm_100a_default_2cta_512cols",
			spec: TMEMCrossLayerSpec{
				Name:         "bw_sm100a_2cta_mega",
				Target:       "sm_100a",
				NumLayers:    2,
				Tokens:       128,
				HiddenDim:    256,
				GateCols:     128,
				UpCols:       128,
				QKVCols:      256,
				WeightFormat: "nvfp4",
			},
			wantClusterSize: 2,
			wantNativeTMEM:  true,
			wantCols:        512,
			wantStages:      12,
			wantAllocs:      6,
		},
		{
			name: "sm_100a_1cta_bf16",
			spec: TMEMCrossLayerSpec{
				Name:         "bw_sm100a_1cta_mega",
				Target:       "sm_100a",
				NumLayers:    1,
				ClusterSize:  1,
				GateCols:     64,
				UpCols:       64,
				QKVCols:      128,
				WeightFormat: "bf16",
			},
			wantClusterSize: 1,
			wantNativeTMEM:  true,
			wantCols:        256,
			wantStages:      6,
			wantAllocs:      3,
		},
		{
			name: "sm_120a_1cta_workstation",
			spec: TMEMCrossLayerSpec{
				Name:      "bw_sm120a_mega",
				Target:    "sm_120a",
				NumLayers: 2,
				GateCols:  128,
				UpCols:    128,
				QKVCols:   128,
			},
			wantClusterSize: 1,
			wantNativeTMEM:  false,
			wantCols:        384,
			wantStages:      12,
			wantAllocs:      6,
		},
		{
			name: "sm_121a_1cta_dgx_spark",
			spec: TMEMCrossLayerSpec{
				Name:      "bw_sm121a_spark_mega",
				Target:    "sm_121a",
				NumLayers: 2,
				GateCols:  128,
				UpCols:    128,
				QKVCols:   256,
			},
			wantClusterSize: 1,
			wantNativeTMEM:  false,
			wantCols:        512,
			wantStages:      12,
			wantAllocs:      6,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := BuildTMEMCrossLayerMegakernel(tc.spec)
			if err != nil {
				t.Fatalf("BuildTMEMCrossLayerMegakernel(%s) failed: %v", tc.name, err)
			}
			if plan.ClusterSize != tc.wantClusterSize {
				t.Errorf("ClusterSize = %d, want %d", plan.ClusterSize, tc.wantClusterSize)
			}
			if plan.UsesNativeTMEM != tc.wantNativeTMEM {
				t.Errorf("UsesNativeTMEM = %v, want %v", plan.UsesNativeTMEM, tc.wantNativeTMEM)
			}
			if plan.TotalTMEMCols != tc.wantCols {
				t.Errorf("TotalTMEMCols = %d, want %d", plan.TotalTMEMCols, tc.wantCols)
			}
			if plan.TotalTMEMBytes != tc.wantCols*TMEMLanesPerSM*4 {
				t.Errorf("TotalTMEMBytes = %d, want %d", plan.TotalTMEMBytes, tc.wantCols*TMEMLanesPerSM*4)
			}
			if len(plan.Stages) != tc.wantStages {
				t.Errorf("len(Stages) = %d, want %d", len(plan.Stages), tc.wantStages)
			}
			if plan.AllocCount != tc.wantAllocs || plan.DeallocCount != tc.wantAllocs {
				t.Errorf("AllocCount=%d DeallocCount=%d, want both %d", plan.AllocCount, plan.DeallocCount, tc.wantAllocs)
			}
			if err := plan.VerifyLifecycle(); err != nil {
				t.Fatalf("VerifyLifecycle() failed: %v", err)
			}
		})
	}
}

func TestTMEMCrossLayerMegakernelValidationAndLifecycleErrors(t *testing.T) {
	tests := []struct {
		name    string
		spec    TMEMCrossLayerSpec
		wantSub string
	}{
		{
			name: "exceeds_512_tmem_column_budget",
			spec: TMEMCrossLayerSpec{
				Target:   "sm_100a",
				GateCols: 256,
				UpCols:   256,
				QKVCols:  128, // 256 + 256 + 128 = 640 > 512
			},
			wantSub: "TMEM column budget exceeded",
		},
		{
			name: "non_power_of_two_columns",
			spec: TMEMCrossLayerSpec{
				Target:   "sm_100a",
				GateCols: 96,
				UpCols:   128,
				QKVCols:  128,
			},
			wantSub: "powers of two",
		},
		{
			name: "invalid_cluster_size_4",
			spec: TMEMCrossLayerSpec{
				Target:      "sm_100a",
				ClusterSize: 4,
			},
			wantSub: "ClusterSize 4 must be 1 or 2",
		},
		{
			name: "sm_121a_rejects_2cta_cluster",
			spec: TMEMCrossLayerSpec{
				Target:      "sm_121a",
				ClusterSize: 2,
			},
			wantSub: "requires sm_100a/sm_103a TMEM hardware",
		},
		{
			name: "unbalanced_tmem_lifecycle_missing_dealloc",
			spec: TMEMCrossLayerSpec{
				Target:               "sm_100a",
				NumLayers:            1,
				OmitLayerExitDealloc: true,
			},
			wantSub: "unbalanced TMEM lifecycle",
		},
		{
			name: "unsupported_target",
			spec: TMEMCrossLayerSpec{
				Target: "sm_90a",
			},
			wantSub: "unsupported target",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := BuildTMEMCrossLayerMegakernel(tc.spec)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantSub)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantSub)
			}
		})
	}
}

func TestTMEMMegakernelPTXPreviewAndHopperLowering(t *testing.T) {
	plan, err := BuildTMEMCrossLayerMegakernel(TMEMCrossLayerSpec{
		Name:            "blackwell_2cta_tmem_mega",
		Target:          "sm_100a",
		NumLayers:       2,
		Tokens:          128,
		HiddenDim:       64,
		IntermediateDim: 64,
		QDim:            64,
		KVDim:           64,
		ClusterSize:     2,
		GateCols:        128,
		UpCols:          128,
		QKVCols:         256,
		WeightFormat:    "nvfp4",
	})
	if err != nil {
		t.Fatalf("BuildTMEMCrossLayerMegakernel: %v", err)
	}

	ptx := plan.PTXPreview()
	for _, want := range []string{
		".target sm_100a",
		".maxclusterrank 2",
		"tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32",
		"tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X",
		"tcgen05.ld.sync.aligned.32x32b.x4.b32",
		"tcgen05.st.sync.aligned.32x32b.x4.b32",
		"tcgen05.dealloc.cta_group::2.sync.aligned.b32",
		"tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned",
		"SwiGLU",
		"RMSNorm",
	} {
		if !strings.Contains(ptx, want) {
			t.Errorf("PTXPreview missing %q\n%s", want, ptx)
		}
	}

	mega, err := plan.ToHopperMegakernel()
	if err != nil {
		t.Fatalf("ToHopperMegakernel: %v", err)
	}
	if len(mega.Stages) != 6 {
		t.Fatalf("len(mega.Stages) = %d, want 6", len(mega.Stages))
	}
	cudaSrc, err := mega.CUDA()
	if err != nil {
		t.Fatalf("mega.CUDA(): %v", err)
	}
	if !strings.Contains(cudaSrc, "blackwell_2cta_tmem_mega(") {
		t.Errorf("lowered Hopper Megakernel CUDA missing kernel entry point:\n%s", cudaSrc)
	}
}

func TestRubinAndBlackwellTMEMCrossLayerMegakernel(t *testing.T) {
	TestBuildTMEMCrossLayerMegakernel(t)
	TestTMEMCrossLayerMegakernelValidationAndLifecycleErrors(t)
	TestTMEMMegakernelPTXPreviewAndHopperLowering(t)
}
