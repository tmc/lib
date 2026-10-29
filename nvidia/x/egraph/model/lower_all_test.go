package model

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

func TestCompileProductionModelAllTargets(t *testing.T) {
	cfg := ArchConfig{
		HiddenSize:        512,
		IntermediateSize:  1024,
		NumHiddenLayers:   4,
		NumAttentionHeads: 8,
		NumKeyValueHeads:  2,
		HeadDim:           64,
		RMSNormEps:        1e-5,
		WeightDType:       NVFP4,
		KVDType:           E4M3,
	}

	tests := []struct {
		name               string
		opts               ProductionCompileOptions
		wantTMEM           bool
		wantIdiomSub       string
		wantMegaSub        string
		wantGemmSubs       []string
		wantAttnSubs       []string
		wantQuantSubs      []string
		wantCollectiveSubs []string
	}{
		{
			name: "sm_90a_hopper",
			opts: ProductionCompileOptions{
				Target:  "sm_90a",
				BatchM:  16,
				SeqLen:  2048,
				TPRanks: 8,
			},
			wantTMEM:     false,
			wantIdiomSub: "prmt.b32",
			wantMegaSub:  "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16",
			wantGemmSubs: []string{"hopper_wgmma_tile_gemm", "wgmma.mma_async"},
			wantAttnSubs: []string{"hopper_paged_attention", "cvt.rn.f16x2.e4m3x2"},
			wantQuantSubs: []string{
				"rmsnorm_dyn_nvfp4_quant",
				"eg_f32_to_ue4m3",
			},
			wantCollectiveSubs: []string{
				"allreduce_residual_rmsnorm",
				"multimem.ld_reduce",
			},
		},
		{
			name: "sm_100a_blackwell_dc",
			opts: ProductionCompileOptions{
				Target:          "sm_100a",
				BatchM:          16,
				SeqLen:          2048,
				UseMLA:          true,
				AllowSparse2to4: true,
				TPRanks:         8,
			},
			wantTMEM:     true,
			wantIdiomSub: "cvt.rn.f16x2.e2m1x2",
			wantMegaSub:  "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X",
			wantGemmSubs: []string{
				"sm100_tcgen05_sparse_gemm",
				"tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X",
			},
			wantAttnSubs: []string{
				"sm100_tcgen05_mla_paged_attention",
				"tcgen05.mma.cta_group::1.kind::f16",
			},
			wantQuantSubs: []string{
				"rmsnorm_dyn_nvfp4_quant",
				"tcgen05_swizzled",
			},
			wantCollectiveSubs: []string{
				"allreduce_residual_rmsnorm",
				"multimem.ld_reduce",
			},
		},
		{
			name: "sm_103a_gb300",
			opts: ProductionCompileOptions{
				Target:          "sm_103a",
				BatchM:          16,
				SeqLen:          2048,
				UseMLA:          false,
				AllowSparse2to4: true,
				TPRanks:         72,
			},
			wantTMEM:     true,
			wantIdiomSub: "cvt.rn.f16x2.e2m1x2",
			wantMegaSub:  "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
			wantGemmSubs: []string{
				"sm103_tcgen05_scale_vec_4x_gemm",
				"scale_vec::4X",
				"sm103_tcgen05_sparse_gemm",
			},
			wantAttnSubs: []string{
				"sm103_tcgen05_paged_attention",
				"scale_vec::4X",
			},
			wantQuantSubs: []string{
				"tcgen05.ld.red.sync.aligned.32x32b.x4.max.abs.f32",
				"rmsnorm_dyn_nvfp4_quant",
			},
			wantCollectiveSubs: []string{
				"allreduce_residual_rmsnorm",
				"multimem.ld_reduce",
			},
		},
		{
			name: "sm_107a_vera_rubin",
			opts: ProductionCompileOptions{
				Target:          "sm_107a",
				BatchM:          16,
				SeqLen:          2048,
				UseMLA:          true,
				AllowSparse2to4: true,
				AllowSub4BitLUT: true,
				TPRanks:         144,
			},
			wantTMEM:     true,
			wantIdiomSub: "fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4",
			wantMegaSub:  "decompress::lut::b.block32.sync_restrict::shared::read::mma::a",
			wantGemmSubs: []string{
				"rubin_tcgen05_lut_decompress_gemm",
				"decompress::lut::b.block32",
				"rubin_tcgen05_sparse_gemm",
				"rubin_tcgen05_early_release_gemm",
				"sync_restrict::shared::read::mma::a",
			},
			wantAttnSubs: []string{
				"rubin_tcgen05_mla_paged_attention",
			},
			wantQuantSubs: []string{
				"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
				"cvt.rn.satfinite.scaled::n1::ue8m0.ue5m3x2.f32",
				"fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4",
			},
			wantCollectiveSubs: []string{
				"rubin_nvl144_allreduce_residual_rmsnorm",
				"fabric.try_red.async.tensor.2d.multimem",
			},
		},
		{
			name: "sm_107f_vera_rubin_fabric",
			opts: ProductionCompileOptions{
				Target:          "sm_107f",
				BatchM:          16,
				SeqLen:          2048,
				UseMLA:          false,
				AllowSparse2to4: true,
				AllowSub4BitLUT: true,
				TPRanks:         576,
				EPRanks:         576,
			},
			wantTMEM:     true,
			wantIdiomSub: "fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4",
			wantMegaSub:  "fabric.try_red.async.tensor.2d.multimem",
			wantGemmSubs: []string{
				"rubin_tcgen05_lut_decompress_gemm",
				"rubin_tcgen05_early_release_gemm",
			},
			wantAttnSubs: []string{
				"rubin_tcgen05_paged_attention",
			},
			wantQuantSubs: []string{
				"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
				"rubin_ue5m3_scaled_quant",
				"rubin_quad_subbyte_simd_epilogue",
			},
			wantCollectiveSubs: []string{
				"rubin_nvl144_allreduce_residual_rmsnorm",
				"fabric.try_red.async.tensor.2d.multimem",
			},
		},
		{
			name: "sm_120a_blackwell_workstation",
			opts: ProductionCompileOptions{
				Target: "sm_120a",
				BatchM: 16,
				SeqLen: 2048,
			},
			wantTMEM:     false,
			wantIdiomSub: "cvt.rn.f16x2.e2m1x2",
			wantMegaSub:  "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			wantGemmSubs: []string{
				"sm120_block_scaled_gemm",
				"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
				"cvt.rn.f16x2.e2m1x2",
			},
			wantAttnSubs: []string{
				"sm120_paged_attention",
				"eg_sm120_dequant_nvfp4_pair",
			},
			wantQuantSubs: []string{
				"rmsnorm_dyn_nvfp4_quant",
			},
			wantCollectiveSubs: []string{
				"allreduce_residual_rmsnorm",
			},
		},
		{
			name: "sm_121a_dgx_spark_gb10",
			opts: ProductionCompileOptions{
				Target: "sm_121a",
				BatchM: 16,
				SeqLen: 2048,
			},
			wantTMEM:     false,
			wantIdiomSub: "prmt.b32",
			wantMegaSub:  "prmt.b32 (eg_e2m1x4_to_bf16x4)",
			wantGemmSubs: []string{
				"sm120_block_scaled_gemm",
				"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
				"eg_e2m1x4_to_bf16x4",
			},
			wantAttnSubs: []string{
				"sm120_paged_attention",
			},
			wantQuantSubs: []string{
				"rmsnorm_dyn_nvfp4_quant",
			},
			wantCollectiveSubs: []string{
				"allreduce_residual_rmsnorm",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := CompileProductionModel(cfg, tt.opts)
			if err != nil {
				t.Fatalf("CompileProductionModel(%s): %v", tt.opts.Target, err)
			}
			if plan.Target != tt.opts.Target {
				t.Errorf("Target = %q, want %q", plan.Target, tt.opts.Target)
			}
			if plan.Twin.Target != tt.opts.Target {
				t.Errorf("Twin.Target = %q, want %q", plan.Twin.Target, tt.opts.Target)
			}
			if !plan.SubByteIdiom.VerifiedExhaustive || !plan.SubByteIdiom.MutantRejected {
				t.Errorf("SubByteIdiom not verified/mutant-rejected: %+v", plan.SubByteIdiom)
			}
			joinedPTX := strings.Join(plan.SubByteIdiom.PTXSequence, "\n")
			if !strings.Contains(joinedPTX, tt.wantIdiomSub) {
				t.Errorf("SubByteIdiom PTX = %q, want substring %q", joinedPTX, tt.wantIdiomSub)
			}
			if plan.MacroSchedule.Target != tt.opts.Target || plan.MacroSchedule.TileM <= 0 {
				t.Errorf("invalid MacroSchedule: %+v", plan.MacroSchedule)
			}
			if !strings.Contains(plan.MegakernelCUDA, tt.wantMegaSub) {
				t.Errorf("MegakernelCUDA missing %q:\n%s", tt.wantMegaSub, plan.MegakernelCUDA)
			}
			if !strings.Contains(plan.MegakernelCUDA, "eg_multiarch_megakernel") {
				t.Errorf("MegakernelCUDA missing eg_multiarch_megakernel")
			}

			for _, role := range []string{"gemm", "attention", "quant_epilogue", "collective"} {
				if strings.TrimSpace(plan.SpecializedKernels[role]) == "" {
					t.Fatalf("SpecializedKernels[%q] is empty for %s", role, tt.opts.Target)
				}
			}
			for _, sub := range tt.wantGemmSubs {
				if !strings.Contains(plan.SpecializedKernels["gemm"], sub) {
					t.Errorf("SpecializedKernels[gemm] for %s missing %q", tt.opts.Target, sub)
				}
			}
			for _, sub := range tt.wantAttnSubs {
				if !strings.Contains(plan.SpecializedKernels["attention"], sub) {
					t.Errorf("SpecializedKernels[attention] for %s missing %q", tt.opts.Target, sub)
				}
			}
			for _, sub := range tt.wantQuantSubs {
				if !strings.Contains(plan.SpecializedKernels["quant_epilogue"], sub) {
					t.Errorf("SpecializedKernels[quant_epilogue] for %s missing %q", tt.opts.Target, sub)
				}
			}
			for _, sub := range tt.wantCollectiveSubs {
				if !strings.Contains(plan.SpecializedKernels["collective"], sub) {
					t.Errorf("SpecializedKernels[collective] for %s missing %q", tt.opts.Target, sub)
				}
			}

			if tt.wantTMEM {
				if len(plan.TMEMTraces) == 0 {
					t.Fatalf("expected non-empty TMEMTraces on %s", tt.opts.Target)
				}
				arch := cudatmem.SM100A
				if tt.opts.Target == "sm_103a" || tt.opts.Target == "sm_107a" || tt.opts.Target == "sm_107f" {
					arch = cudatmem.SM103A
				}
				for i, tr := range plan.TMEMTraces {
					if _, err := cudatmem.VerifyTrace(arch, tr); err != nil {
						t.Errorf("VerifyTrace[%d] on %s: %v", i, tt.opts.Target, err)
					}
				}
			} else if len(plan.TMEMTraces) != 0 {
				t.Errorf("expected 0 TMEMTraces on %s, got %d", tt.opts.Target, len(plan.TMEMTraces))
			}

			if plan.Fatbin == nil || len(plan.Fatbin.Units) < 6 {
				t.Fatalf("Fatbin has %v units, want >= 6", plan.Fatbin)
			}
			if !strings.Contains(plan.Fatbin.DispatcherCUDA, "dispatch_production_decoder_layer_by_device") {
				t.Errorf("Fatbin.DispatcherCUDA missing dispatcher entry")
			}
		})
	}

	t.Run("online_telemetry_adaptation", func(t *testing.T) {
		basePlan, err := CompileProductionModel(cfg, ProductionCompileOptions{
			Target: "sm_107a",
			BatchM: 16,
			SeqLen: 2048,
		})
		if err != nil {
			t.Fatalf("CompileProductionModel baseline: %v", err)
		}

		// Memory latency stall telemetry promotes deeper staging and LUT decompression.
		memStallPlan, err := CompileProductionModel(cfg, ProductionCompileOptions{
			Target: "sm_107a",
			BatchM: 16,
			SeqLen: 2048,
			Stalls: &autotune.StallProfile{
				TotalSamples:        1000,
				StallLongScoreboard: 700,
			},
			Metrics: &autotune.RangeMetrics{
				DurationNS:    25000,
				RegisterCount: 168,
			},
		})
		if err != nil {
			t.Fatalf("CompileProductionModel mem stall: %v", err)
		}
		if memStallPlan.Recommendation.Bottleneck != autotune.BottleneckMemoryLatency {
			t.Errorf("Bottleneck = %q, want %q", memStallPlan.Recommendation.Bottleneck, autotune.BottleneckMemoryLatency)
		}
		if memStallPlan.MegakernelPlan.PipelineStages <= basePlan.MegakernelPlan.PipelineStages {
			t.Errorf("MegakernelPlan.PipelineStages = %d, want > %d",
				memStallPlan.MegakernelPlan.PipelineStages, basePlan.MegakernelPlan.PipelineStages)
		}

		// Register spill telemetry rebalances warp-specialized register budgets.
		spillPlan, err := CompileProductionModel(cfg, ProductionCompileOptions{
			Target: "sm_100a",
			BatchM: 16,
			SeqLen: 2048,
			Stalls: &autotune.StallProfile{
				TotalSamples:         1000,
				StallShortScoreboard: 500,
			},
			Metrics: &autotune.RangeMetrics{
				DurationNS:          40000,
				RegisterCount:       255,
				LocalSpillLoadBytes: 256,
			},
		})
		if err != nil {
			t.Fatalf("CompileProductionModel spill: %v", err)
		}
		if spillPlan.Recommendation.Bottleneck != autotune.BottleneckRegisterSpill {
			t.Errorf("Bottleneck = %q, want %q", spillPlan.Recommendation.Bottleneck, autotune.BottleneckRegisterSpill)
		}
		if spillPlan.MegakernelPlan.ConsumerMaxRegs >= 232 {
			t.Errorf("ConsumerMaxRegs = %d, want < 232 on register spill", spillPlan.MegakernelPlan.ConsumerMaxRegs)
		}
	})
}

func ExampleCompileProductionModel() {
	cfg := ArchConfig{
		HiddenSize:        512,
		IntermediateSize:  1024,
		NumAttentionHeads: 8,
		NumKeyValueHeads:  2,
		HeadDim:           64,
		WeightDType:       NVFP4,
		KVDType:           E4M3,
	}
	plan, err := CompileProductionModel(cfg, ProductionCompileOptions{
		Target:          "sm_107f",
		BatchM:          16,
		SeqLen:          2048,
		UseMLA:          true,
		AllowSparse2to4: true,
		AllowSub4BitLUT: true,
		TPRanks:         144,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(
		plan.Target,
		plan.Twin.PTXISAVersion,
		plan.SubByteIdiom.Name,
		plan.MegakernelPlan.TMEMColumns,
		len(plan.TMEMTraces) >= 2,
		len(plan.Fatbin.Units) >= 6,
	)
	// Output:
	// sm_107f 9.4 ptx94_quad_fma_e2m1p4x4 576 true true
}
