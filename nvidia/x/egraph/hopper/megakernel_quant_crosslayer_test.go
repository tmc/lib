package hopper

import (
	"strings"
	"testing"
)

func TestQuantCrossLayerMegakernelFormats(t *testing.T) {
	formats := []struct {
		format       QuantFormat
		groupSize    int
		hiddenDim    int
		intermDim    int
		wantHelper   string
		minCompress  float64
	}{
		{
			format:      FormatNVFP4,
			groupSize:   16,
			hiddenDim:   1024,
			intermDim:   2048,
			wantHelper:  "eg_quant_gemm_nvfp4_dev",
			minCompress: 3.2,
		},
		{
			format:      FormatFP8E4M3,
			groupSize:   128,
			hiddenDim:   512,
			intermDim:   1024,
			wantHelper:  "eg_quant_gemm_fp8_e4m3_dev",
			minCompress: 1.8,
		},
		{
			format:      FormatAWQ4,
			groupSize:   128,
			hiddenDim:   512,
			intermDim:   1024,
			wantHelper:  "eg_quant_gemm_awq4_dev",
			minCompress: 3.2,
		},
		{
			format:      FormatMXFP4,
			groupSize:   32,
			hiddenDim:   1024,
			intermDim:   2048,
			wantHelper:  "eg_quant_gemm_mxfp4_dev",
			minCompress: 3.2,
		},
		{
			format:      FormatGPTQ4,
			groupSize:   128,
			hiddenDim:   512,
			intermDim:   1024,
			wantHelper:  "eg_quant_gemm_gptq4_dev",
			minCompress: 3.2,
		},
	}

	for _, tc := range formats {
		t.Run(tc.format.helperName(), func(t *testing.T) {
			cfg := QuantCrossLayerConfig{
				NamePrefix: "qcross_" + tc.format.helperName(),
				Format:     tc.format,
				GroupSize:  tc.groupSize,
				BatchM:     1,
				HiddenDim:  tc.hiddenDim,
				IntermDim:  tc.intermDim,
				NumQHeads:  8,
				NumKVHeads: 2,
				HeadDim:    tc.hiddenDim / 8,
				NumLayers:  2,
				Arch:       ArchSM121A,
			}

			layerPlans, err := BuildQuantizedLayerPlans(cfg)
			if err != nil {
				t.Fatalf("BuildQuantizedLayerPlans(%s): %v", tc.format, err)
			}
			if layerPlans == nil || layerPlans.AttnIn == nil || layerPlans.Post == nil {
				t.Fatalf("BuildQuantizedLayerPlans(%s) returned nil plan components", tc.format)
			}
			if len(layerPlans.AttnIn.Kernels) != 1 {
				t.Fatalf("len(AttnIn.Kernels) = %d, want 1", len(layerPlans.AttnIn.Kernels))
			}
			if len(layerPlans.Post.Kernels) != 6 {
				t.Fatalf("len(Post.Kernels) = %d, want 6 (o_proj, residual_rmsnorm, gate_proj, up_swiglu, down_proj, post_residual_rmsnorm)",
					len(layerPlans.Post.Kernels))
			}

			stack, err := BuildQuantizedCrossLayerMegakernel(cfg)
			if err != nil {
				t.Fatalf("BuildQuantizedCrossLayerMegakernel(%s): %v", tc.format, err)
			}
			if len(stack.CrossMegakernels) != 1 {
				t.Fatalf("len(CrossMegakernels) = %d, want 1 for 2-layer stack", len(stack.CrossMegakernels))
			}
			mega := stack.CrossMegakernels[0]
			if len(mega.Stages) != 7 {
				t.Fatalf("len(mega.Stages) = %d, want 7 (6 post stages from layer 0 + 1 QKV stage from layer 1)", len(mega.Stages))
			}

			cudaSrc, err := mega.CUDA()
			if err != nil {
				t.Fatalf("mega.CUDA(%s): %v", tc.format, err)
			}
			if !strings.Contains(cudaSrc, tc.wantHelper) {
				t.Fatalf("CUDA source for %s missing register-dequant helper %q", tc.format, tc.wantHelper)
			}
			for _, wantStageFn := range []string{
				"eg_quant_gemm_l0_o_proj_dev",
				"eg_quant_gemm_l0_gate_proj_dev",
				"eg_quant_gemm_l0_up_swiglu_dev",
				"eg_quant_gemm_l0_down_proj_dev",
				"eg_quant_gemm_l1_qkv_proj_dev",
				"l0_o_proj_dev(",
				"l0_residual_rmsnorm_dev(",
				"l0_gate_proj_dev(",
				"l0_up_swiglu_dev(",
				"l0_down_proj_dev(",
				"l0_post_residual_rmsnorm_dev(",
				"l1_qkv_proj_dev(",
			} {
				if !strings.Contains(cudaSrc, wantStageFn) {
					t.Errorf("CUDA source for %s missing %q", tc.format, wantStageFn)
				}
			}

			fp16Bytes, quantBytes, ratio := EstimateQuantizedMegakernelBandwidth(cfg)
			if fp16Bytes <= 0 || quantBytes <= 0 {
				t.Fatalf("invalid bandwidth estimate: fp16=%d quant=%d", fp16Bytes, quantBytes)
			}
			if ratio < tc.minCompress {
				t.Fatalf("compressionRatio for %s = %.3fx (fp16=%d, quant=%d), want >= %.2fx",
					tc.format, ratio, fp16Bytes, quantBytes, tc.minCompress)
			}
		})
	}
}

func TestQuantCrossLayerMultiLayerAndValidation(t *testing.T) {
	cfg := QuantCrossLayerConfig{
		NamePrefix: "qstack4",
		Format:     FormatNVFP4,
		GroupSize:  16,
		BatchM:     2,
		HiddenDim:  1024,
		IntermDim:  2048,
		NumQHeads:  8,
		NumKVHeads: 2,
		HeadDim:    128,
		NumLayers:  4,
		Arch:       ArchSM100A,
	}
	stack, err := BuildQuantizedCrossLayerMegakernel(cfg)
	if err != nil {
		t.Fatalf("BuildQuantizedCrossLayerMegakernel 4-layer: %v", err)
	}
	if got, want := stack.UnfusedLaunchCount(), 9; got != want {
		t.Fatalf("UnfusedLaunchCount() = %d, want %d", got, want)
	}
	if got, want := stack.FusedLaunchCount(), 5; got != want {
		t.Fatalf("FusedLaunchCount() = %d, want %d", got, want)
	}
	if len(stack.CrossMegakernels) != 3 {
		t.Fatalf("len(CrossMegakernels) = %d, want 3", len(stack.CrossMegakernels))
	}
	if stack.FinalMegakernel == nil {
		t.Fatal("expected non-nil FinalMegakernel")
	}
	if stack.ArenaBytes == 0 {
		t.Fatal("expected non-zero ArenaBytes from shared temp packing")
	}

	// Verify invalid configurations return errors.
	if _, err := BuildQuantizedLayerPlans(QuantCrossLayerConfig{Format: QuantFormat(99)}); err == nil {
		t.Error("expected error for invalid QuantFormat")
	}
	if _, err := BuildQuantizedCrossLayerMegakernel(QuantCrossLayerConfig{NamePrefix: "eg_bad"}); err == nil {
		t.Error("expected error for reserved eg_ prefix")
	}
	if _, err := BuildQuantizedCrossLayerMegakernel(QuantCrossLayerConfig{NumQHeads: 5, NumKVHeads: 2}); err == nil {
		t.Error("expected error when NumQHeads is not divisible by NumKVHeads")
	}
}
