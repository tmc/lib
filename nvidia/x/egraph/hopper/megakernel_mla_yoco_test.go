package hopper

import (
	"slices"
	"strings"
	"testing"
)

func TestMLAYoCoStackMegakernel(t *testing.T) {
	cfg := MLAYoCoStackConfig{
		NamePrefix:    "deepseek_yoco",
		BatchM:        1,
		HiddenDim:     512,
		IntermDim:     1024,
		NumHeads:      16,
		LatentKVLoRA:  512,
		DecoupledRoPE: 64,
		SlidingWindow: 256,
		LayerSchedule: []MLAYoCoLayerKind{
			LayerKindMLA,
			LayerKindSlidingWindow,
			LayerKindMLA,
			LayerKindSharedKVYoCo,
		},
		Arch: ArchSM90A,
	}

	plan, mega, err := BuildMLAYoCoStackMegakernel(cfg)
	if err != nil {
		t.Fatalf("BuildMLAYoCoStackMegakernel: %v", err)
	}
	if plan == nil || mega == nil {
		t.Fatal("expected non-nil Plan and Megakernel")
	}

	stageArgs := SharedKVStageBuffers(plan)

	// Layer 0 (MLA) and Layer 2 (MLA) must have KV compression and decoupled RoPE stages.
	for _, wantPresent := range []string{
		"l0_kv_lora_compress",
		"l0_kr_decoupled_rope",
		"l0_mla_absorbed_decode",
		"l1_swa_kv_append",
		"l1_swa_paged_attn",
		"l2_kv_lora_compress",
		"l2_kr_decoupled_rope",
		"l2_mla_absorbed_decode",
		"l3_yoco_shared_kv_mla_decode",
	} {
		if _, ok := stageArgs[wantPresent]; !ok {
			t.Errorf("expected stage %q in fused Plan, got stages %v", wantPresent, planStageNames(plan))
		}
	}

	// Layer 3 (LayerKindSharedKVYoCo) MUST omit redundant KV projection / compression stages.
	for _, wantAbsent := range []string{
		"l3_kv_lora_compress",
		"l3_kr_decoupled_rope",
		"l3_swa_kv_append",
	} {
		if _, exists := stageArgs[wantAbsent]; exists {
			t.Errorf("LayerKindSharedKVYoCo layer 3 should omit %q, but it was present", wantAbsent)
		}
	}

	// Layer 3's YoCo attention stage must bind directly to producer Layer 2's c_kv_cache and k_r_cache.
	l2Args := stageArgs["l2_mla_absorbed_decode"]
	l3Args := stageArgs["l3_yoco_shared_kv_mla_decode"]
	for _, sharedBuf := range []string{"l2_c_kv_cache", "l2_k_r_cache"} {
		if !slices.Contains(l2Args, sharedBuf) {
			t.Errorf("producer stage l2_mla_absorbed_decode args %v missing %q", l2Args, sharedBuf)
		}
		if !slices.Contains(l3Args, sharedBuf) {
			t.Errorf("YoCo consumer stage l3_yoco_shared_kv_mla_decode args %v missing producer buffer %q", l3Args, sharedBuf)
		}
	}

	// Verify Layer 1's sliding-window PagedAttention stage preserves cfg.SlidingWindow.
	foundSWA := false
	for _, k := range plan.Kernels {
		if k.Attention != nil && k.Attention.Name == "l1_swa_paged_attn" {
			foundSWA = true
			if k.Attention.Window != cfg.SlidingWindow {
				t.Errorf("l1_swa_paged_attn Window = %d, want %d", k.Attention.Window, cfg.SlidingWindow)
			}
		}
	}
	if !foundSWA {
		t.Fatal("l1_swa_paged_attn PagedAttention kernel not found in fused plan")
	}

	// Verify Megakernel CUDA generation and stage wait dependencies.
	cudaSrc, err := mega.CUDA()
	if err != nil {
		t.Fatalf("mega.CUDA(): %v", err)
	}
	for _, snippet := range []string{
		"eg_mla_absorbed_decode_l0_mla_absorbed_decode_dev",
		"eg_mla_absorbed_decode_l2_mla_absorbed_decode_dev",
		"eg_mla_absorbed_decode_l3_yoco_shared_kv_mla_decode_dev",
		"l0_kv_lora_compress_dev(",
		"l1_swa_paged_attn_dev(",
		"l2_kv_lora_compress_dev(",
		"l3_yoco_shared_kv_mla_decode_dev(",
	} {
		if !strings.Contains(cudaSrc, snippet) {
			t.Errorf("Megakernel CUDA source missing %q", snippet)
		}
	}

	// Verify KV-cache byte reduction per token.
	stdBytes, mlaYoCoBytes, reduction := EstimateMLAYoCoKVBytesPerToken(cfg)
	if stdBytes <= 0 || mlaYoCoBytes <= 0 {
		t.Fatalf("invalid KV byte estimates: std=%d mlaYoCo=%d", stdBytes, mlaYoCoBytes)
	}
	if reduction < 8.0 {
		t.Fatalf("KV byte reductionFactor = %.2fx (std=%d, mlaYoCo=%d), want >= 8.0x",
			reduction, stdBytes, mlaYoCoBytes)
	}
}

func TestMLAYoCoSlidingWindowProducerAndValidation(t *testing.T) {
	// Test YoCo sharing from a SlidingWindow producer layer.
	cfg := MLAYoCoStackConfig{
		NamePrefix:    "swa_yoco",
		BatchM:        1,
		HiddenDim:     256,
		IntermDim:     512,
		NumHeads:      4,
		LatentKVLoRA:  128,
		DecoupledRoPE: 64,
		SlidingWindow: 128,
		LayerSchedule: []MLAYoCoLayerKind{
			LayerKindSlidingWindow,
			LayerKindSharedKVYoCo,
		},
		Arch: ArchSM121A,
	}
	plan, mega, err := BuildMLAYoCoStackMegakernel(cfg)
	if err != nil {
		t.Fatalf("BuildMLAYoCoStackMegakernel(SWA->YoCo): %v", err)
	}
	stageArgs := SharedKVStageBuffers(plan)
	l1Args := stageArgs["l1_yoco_shared_kv_swa_attn"]
	if !slices.Contains(l1Args, "l0_k_cache") || !slices.Contains(l1Args, "l0_v_cache") {
		t.Fatalf("l1_yoco_shared_kv_swa_attn args = %v, want l0_k_cache and l0_v_cache", l1Args)
	}
	if _, err := mega.CUDA(); err != nil {
		t.Fatalf("mega.CUDA(SWA->YoCo): %v", err)
	}

	// Verify validation errors.
	if _, _, err := BuildMLAYoCoStackMegakernel(MLAYoCoStackConfig{}); err == nil {
		t.Error("expected error for empty LayerSchedule")
	}
	if _, _, err := BuildMLAYoCoStackMegakernel(MLAYoCoStackConfig{
		LayerSchedule: []MLAYoCoLayerKind{LayerKindSharedKVYoCo},
	}); err == nil {
		t.Error("expected error when LayerKindSharedKVYoCo is first layer with no producer")
	}
	if _, _, err := BuildMLAYoCoStackMegakernel(MLAYoCoStackConfig{
		LayerSchedule: []MLAYoCoLayerKind{"unknown_kind"},
	}); err == nil {
		t.Error("expected error for unknown MLAYoCoLayerKind")
	}
}

func planStageNames(p *Plan) []string {
	var names []string
	for _, k := range p.Kernels {
		names = append(names, planKernelName(k))
	}
	return names
}
