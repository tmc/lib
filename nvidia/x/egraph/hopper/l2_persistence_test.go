package hopper

import (
	"strings"
	"testing"
)

func TestL2PersistenceStackClassification(t *testing.T) {
	norm, layers, head := buildTestStackPlans(t, 4, 1, 256, 256, 64, 256, 256)
	// Add a shared hot MoE expert and KV prefix cache buffer to Layer 0's AttnIn inputs
	// to verify hot MoE/KV prefix classification alongside cold layer weights.
	layers[0].AttnIn.Inputs = append(layers[0].AttnIn.Inputs,
		Tensor{Name: "kv_prefix_pages", DType: BF16, Shape: []int{16, 64}},
		Tensor{Name: "shared_expert_w", DType: BF16, Shape: []int{128, 256}},
	)

	stack, err := FuseDecoderLayerStack(layers, CrossLayerStackConfig{
		NamePrefix:         "llama_l2",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
		Norm:               norm,
		Head:               head,
		FuseFinalHead:      true,
		PackArena:          true,
		Mega:               true,
	})
	if err != nil {
		t.Fatalf("FuseDecoderLayerStack: %v", err)
	}

	const l2Cap = 50 * 1024 * 1024 // 50 MiB Hopper H100 L2 cache
	lp, err := PlanL2Persistence(stack, l2Cap)
	if err != nil {
		t.Fatalf("PlanL2Persistence: %v", err)
	}

	if lp.L2CapacityBytes != l2Cap {
		t.Fatalf("L2CapacityBytes = %d, want %d", lp.L2CapacityBytes, l2Cap)
	}
	if lp.ArenaPingPongBytes <= 0 {
		t.Fatalf("ArenaPingPongBytes = %d, want > 0", lp.ArenaPingPongBytes)
	}
	if lp.ReadyBarrierBytes < 128 {
		t.Fatalf("ReadyBarrierBytes = %d, want >= 128", lp.ReadyBarrierBytes)
	}

	// Verify Persist-in-L2 (L2PolicyEvictLast, hitRatio=1.0) on eg_arena_even,
	// eg_arena_odd, ready, and hot MoE / KV prefix buffers.
	for _, pinnedName := range []string{
		"eg_arena_even",
		"eg_arena_odd",
		"ready",
		"kv_prefix_pages",
		"shared_expert_w",
	} {
		entry, ok := lp.ByName[pinnedName]
		if !ok {
			t.Fatalf("missing expected buffer %q in L2PersistencePlan.ByName", pinnedName)
		}
		if entry.Policy != L2PolicyEvictLast {
			t.Errorf("buffer %q Policy = %q, want %q", pinnedName, entry.Policy, L2PolicyEvictLast)
		}
		if entry.HitRatio != 1.0 {
			t.Errorf("buffer %q HitRatio = %v, want 1.0", pinnedName, entry.HitRatio)
		}
		if entry.Bytes <= 0 {
			t.Errorf("buffer %q Bytes = %d, want > 0", pinnedName, entry.Bytes)
		}
	}

	// Verify single-use cold layer weight matrices at M=1 are classified as
	// Stream-Through (L2PolicyEvictFirst) so weight streaming never thrashes L2 activations.
	for _, weightName := range []string{
		"w_qkv",
		"w_o",
		"w_gu",
		"w_down",
		"w_head",
	} {
		entry, ok := lp.ByName[weightName]
		if !ok {
			t.Fatalf("missing expected weight buffer %q in L2PersistencePlan.ByName", weightName)
		}
		if entry.Policy != L2PolicyEvictFirst {
			t.Errorf("weight buffer %q Policy = %q, want %q", weightName, entry.Policy, L2PolicyEvictFirst)
		}
		if entry.HitRatio != 0.0 {
			t.Errorf("weight buffer %q HitRatio = %v, want 0.0", weightName, entry.HitRatio)
		}
	}

	if lp.TotalPinnedBytes <= 0 {
		t.Errorf("TotalPinnedBytes = %d, want > 0", lp.TotalPinnedBytes)
	}
	if lp.TotalStreamBytes <= 0 {
		t.Errorf("TotalStreamBytes = %d, want > 0", lp.TotalStreamBytes)
	}
}

func TestL2PersistenceCUDAPolicyPreludeAndErrors(t *testing.T) {
	if _, err := PlanL2Persistence(nil, 1024*1024); err == nil {
		t.Error("nil stack: expected error")
	}
	if _, err := PlanL2Persistence(&CrossLayerStack{}, 1024*1024); err == nil {
		t.Error("empty stack: expected error")
	}

	post, nextAttnIn := buildTestCrossLayerPlans(t, 1, 256, 256, 64, 256, true, false)
	fused, err := FuseCrossLayerPlans(post, nextAttnIn, CrossLayerGlue{
		Name: "single_plan_l2",
	})
	if err != nil {
		t.Fatalf("FuseCrossLayerPlans: %v", err)
	}
	if _, err := PlanL2PersistenceForPlan(fused, 0); err == nil {
		t.Error("non-positive L2 capacity: expected error")
	}
	if _, err := PlanL2PersistenceForPlan(nil, 1024); err == nil {
		t.Error("nil plan: expected error")
	}

	lp, err := PlanL2PersistenceForPlan(fused, 32*1024*1024)
	if err != nil {
		t.Fatalf("PlanL2PersistenceForPlan: %v", err)
	}

	prelude := lp.CUDAPolicyPrelude()
	wantSubstrings := []string{
		`createpolicy.fractional.L2::evict_last.b64`,
		`createpolicy.fractional.L2::evict_first.b64`,
		`ld.global.L2::cache_hint`,
		`st.global.L2::cache_hint`,
		`eg_arena_even`,
		`eg_arena_odd`,
		`ready`,
	}
	for _, sub := range wantSubstrings {
		if !strings.Contains(prelude, sub) {
			t.Errorf("CUDAPolicyPrelude() missing expected substring %q", sub)
		}
	}

	// Verify L2PolicyStreaming and L2PolicyNormal string representations.
	if L2PolicyStreaming.String() != "streaming" || L2PolicyNormal.String() != "normal" {
		t.Errorf("unexpected policy strings: %q, %q", L2PolicyStreaming, L2PolicyNormal)
	}
}
