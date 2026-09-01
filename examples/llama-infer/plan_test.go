package main

import (
	"testing"
)

func TestExecutionPlanGeneration(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.FuseEGraph = false

	planUnfused := BuildExecutionPlan(cfg)
	if planUnfused.DType != "f32" {
		t.Fatalf("expected DType f32, got %s", planUnfused.DType)
	}
	if planUnfused.FuseEGraph {
		t.Fatal("expected FuseEGraph to be false")
	}
	if len(planUnfused.Buffers) == 0 {
		t.Fatal("expected non-empty buffers in execution plan")
	}

	cfg.FuseEGraph = true
	planFused := BuildExecutionPlan(cfg)
	if !planFused.FuseEGraph {
		t.Fatal("expected FuseEGraph to be true")
	}

	// Verify fused plan has fewer total op dispatches than unfused
	if len(planFused.Ops) >= len(planUnfused.Ops) {
		t.Fatalf("expected fused plan ops (%d) < unfused plan ops (%d)", len(planFused.Ops), len(planUnfused.Ops))
	}

	hasFusedQKV := false
	for _, op := range planFused.Ops {
		if op.Name == "FusedRMSNormQKV" && op.Fused {
			hasFusedQKV = true
			break
		}
	}
	if !hasFusedQKV {
		t.Fatal("fused execution plan missing FusedRMSNormQKV op")
	}
}
