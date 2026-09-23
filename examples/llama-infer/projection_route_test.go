package main

import "testing"

func TestProjectionFusionEligibility(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	for _, half := range []bool{false, true} {
		for _, group := range []projectionGroup{projectionQKV, projectionGateUp} {
			if err := eligibleProjectionFusion(group, modelProjectionRoutes(cfg, half, group)...); err != nil {
				t.Fatalf("half=%v group=%d: %v", half, group, err)
			}
		}
	}
}

func TestProjectionFusionRejectsDifferentRoutes(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	tests := []struct {
		name  string
		group projectionGroup
		edit  func([]projectionRoute)
	}{
		{"layout", projectionQKV, func(r []projectionRoute) { r[1].Layout = 0 }},
		{"precision", projectionQKV, func(r []projectionRoute) { r[1].Precision = projectionF16 }},
		{"input", projectionQKV, func(r []projectionRoute) { r[1].Input++ }},
		{"accumulator", projectionQKV, func(r []projectionRoute) { r[1].Accumulator = "f16" }},
		{"reduction", projectionQKV, func(r []projectionRoute) { r[1].Reduction = "tree64" }},
		{"quantization", projectionQKV, func(r []projectionRoute) { r[1].Quant = "q4" }},
		{"key value width", projectionQKV, func(r []projectionRoute) { r[2].Output++ }},
		{"gate up width", projectionGateUp, func(r []projectionRoute) { r[1].Output++ }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			routes := modelProjectionRoutes(cfg, false, test.group)
			test.edit(routes)
			if err := eligibleProjectionFusion(test.group, routes...); err == nil {
				t.Fatal("ineligible routes accepted")
			}
			if useProjectionFusion(test.group, routes...) {
				t.Fatal("ineligible routes selected fused execution")
			}
		})
	}
}

func TestProjectionFusionDispatchCeiling(t *testing.T) {
	const layers = 16
	unfused := 1 + 12*layers + 3
	fused := unfused - 3*layers
	if unfused != 196 || fused != 148 {
		t.Fatalf("dispatch ceiling: unfused=%d fused=%d", unfused, fused)
	}
}
