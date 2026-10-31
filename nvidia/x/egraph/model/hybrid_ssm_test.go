package model

import (
	"strings"
	"testing"
)

func TestHybridSSMMegakernelStack(t *testing.T) {
	cfg := HybridStackConfig{
		Name:        "nemotron_jamba_hybrid",
		Target:      "sm_121a",
		BatchM:      2,
		HiddenDim:   256,
		SSMStateDim: 256,
		QKVDim:      256,
		InterDim:    512,
		Layers: []HybridLayerKind{
			HybridLayerMamba2SSM,
			HybridLayerGQA,
			HybridLayerSwiGLUMLP,
		},
	}

	plan, mega, err := BuildHybridSSMMegakernelStack(cfg)
	if err != nil {
		t.Fatalf("BuildHybridSSMMegakernelStack: %v", err)
	}
	if plan == nil || mega == nil {
		t.Fatalf("expected non-nil plan and megakernel")
	}
	if len(mega.Stages) < 6 {
		t.Fatalf("expected at least 6 fused stages across Mamba-2 SSM + GQA + SwiGLU MLP, got %d", len(mega.Stages))
	}

	var (
		hasRowStage  bool
		hasGEMMStage bool
	)
	for _, st := range mega.Stages {
		if st.Kernel.Row != nil {
			hasRowStage = true
		}
		if st.Kernel.GEMM != nil || st.Kernel.Tile != nil || st.Kernel.MMA != nil {
			hasGEMMStage = true
		}
	}
	if !hasRowStage || !hasGEMMStage {
		t.Fatalf("expected both Row (selective scan / RMSNorm) and GEMM stages in hybrid megakernel, got hasRow=%v hasGEMM=%v", hasRowStage, hasGEMMStage)
	}

	src, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("mega.LineInfoSource: %v", err)
	}
	if !strings.Contains(src, mega.Name) {
		t.Fatalf("emitted CUDA source missing megakernel entry %s", mega.Name)
	}
	if len(spans) != len(mega.Stages) {
		t.Fatalf("expected %d #line stage spans, got %d", len(mega.Stages), len(spans))
	}
}
