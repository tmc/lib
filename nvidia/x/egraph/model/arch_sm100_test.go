package model

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func TestExtendedArchSM100(t *testing.T) {
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

	opts := ExtendedArchOptions{
		Device:        "NVIDIA B200",
		BatchM:        256,
		SeqLen:        1024,
		QTokensPerSeq: 4,
		TPRanks:       4,
		EPRanks:       2,
		EnableCLC:     true,
	}
	extCfg, err := ValidateExtendedArchConfig(ExtendedArchConfig{Arch: cfg, Options: opts})
	if err != nil {
		t.Fatalf("ValidateExtendedArchConfig: %v", err)
	}
	if extCfg.Options.Device != "NVIDIA B200" {
		t.Errorf("Device = %q, want NVIDIA B200", extCfg.Options.Device)
	}

	kernels, traces, ctaGroup, err := EmitSM100TCGen05Suite("layer", cfg, opts)
	if err != nil {
		t.Fatalf("EmitSM100TCGen05Suite: %v", err)
	}
	if ctaGroup != 2 {
		t.Errorf("ctaGroup = %d, want 2", ctaGroup)
	}
	if len(kernels) < 3 || len(traces) != len(kernels) {
		t.Fatalf("len(kernels)=%d len(traces)=%d", len(kernels), len(traces))
	}
	for i, tr := range traces {
		if _, err := cudatmem.VerifyTrace(cudatmem.SM100A, tr); err != nil {
			t.Fatalf("VerifyTrace[%d]: %v", i, err)
		}
	}

	clcKernels, err := EmitLayerCLCKernels("layer", []CLCTileSpec{
		{Name: "qkv", N: 512, K: 512, Weight: WeightNVFP4},
	}, cfg)
	if err != nil {
		t.Fatalf("EmitLayerCLCKernels: %v", err)
	}
	if len(clcKernels) == 0 || !strings.Contains(clcKernels[0], "clusterlaunchcontrol.try_cancel") {
		t.Errorf("unexpected CLCKernels: %v", clcKernels)
	}

	colSrc, err := EmitLayerCollectives("// prelude\n", cfg, 4, 2)
	if err != nil {
		t.Fatalf("EmitLayerCollectives: %v", err)
	}
	for _, want := range []string{
		"allreduce_residual_rmsnorm",
		"nvlink_moe_combine",
		"multimem.ld_reduce",
	} {
		if !strings.Contains(colSrc, want) {
			t.Errorf("collective source missing %q", want)
		}
	}
}

func ExampleEmitSM100TCGen05Suite() {
	cfg := ArchConfig{
		HiddenSize:        512,
		IntermediateSize:  1024,
		NumAttentionHeads: 8,
		NumKeyValueHeads:  2,
		HeadDim:           64,
		WeightDType:       NVFP4,
		KVDType:           E4M3,
	}
	kernels, traces, ctaGroup, err := EmitSM100TCGen05Suite("layer", cfg, ExtendedArchOptions{
		Device: "NVIDIA B200",
		BatchM: 256,
		SeqLen: 1024,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(ctaGroup, len(kernels) == len(traces), len(kernels) >= 3)
	// Output:
	// 2 true true
}
