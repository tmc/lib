package cudakernel_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestGB10PhysicalArchBinding(t *testing.T) {
	info, err := cudakernel.TargetSM("sm_121a")
	if err != nil {
		t.Fatalf("TargetSM sm_121a: %v", err)
	}

	// DGX GB10 hardware invariants:
	if info.MaxSharedMemoryBytes != 102400 {
		t.Errorf("sm_121a MaxSharedMemoryBytes = %d, want 102400", info.MaxSharedMemoryBytes)
	}
	if info.MaxOptInSharedBytes != 101376 {
		t.Errorf("sm_121a MaxOptInSharedBytes = %d, want 101376", info.MaxOptInSharedBytes)
	}
	if info.WarpsPerSM != 48 {
		t.Errorf("sm_121a WarpsPerSM = %d, want 48", info.WarpsPerSM)
	}
	if info.MaxBlocksPerSM != 24 {
		t.Errorf("sm_121a MaxBlocksPerSM = %d, want 24", info.MaxBlocksPerSM)
	}
	if info.RegisterQuantum != 256 {
		t.Errorf("sm_121a RegisterQuantum = %d, want 256", info.RegisterQuantum)
	}

	// Architecture fencing: GB10 strictly lacks tcgen05, wgmma, and TMEM
	if info.SupportsTCGen05 {
		t.Error("sm_121a must NOT support tcgen05")
	}
	if info.SupportsWGMMA {
		t.Error("sm_121a must NOT support wgmma")
	}
	if info.SupportsTMEM {
		t.Error("sm_121a must NOT support TMEM")
	}

	// Verify that requesting resources beyond GB10 limits is rejected
	schema, _ := cudakernel.NewSchema(cudakernel.Param{Name: "x", Type: "u32"})
	ptx := []byte(".version 8.0\n.target sm_121a\n.entry k(.param .u32 x) { ret; }\n")

	// Excess shared memory beyond 102,400 B
	_, err = cudakernel.NewArtifact(
		cudakernel.KindPTX,
		ptx,
		"k",
		schema,
		"sm_121a",
		cudakernel.WithResources(cudakernel.ResourceRequirements{
			StaticSharedMemory: 102401, // Exceeds GB10 limit!
		}),
	)
	if err == nil {
		t.Fatal("negative control failed: accepted static shared memory exceeding GB10 limit")
	} else if !strings.Contains(err.Error(), "exceeds target sm_121a limit") {
		t.Fatalf("expected limit exceeded error, got: %v", err)
	}

	// Requesting TMEM on GB10
	_, err = cudakernel.NewArtifact(
		cudakernel.KindPTX,
		ptx,
		"k",
		schema,
		"sm_121a",
		cudakernel.WithResources(cudakernel.ResourceRequirements{
			TMEMBytes: 1024, // TMEM unsupported on sm_121a!
		}),
	)
	if err == nil {
		t.Fatal("negative control failed: accepted TMEM allocation on sm_121a")
	} else if !strings.Contains(err.Error(), "does not support TMEM") {
		t.Fatalf("expected TMEM unsupported error, got: %v", err)
	}
}

func TestTargetFeatureMismatch(t *testing.T) {
	// Artifact compiled for Blackwell DC (sm_100a) with tcgen05 and TMEM support
	schema, _ := cudakernel.NewSchema(cudakernel.Param{Name: "x", Type: "u32"})
	ptx := []byte(".version 8.0\n.target sm_100a\n.entry dc_kernel(.param .u32 x) { ret; }\n")

	art, err := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		ptx,
		"dc_kernel",
		schema,
		"sm_100a",
	)
	if err != nil {
		t.Fatalf("NewArtifact sm_100a: %v", err)
	}

	// Validate against same target: OK
	if err := art.ValidateTarget("sm_100a"); err != nil {
		t.Fatalf("ValidateTarget sm_100a on sm_100a: %v", err)
	}

	// Negative control: Attempt to run sm_100a artifact on GB10 (sm_121a)
	// sm_100a requires tcgen05/wgmma/TMEM which sm_121a lacks.
	err = art.ValidateTarget("sm_121a")
	if err == nil {
		t.Fatal("negative control failed: sm_100a artifact accepted on sm_121a device")
	}
	if !strings.Contains(err.Error(), "target feature mismatch") {
		t.Fatalf("expected target feature mismatch error, got: %v", err)
	}

	// Negative control: Substituted target features
	// Directly mutate artifact target to invalid arch
	mutated := *art
	mutated.Target = "invalid_arch_999"
	if err := mutated.Validate(); err == nil {
		t.Fatal("negative control failed: invalid target passed validation")
	}
}
