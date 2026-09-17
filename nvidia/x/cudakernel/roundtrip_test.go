package cudakernel_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestArtifactRoundTrip(t *testing.T) {
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "output", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "input", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "scale", Type: "f32"},
		cudakernel.Param{Name: "n", Type: "u32"},
	)
	if err != nil {
		t.Fatalf("NewSchema: %v", err)
	}

	ptxCode := []byte(".version 8.0\n.target sm_121a\n.address_size 64\n.entry scale(.param .u64 o, .param .u64 i, .param .f32 s, .param .u32 n) { ret; }\n")

	orig, err := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		ptxCode,
		"scale",
		schema,
		"sm_121a",
		cudakernel.WithResources(cudakernel.ResourceRequirements{
			StaticSharedMemory:  2048,
			DynamicSharedMemory: 4096,
			AllocatedRegisters:  32,
			PeakLiveGPR:         18, // Distinct from allocated!
			RegisterQuantum:     256,
			ExecutionGroup:      "cta",
		}),
		cudakernel.WithJIT(cudakernel.JITOptions{
			MaxRegisters:    32,
			ThreadsPerBlock: 256,
			OptLevel:        4,
		}),
		cudakernel.WithToolchain("nvrtc 13.0"),
		cudakernel.WithCompilerOptions([]string{"--std=c++20", "--gpu-architecture=sm_121a"}),
		cudakernel.WithDependencies([]cudakernel.Dependency{
			{Path: "helper.h", Digest: "abcdef0123456789", Tracked: true},
		}),
	)
	if err != nil {
		t.Fatalf("NewArtifact: %v", err)
	}

	// Marshal to JSON
	data, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("MarshalJSON: %v", err)
	}

	// Unmarshal back
	var roundTripped cudakernel.Artifact
	if err := json.Unmarshal(data, &roundTripped); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}

	// Verify all fields
	if roundTripped.Kind != orig.Kind {
		t.Errorf("Kind mismatch: got %v, want %v", roundTripped.Kind, orig.Kind)
	}
	if roundTripped.Digest != orig.Digest {
		t.Errorf("Digest mismatch: got %s, want %s", roundTripped.Digest, orig.Digest)
	}
	if string(roundTripped.Raw) != string(orig.Raw) {
		t.Errorf("Raw bytes mismatch")
	}
	if roundTripped.EntryPoint != orig.EntryPoint {
		t.Errorf("EntryPoint mismatch: got %s, want %s", roundTripped.EntryPoint, orig.EntryPoint)
	}
	if roundTripped.Target != orig.Target {
		t.Errorf("Target mismatch: got %s, want %s", roundTripped.Target, orig.Target)
	}
	if roundTripped.Resources.AllocatedRegisters != 32 {
		t.Errorf("AllocatedRegisters mismatch: got %d, want 32", roundTripped.Resources.AllocatedRegisters)
	}
	if roundTripped.Resources.PeakLiveGPR != 18 {
		t.Errorf("PeakLiveGPR mismatch: got %d, want 18", roundTripped.Resources.PeakLiveGPR)
	}
	if roundTripped.JITOptions.MaxRegisters != 32 || roundTripped.JITOptions.OptLevel != 4 {
		t.Errorf("JITOptions mismatch: got %+v", roundTripped.JITOptions)
	}
	if len(roundTripped.Dependencies) != 1 || roundTripped.Dependencies[0].Path != "helper.h" {
		t.Errorf("Dependencies mismatch: got %+v", roundTripped.Dependencies)
	}

	// Negative control: corrupted artifact bytes rejected
	corrupted := roundTripped
	corrupted.Raw = append([]byte(nil), corrupted.Raw...)
	corrupted.Raw[0] ^= 0xFF // Mutate a byte!
	if err := corrupted.Validate(); err == nil {
		t.Fatal("negative control failed: corrupted artifact bytes passed validation")
	} else if !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("expected digest mismatch error, got: %v", err)
	}

	// Negative control: wrong digest
	corruptedDigest := roundTripped
	corruptedDigest.Digest = hex.EncodeToString(sha256.New().Sum(nil))
	if err := corruptedDigest.Validate(); err == nil {
		t.Fatal("negative control failed: mismatched digest passed validation")
	}
}
