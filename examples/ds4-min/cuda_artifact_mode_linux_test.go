//go:build linux

package main

import "testing"

func TestLoadTimeArtifactMode(t *testing.T) {
	t.Setenv("DS4_CUDA_MOE_ARTIFACT_LAYERS", "")
	t.Setenv("DS4_CUDA_RELEASE_MAPPED_ARTIFACTS", "")
	t.Setenv("DS4_CUDA_LOAD_TIME_ARTIFACTS", "")
	t.Setenv("DS4_CUDA_MAPPED_MODEL", "")
	if got := cudaMoEArtifactLayers(); got != cudaMoEArtifactDefaultLayers {
		t.Fatalf("default MoE artifact layers = %d, want %d", got, cudaMoEArtifactDefaultLayers)
	}
	// The resident model replaces every expert tensor with its artifact.
	if got := cudaReleaseMappedArtifacts(); got != 2 {
		t.Fatalf("default mapped-artifact release mode = %d, want 2", got)
	}

	t.Setenv("DS4_CUDA_MAPPED_MODEL", "1")
	if got := cudaReleaseMappedArtifacts(); got != 0 {
		t.Fatalf("mapped-model mapped-artifact release mode = %d, want 0", got)
	}

	t.Setenv("DS4_CUDA_LOAD_TIME_ARTIFACTS", "1")
	if got := cudaMoEArtifactLayers(); got != cudaLayers {
		t.Fatalf("load-time MoE artifact layers = %d, want %d", got, cudaLayers)
	}
	if got := cudaReleaseMappedArtifacts(); got != 2 {
		t.Fatalf("load-time mapped-artifact release mode = %d, want 2", got)
	}

	// Explicit knobs remain authoritative for diagnostic arms.
	t.Setenv("DS4_CUDA_MOE_ARTIFACT_LAYERS", "2")
	t.Setenv("DS4_CUDA_RELEASE_MAPPED_ARTIFACTS", "1")
	if got := cudaMoEArtifactLayers(); got != 2 {
		t.Fatalf("explicit MoE artifact layers = %d, want 2", got)
	}
	if got := cudaReleaseMappedArtifacts(); got != 1 {
		t.Fatalf("explicit mapped-artifact release mode = %d, want 1", got)
	}
}
