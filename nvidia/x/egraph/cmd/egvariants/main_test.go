package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
)

func TestEGVariantsPTXASRefine(t *testing.T) {
	dir := t.TempDir()
	specPath := filepath.Join(dir, "spec.json")
	specJSON := `{
		"skinny_gemm": [
			{"Name": "skinny_qkv", "N": 2304, "K": 2048, "MaxM": 4, "Weight": "bf16", "Out": "bf16", "RowsPerBlock": 2, "Warps": 4, "Unroll": 4, "Prefetch": 2}
		],
		"tile_gemm": [
			{"Name": "tile_proj", "N": 2048, "K": 1024, "Weight": "bf16", "Out": "bf16", "BM": 128, "BN": 128, "Stages": 3, "WGMMA": true, "Swizzle128B": true}
		],
		"paged_attention": [
			{"Name": "attn_decode", "Hq": 16, "Hkv": 2, "QLen": 1, "Window": 2048, "Page": 1, "Q": "e4m3", "Out": "f32", "Chunk": 128, "Depth": 2}
		]
	}`
	if err := os.WriteFile(specPath, []byte(specJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	ptxasPath := filepath.Join(dir, "ptxas.log")
	ptxasLog := `
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z10skinny_qkvPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z10skinny_qkvPKtS0_Pt
    16 bytes stack frame, 16 bytes spill stores, 16 bytes spill loads
ptxas info    : Used 128 registers, 1024 bytes smem, 384 bytes cmem[0]
ptxas info    : (C7519) Potential Performance Loss: wgmma.mma_async instructions are serialized due to program dependence on compiler-inserted WG.AR in divergent path in the function '_Z9tile_projiPKtS0_Pt' in line 88
ptxas info    : Compiling entry function '_Z9tile_projiPKtS0_Pt' for 'sm_90a'
ptxas info    : Function properties for _Z9tile_projiPKtS0_Pt
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 128 registers, 16384 bytes smem, 384 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11attn_decodePKuPKhS2_PKiiS4_PKfS6_PfPh' for 'sm_90a'
ptxas info    : Function properties for _Z11attn_decodePKuPKhS2_PKiiS4_PKfS6_PfPh
    16 bytes stack frame, 12 bytes spill stores, 12 bytes spill loads
ptxas info    : Used 144 registers, 24576 bytes smem, 384 bytes cmem[0]
`
	if err := os.WriteFile(ptxasPath, []byte(ptxasLog), 0o644); err != nil {
		t.Fatal(err)
	}

	cacheDir := filepath.Join(dir, "pgo_cache")
	cache := autotune.NewVerifiedPGOCache(cacheDir)
	cubin := []byte("CUBIN_VARIANT_PAYLOAD")
	key := autotune.PGOCacheKey{
		Arch:          "sm_90a",
		DriverVersion: "570.124",
		CUDAVersion:   "12.8",
		WeightDigest:  "sha256:spec",
		PlanKey:       "tile_proj",
		ShapeBucket:   128,
	}
	if err := cache.Store(key, autotune.PGOPromotionProof{
		Key:            key,
		CandidateName:  "tile_proj_ws",
		CubinSHA256:    sha256.Sum256(cubin),
		SourceSHA256:   sha256.Sum256([]byte("// src")),
		ScreenBlocks:   3,
		HoldoutBlocks:  6,
		ScreenRatio:    1.06,
		HoldoutRatio:   1.05,
		AANullRatio:    1.00,
		OneSidedPValue: 0.015625,
		TokensMatched:  true,
	}, cubin); err != nil {
		t.Fatal(err)
	}

	outDir := filepath.Join(dir, "out")
	var stderr bytes.Buffer
	code := run([]string{
		"-ptxas", ptxasPath,
		"-ptxas-refine",
		"-rank-device", "NVIDIA H200",
		"-pgo-cache-dir", cacheDir,
		"-o", outDir,
		specPath,
	}, &stderr)
	if code != 0 {
		t.Fatalf("egvariants exit %d: %s", code, stderr.String())
	}

	for _, name := range []string{"manifest.json", "refined.json", "ranked.json", "pgo_cache.json"} {
		if _, err := os.Stat(filepath.Join(outDir, name)); err != nil {
			t.Fatalf("missing expected output file %s: %v", name, err)
		}
	}

	refBytes, err := os.ReadFile(filepath.Join(outDir, "refined.json"))
	if err != nil {
		t.Fatal(err)
	}
	var summary refinedSummary
	if err := json.Unmarshal(refBytes, &summary); err != nil {
		t.Fatalf("unmarshal refined.json: %v", err)
	}
	if len(summary.Steps) != 3 {
		t.Fatalf("len(summary.Steps) = %d, want 3", len(summary.Steps))
	}
	if len(summary.Actions) < 3 {
		t.Fatalf("expected >= 3 refinement actions, got %v", summary.Actions)
	}

	// Verify skinny_qkv register spill cure: Prefetch disabled and RowsPerBlock stepped down 2 -> 1.
	stSkinny := summary.Steps[0]
	if !stSkinny.Refined || !slices.Contains(stSkinny.Bottlenecks, qualify.BottleneckRegisterSpill) {
		t.Errorf("skinny_qkv step = %+v, want Refined=true with register_spill", stSkinny)
	}
	if stSkinny.RefinedStep.GEMM == nil || stSkinny.RefinedStep.GEMM.Prefetch != 0 || stSkinny.RefinedStep.GEMM.RowsPerBlock != 1 {
		t.Errorf("skinny_qkv refined GEMM = %+v, want Prefetch=0 RowsPerBlock=1", stSkinny.RefinedStep.GEMM)
	}

	// Verify tile_proj WGMMA serialization cure: WarpSpecialized enabled.
	stTile := summary.Steps[1]
	if !stTile.Refined || !slices.Contains(stTile.Bottlenecks, qualify.BottleneckWGMMASerial) {
		t.Errorf("tile_proj step = %+v, want Refined=true with wgmma_serialization", stTile)
	}
	if stTile.RefinedStep.Tile == nil || !stTile.RefinedStep.Tile.WarpSpecialized {
		t.Errorf("tile_proj refined Tile = %+v, want WarpSpecialized=true", stTile.RefinedStep.Tile)
	}

	// Verify attn_decode register spill cure: Depth stepped down 2 -> 1.
	stAttn := summary.Steps[2]
	if !stAttn.Refined || !slices.Contains(stAttn.Bottlenecks, qualify.BottleneckRegisterSpill) {
		t.Errorf("attn_decode step = %+v, want Refined=true with register_spill", stAttn)
	}
	if stAttn.RefinedStep.Attention == nil || stAttn.RefinedStep.Attention.Depth != 1 {
		t.Errorf("attn_decode refined Attention = %+v, want Depth=1", stAttn.RefinedStep.Attention)
	}

	// Verify -ptxas-refine without -ptxas fails with usage/validation error.
	stderr.Reset()
	if code := run([]string{"-ptxas-refine", "-o", filepath.Join(dir, "bad"), specPath}, &stderr); code == 0 {
		t.Fatal("expected non-zero exit code when -ptxas-refine is set without -ptxas")
	}
	if !strings.Contains(stderr.String(), "-ptxas") {
		t.Errorf("unexpected error message: %s", stderr.String())
	}
}
