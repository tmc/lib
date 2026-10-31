package main

import (
	"bytes"
	"crypto/sha256"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

func TestEGBundleBuildAlignAndInspect(t *testing.T) {
	tmpDir := t.TempDir()
	cacheDir := filepath.Join(tmpDir, "pgo_cache")
	cache := autotune.NewVerifiedPGOCache(cacheDir)

	key := autotune.PGOCacheKey{
		Arch:          "sm_121a",
		DriverVersion: "590.48.01",
		CUDAVersion:   "13.0",
		WeightDigest:  "sha256:llama_q4",
		PlanKey:       "cross_layer_decode_m1",
		ShapeBucket:   1,
	}
	cubin := []byte("MOCK_CUBIN_ELF_PAYLOAD_M1")
	cubinSum := sha256.Sum256(cubin)
	srcSum := sha256.Sum256([]byte("extern \"C\" __global__ void k() {}"))
	proof := autotune.PGOPromotionProof{
		Key:            key,
		CandidateName:  "cross_l0_l1_skinny_m1_rpb16",
		CubinSHA256:    cubinSum,
		SourceSHA256:   srcSum,
		ScreenBlocks:   3,
		HoldoutBlocks:  6,
		ScreenRatio:    1.07,
		HoldoutRatio:   1.06,
		AANullRatio:    1.001,
		OneSidedPValue: 0.015625,
		TokensMatched:  true,
		PromotedAt:     time.Date(2026, 10, 31, 9, 0, 0, 0, time.UTC),
	}
	if err := cache.Store(key, proof, cubin); err != nil {
		t.Fatalf("cache.Store: %v", err)
	}

	weights := []WeightTensorSpec{
		{
			Name:  "layers.0.attn.wqkv.weight",
			DType: "BF16",
			Shape: []int{256, 256},
			Data:  bytes.Repeat([]byte{0x3c, 0x00}, 512),
		},
		{
			Name:  "layers.0.mlp.w_down.weight",
			DType: "BF16",
			Shape: []int{256, 512},
			Data:  bytes.Repeat([]byte{0x38, 0x00}, 777),
		},
	}

	bundlePath := filepath.Join(tmpDir, "model.egbundle")
	manifest, err := WriteBundleFile(bundlePath, BundleBuildConfig{
		Arch:     "sm_121a",
		PGOCache: cache,
		Weights:  weights,
	})
	if err != nil {
		t.Fatalf("WriteBundleFile: %v", err)
	}

	if len(manifest.Buckets) != 4 {
		t.Fatalf("expected 4 buckets (M=1,2,4,8), got %d", len(manifest.Buckets))
	}
	wantBuckets := []int{1, 2, 4, 8}
	for i, wantM := range wantBuckets {
		if manifest.Buckets[i].BucketM != wantM {
			t.Fatalf("bucket[%d].BucketM=%d, want %d", i, manifest.Buckets[i].BucketM, wantM)
		}
	}
	if len(manifest.PGOManifest) != 1 {
		t.Fatalf("expected 1 PGO manifest entry, got %d", len(manifest.PGOManifest))
	}
	if manifest.PGOManifest[0].CubinOffset%PageAlignment != 0 {
		t.Fatalf("PGO cubin offset %d not 4096-byte aligned", manifest.PGOManifest[0].CubinOffset)
	}
	for name, w := range manifest.Weights {
		if w.PageOffset%PageAlignment != 0 {
			t.Fatalf("weight %s PageOffset=%d not 4096-byte aligned", name, w.PageOffset)
		}
	}

	// Verify -inspect mode via CLI run().
	var stdout, stderr bytes.Buffer
	if err := run([]string{"-inspect", bundlePath}, &stdout, &stderr); err != nil {
		t.Fatalf("run(-inspect): %v (stderr=%s)", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "layers.0.attn.wqkv.weight") {
		t.Fatalf("-inspect output missing weight header: %s", stdout.String())
	}

	// Verify tampering with payload bytes fails SHA-256 verification.
	_, rawArchive, err := BuildBundle(BundleBuildConfig{
		Arch:     "sm_121a",
		PGOCache: cache,
		Weights:  weights,
	})
	if err != nil {
		t.Fatalf("BuildBundle: %v", err)
	}
	rawArchive[len(rawArchive)-1] ^= 0xff
	if _, err := InspectBundle(bytes.NewReader(rawArchive)); err == nil {
		t.Fatalf("expected InspectBundle to fail on corrupted payload")
	}
}
