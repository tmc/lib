package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

func TestLlamaAOTBundleExportVerifyAndWarmLoad(t *testing.T) {
	tmpDir := t.TempDir()
	cacheDir := filepath.Join(tmpDir, "pgo-cache")
	pgoCache := autotune.NewVerifiedPGOCache(cacheDir)

	modelCfg := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         64,
	}

	buckets := []int{1, 2, 4, 8}
	pgoCubins := make(map[int][]byte, len(buckets))
	for _, m := range buckets {
		cubin := []byte(fmt.Sprintf("ELF-CUBIN-SM121A-LLAMA-CROSS-L0-L1-BUCKET-M%d", m))
		pgoCubins[m] = cubin
	}

	bundlePath := filepath.Join(tmpDir, "llama_l0_l1.egbundle")
	exported, err := BuildAndExportLlamaBundle(LlamaAOTBundleConfig{
		Model:         &modelCfg,
		QKVBias:       true,
		LayerL:        0,
		LayerNext:     1,
		Buckets:       buckets,
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.0",
		WeightDigest:  "llama3_8b_test_weights_v1",
		PGOCache:      pgoCache,
		PGOCubins:     pgoCubins,
	}, bundlePath)
	if err != nil {
		t.Fatalf("BuildAndExportLlamaBundle: %v", err)
	}

	if exported == nil {
		t.Fatal("BuildAndExportLlamaBundle returned nil manifest")
	}
	if exported.PageAlignment != LlamaBundlePageAlignment {
		t.Fatalf("PageAlignment = %d, want %d", exported.PageAlignment, LlamaBundlePageAlignment)
	}
	if exported.DataBasePage == 0 || exported.DataBasePage%LlamaBundlePageAlignment != 0 {
		t.Fatalf("DataBasePage = %d, want non-zero multiple of %d", exported.DataBasePage, LlamaBundlePageAlignment)
	}
	if len(exported.Buckets) != len(buckets) {
		t.Fatalf("len(exported.Buckets) = %d, want %d", len(exported.Buckets), len(buckets))
	}

	// Verify all buckets {1, 2, 4, 8} passed VerifyMegakernelBarrierSafety.
	for i, wantM := range buckets {
		b := exported.Buckets[i]
		if b.BucketM != wantM {
			t.Fatalf("bucket[%d].BucketM = %d, want %d", i, b.BucketM, wantM)
		}
		if b.BarrierReport == nil || !b.BarrierReport.Passed() {
			t.Fatalf("bucket M=%d BarrierReport failed: %+v", wantM, b.BarrierReport)
		}
		if !b.BarrierReport.RAWBarrierDominanceOK ||
			!b.BarrierReport.ArenaAliasingSafetyOK ||
			!b.BarrierReport.NamedBarrierUniformityOK ||
			!b.BarrierReport.EpochSlotNonOverlapOK {
			t.Fatalf("bucket M=%d BarrierReport properties incomplete: %+v", wantM, b.BarrierReport)
		}
		if b.NumStages != 6 {
			t.Fatalf("bucket M=%d NumStages = %d, want 6", wantM, b.NumStages)
		}
		if len(b.StageSpans) != 6 {
			t.Fatalf("bucket M=%d len(StageSpans) = %d, want 6", wantM, len(b.StageSpans))
		}
		if !strings.Contains(b.LineInfoSource, "#line ") {
			t.Fatalf("bucket M=%d LineInfoSource missing #line directives", wantM)
		}
		if b.PGOEntry == nil {
			t.Fatalf("bucket M=%d missing PGOEntry", wantM)
		}
		if b.PGOEntry.CubinOffset%LlamaBundlePageAlignment != 0 {
			t.Fatalf("bucket M=%d PGO CubinOffset = %d, not %d-byte page aligned",
				wantM, b.PGOEntry.CubinOffset, LlamaBundlePageAlignment)
		}
	}

	// Verify 4096-byte page alignment of all Safetensors weight tensor offsets.
	if len(exported.Weights) == 0 {
		t.Fatal("exported.Weights is empty")
	}
	for name, hdr := range exported.Weights {
		if hdr.PageOffset < exported.DataBasePage {
			t.Fatalf("weight %q PageOffset %d < DataBasePage %d", name, hdr.PageOffset, exported.DataBasePage)
		}
		if hdr.PageOffset%LlamaBundlePageAlignment != 0 {
			t.Fatalf("weight %q PageOffset = %d, not %d-byte page aligned", name, hdr.PageOffset, LlamaBundlePageAlignment)
		}
		if hdr.ByteLength == 0 {
			t.Fatalf("weight %q ByteLength = 0", name)
		}
	}

	// Verify < 10ms warm load latency and exact source recovery.
	t0 := time.Now()
	loaded, sources, err := LoadVerifiedLlamaBundle(bundlePath)
	warmElapsed := time.Since(t0)
	if err != nil {
		t.Fatalf("LoadVerifiedLlamaBundle: %v", err)
	}
	if warmElapsed >= 10*time.Millisecond {
		t.Fatalf("warm load took %v, want < 10ms", warmElapsed)
	}
	if loaded == nil {
		t.Fatal("LoadVerifiedLlamaBundle returned nil manifest")
	}
	if loaded.PayloadSHA256 != exported.PayloadSHA256 {
		t.Fatalf("loaded.PayloadSHA256 = %s, want %s", loaded.PayloadSHA256, exported.PayloadSHA256)
	}
	if len(sources) != len(buckets) {
		t.Fatalf("len(sources) = %d, want %d", len(sources), len(buckets))
	}
	for i, m := range buckets {
		src, ok := sources[m]
		if !ok || src == "" {
			t.Fatalf("missing CUDA source for bucket M=%d", m)
		}
		if src != exported.Buckets[i].CUDASource {
			t.Fatalf("bucket M=%d loaded source mismatch", m)
		}
	}
}

func TestLlamaAOTBundleTamperRejection(t *testing.T) {
	tmpDir := t.TempDir()
	modelCfg := DefaultModelConfig()

	cacheDir := filepath.Join(tmpDir, "pgo")
	cache := autotune.NewVerifiedPGOCache(cacheDir)
	key := autotune.PGOCacheKey{
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.0",
		WeightDigest:  "llama_tamper_test",
		PlanKey:       "llama_cross_l0_l1_m1",
		ShapeBucket:   1,
	}
	cubin := []byte("ELF-CUBIN-TAMPER-TEST-M1")
	proof := autotune.PGOPromotionProof{
		Key:            key,
		CandidateName:  "llama_cross_l0_l1_skinny_m1",
		CubinSHA256:    sha256.Sum256(cubin),
		SourceSHA256:   sha256.Sum256([]byte("__global__ void k(){}")),
		ScreenBlocks:   3,
		HoldoutBlocks:  6,
		ScreenRatio:    1.07,
		HoldoutRatio:   1.06,
		AANullRatio:    1.00,
		OneSidedPValue: 0.015625,
		TokensMatched:  true,
		PromotedAt:     time.Date(2026, 10, 31, 10, 0, 0, 0, time.UTC),
	}
	if err := cache.Store(key, proof, cubin); err != nil {
		t.Fatalf("cache.Store: %v", err)
	}

	bundlePath := filepath.Join(tmpDir, "tamper.egbundle")
	manifest, err := BuildAndExportLlamaBundle(LlamaAOTBundleConfig{
		Model:         &modelCfg,
		LayerL:        0,
		LayerNext:     1,
		Buckets:       []int{1, 2, 4, 8},
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.0",
		WeightDigest:  "llama_tamper_test",
		PlanKeyPrefix: "llama_cross_l0_l1",
		PGOCache:      cache,
	}, bundlePath)
	if err != nil {
		t.Fatalf("BuildAndExportLlamaBundle: %v", err)
	}

	origBytes, err := os.ReadFile(bundlePath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	var firstWeightOffset int
	for _, w := range manifest.Weights {
		firstWeightOffset = int(w.PageOffset)
		break
	}
	cubinOffset := int(manifest.PGOCache[0].CubinOffset)

	tamperOffsets := []struct {
		name   string
		offset int
	}{
		{"magic_byte", 0},
		{"manifest_json_byte", 32},
		{"header_padding_byte", int(manifest.DataBasePage) - 1},
		{"weight_tensor_byte", firstWeightOffset},
		{"pgo_cubin_byte", cubinOffset},
		{"trailing_payload_byte", len(origBytes) - 1},
	}

	for _, tc := range tamperOffsets {
		t.Run(tc.name, func(t *testing.T) {
			corrupted := bytes.Clone(origBytes)
			corrupted[tc.offset] ^= 0x01
			corruptedPath := filepath.Join(tmpDir, tc.name+".egbundle")
			if err := os.WriteFile(corruptedPath, corrupted, 0o644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}
			if _, _, err := LoadVerifiedLlamaBundle(corruptedPath); err == nil {
				t.Fatalf("LoadVerifiedLlamaBundle succeeded on corrupted %s at offset %d, want error", tc.name, tc.offset)
			}
		})
	}
}

func TestLlamaAOTBundleFullLayerExportAndLoad(t *testing.T) {
	tmpDir := t.TempDir()
	modelCfg := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         64,
	}

	buckets := []int{1, 2, 4, 8}
	bundlePath := filepath.Join(tmpDir, "llama_full.egbundle")
	manifest, err := BuildAndExportLlamaBundle(LlamaAOTBundleConfig{
		Model:     &modelCfg,
		LayerL:    0,
		LayerNext: 1,
		FullLayer: true,
		Buckets:   buckets,
		Arch:      "sm_121a",
	}, bundlePath)
	if err != nil {
		t.Fatalf("BuildAndExportLlamaBundle (full-layer): %v", err)
	}
	if !manifest.FullLayer {
		t.Fatal("expected manifest.FullLayer = true")
	}
	if len(manifest.Buckets) != len(buckets) {
		t.Fatalf("len(manifest.Buckets) = %d, want %d", len(manifest.Buckets), len(buckets))
	}
	for i, wantM := range buckets {
		b := manifest.Buckets[i]
		if b.BucketM != wantM {
			t.Fatalf("bucket[%d].BucketM = %d, want %d", i, b.BucketM, wantM)
		}
		if b.BarrierReport == nil || !b.BarrierReport.Passed() {
			t.Fatalf("full-layer bucket M=%d barrier report failed: %+v", wantM, b.BarrierReport)
		}
		if b.NumStages != 18 {
			t.Fatalf("full-layer bucket M=%d NumStages = %d, want 18", wantM, b.NumStages)
		}
	}

	t0 := time.Now()
	loaded, sources, err := LoadVerifiedLlamaBundle(bundlePath)
	elapsed := time.Since(t0)
	if err != nil {
		t.Fatalf("LoadVerifiedLlamaBundle (full-layer): %v", err)
	}
	if elapsed >= 10*time.Millisecond {
		t.Fatalf("full-layer warm load took %v, want < 10ms", elapsed)
	}
	if loaded.BarriersSHA256 != manifest.BarriersSHA256 || len(sources) != len(buckets) {
		t.Fatalf("loaded full-layer bundle mismatch")
	}
}

