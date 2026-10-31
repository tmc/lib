package autotune

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func validPGOTestFixture() (PGOCacheKey, PGOPromotionProof, []byte) {
	key := PGOCacheKey{
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		WeightDigest:  "a3f19b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a",
		PlanKey:       "llama3_8b_layer_post",
		ShapeBucket:   1,
	}
	cubin := []byte("\x7fELF-synthetic-sm121a-megakernel-cubin-payload-v1")
	source := []byte("__global__ void fused_post_attn_norm_qkv() {}")
	proof := PGOPromotionProof{
		CandidateName:  "cand_tma2_reg128_unroll4",
		ScheduleJSON:   []byte(`{"tma_stages":2,"max_regs":128,"unroll":4}`),
		CubinSHA256:    sha256.Sum256(cubin),
		SourceSHA256:   sha256.Sum256(source),
		ScreenBlocks:   3,
		HoldoutBlocks:  6,
		ScreenRatio:    1.048,
		HoldoutRatio:   1.042,
		AANullRatio:    1.001,
		OneSidedPValue: 0.015625,
		TokensMatched:  true,
		PromotedAt:     time.Date(2026, 10, 31, 8, 0, 0, 0, time.UTC),
	}
	return key, proof, cubin
}

func TestVerifiedPGOCacheStoreAndReload(t *testing.T) {
	dir := t.TempDir()
	cache := NewVerifiedPGOCache(dir)
	key, proof, cubin := validPGOTestFixture()

	if err := cache.Store(key, proof, cubin); err != nil {
		t.Fatalf("Store: %v", err)
	}

	// Warm file cache once and verify sub-millisecond reload latency.
	if _, _, err := cache.LoadVerified(key); err != nil {
		t.Fatalf("initial LoadVerified: %v", err)
	}

	const reloads = 10
	durations := make([]time.Duration, reloads)
	var gotProof PGOPromotionProof
	var gotCubin []byte
	for i := range reloads {
		start := time.Now()
		var err error
		gotProof, gotCubin, err = cache.LoadVerified(key)
		if err != nil {
			t.Fatalf("LoadVerified(%d): %v", i, err)
		}
		durations[i] = time.Since(start)
	}
	slices.Sort(durations)
	medianReload := durations[len(durations)/2]
	t.Logf("VerifiedPGOCache reload median=%v min=%v", medianReload, durations[0])
	if durations[0] > 5*time.Millisecond {
		t.Errorf("min reload latency = %v, want < 5ms (target <1ms)", durations[0])
	}

	if !bytes.Equal(gotCubin, cubin) {
		t.Fatalf("cubin mismatch: got %q, want %q", gotCubin, cubin)
	}
	if gotProof.Key != key {
		t.Errorf("proof.Key = %+v, want %+v", gotProof.Key, key)
	}
	if gotProof.CandidateName != proof.CandidateName {
		t.Errorf("CandidateName = %q, want %q", gotProof.CandidateName, proof.CandidateName)
	}
	if !bytes.Equal(gotProof.ScheduleJSON, proof.ScheduleJSON) {
		t.Errorf("ScheduleJSON = %s, want %s", gotProof.ScheduleJSON, proof.ScheduleJSON)
	}
	if gotProof.CubinSHA256 != proof.CubinSHA256 {
		t.Errorf("CubinSHA256 = %x, want %x", gotProof.CubinSHA256, proof.CubinSHA256)
	}
	if gotProof.SourceSHA256 != proof.SourceSHA256 {
		t.Errorf("SourceSHA256 = %x, want %x", gotProof.SourceSHA256, proof.SourceSHA256)
	}

	// Ensure no leftover temporary files remain after atomic write.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("dir entries = %d, want 2 (.json and .cubin)", len(entries))
	}
}

func TestVerifiedPGOCacheStoreRejections(t *testing.T) {
	baseKey, baseProof, baseCubin := validPGOTestFixture()

	tests := []struct {
		name    string
		mutate  func(k *PGOCacheKey, p *PGOPromotionProof, cubin *[]byte)
		wantErr error
	}{
		{
			name: "TokensMatchedFalse",
			mutate: func(_ *PGOCacheKey, p *PGOPromotionProof, _ *[]byte) {
				p.TokensMatched = false
			},
			wantErr: ErrPGOTokenMismatch,
		},
		{
			name: "HoldoutBlocksBelow6",
			mutate: func(_ *PGOCacheKey, p *PGOPromotionProof, _ *[]byte) {
				p.HoldoutBlocks = 5
			},
			wantErr: ErrPGOHoldoutGate,
		},
		{
			name: "OneSidedPValueAbove005",
			mutate: func(_ *PGOCacheKey, p *PGOPromotionProof, _ *[]byte) {
				p.OneSidedPValue = 0.0625
			},
			wantErr: ErrPGOHoldoutGate,
		},
		{
			name: "OneSidedPValueNegative",
			mutate: func(_ *PGOCacheKey, p *PGOPromotionProof, _ *[]byte) {
				p.OneSidedPValue = -0.01
			},
			wantErr: ErrPGOHoldoutGate,
		},
		{
			name: "HoldoutRatioNotAboveOne",
			mutate: func(_ *PGOCacheKey, p *PGOPromotionProof, _ *[]byte) {
				p.HoldoutRatio = 1.0
			},
			wantErr: ErrPGOHoldoutGate,
		},
		{
			name: "EmptyCubin",
			mutate: func(_ *PGOCacheKey, _ *PGOPromotionProof, cubin *[]byte) {
				*cubin = nil
			},
			wantErr: ErrPGODigestMismatch,
		},
		{
			name: "CubinSHA256Mismatch",
			mutate: func(_ *PGOCacheKey, _ *PGOPromotionProof, cubin *[]byte) {
				*cubin = append([]byte(nil), (*cubin)...)
				(*cubin)[len(*cubin)-1] ^= 0xff
			},
			wantErr: ErrPGODigestMismatch,
		},
		{
			name: "ProofKeyMismatch",
			mutate: func(k *PGOCacheKey, p *PGOPromotionProof, _ *[]byte) {
				other := *k
				other.Arch = "sm_90a"
				p.Key = other
			},
			wantErr: ErrPGOKeyMismatch,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cache := NewVerifiedPGOCache(dir)
			k := baseKey
			p := baseProof
			c := append([]byte(nil), baseCubin...)
			tc.mutate(&k, &p, &c)

			err := cache.Store(k, p, c)
			if err == nil {
				t.Fatalf("Store succeeded, want error %v", tc.wantErr)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Fatalf("Store error = %v, want errors.Is(_, %v)", err, tc.wantErr)
			}
			keys, listErr := cache.List()
			if listErr != nil {
				t.Fatalf("List: %v", listErr)
			}
			if len(keys) != 0 {
				t.Fatalf("rejected Store left %d entries on disk", len(keys))
			}
		})
	}
}

func TestVerifiedPGOCacheLoadTamperDetection(t *testing.T) {
	key, proof, cubin := validPGOTestFixture()

	t.Run("CorruptedCubinBytes", func(t *testing.T) {
		dir := t.TempDir()
		cache := NewVerifiedPGOCache(dir)
		if err := cache.Store(key, proof, cubin); err != nil {
			t.Fatalf("Store: %v", err)
		}
		cubinPath := filepath.Join(dir, key.CacheID()+".cubin")
		corrupted := append([]byte(nil), cubin...)
		corrupted[0] ^= 0x55
		if err := os.WriteFile(cubinPath, corrupted, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, _, err := cache.LoadVerified(key)
		if !errors.Is(err, ErrPGODigestMismatch) {
			t.Fatalf("LoadVerified error = %v, want %v", err, ErrPGODigestMismatch)
		}
	})

	t.Run("TamperedJSONProofTokensMatched", func(t *testing.T) {
		dir := t.TempDir()
		cache := NewVerifiedPGOCache(dir)
		if err := cache.Store(key, proof, cubin); err != nil {
			t.Fatalf("Store: %v", err)
		}
		jsonPath := filepath.Join(dir, key.CacheID()+".json")
		tampered := proof
		tampered.Key = key
		tampered.TokensMatched = false
		b, err := json.Marshal(tampered)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		if err := os.WriteFile(jsonPath, b, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, _, err = cache.LoadVerified(key)
		if !errors.Is(err, ErrPGOTokenMismatch) {
			t.Fatalf("LoadVerified error = %v, want %v", err, ErrPGOTokenMismatch)
		}
	})

	t.Run("TamperedJSONProofHoldoutStats", func(t *testing.T) {
		dir := t.TempDir()
		cache := NewVerifiedPGOCache(dir)
		if err := cache.Store(key, proof, cubin); err != nil {
			t.Fatalf("Store: %v", err)
		}
		jsonPath := filepath.Join(dir, key.CacheID()+".json")
		tampered := proof
		tampered.Key = key
		tampered.OneSidedPValue = 0.25
		b, err := json.Marshal(tampered)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		if err := os.WriteFile(jsonPath, b, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, _, err = cache.LoadVerified(key)
		if !errors.Is(err, ErrPGOHoldoutGate) {
			t.Fatalf("LoadVerified error = %v, want %v", err, ErrPGOHoldoutGate)
		}
	})

	t.Run("TamperedJSONProofKey", func(t *testing.T) {
		dir := t.TempDir()
		cache := NewVerifiedPGOCache(dir)
		if err := cache.Store(key, proof, cubin); err != nil {
			t.Fatalf("Store: %v", err)
		}
		jsonPath := filepath.Join(dir, key.CacheID()+".json")
		tampered := proof
		tampered.Key = key
		tampered.Key.DriverVersion = "575.00.00"
		b, err := json.Marshal(tampered)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		if err := os.WriteFile(jsonPath, b, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		_, _, err = cache.LoadVerified(key)
		if !errors.Is(err, ErrPGOKeyMismatch) {
			t.Fatalf("LoadVerified error = %v, want %v", err, ErrPGOKeyMismatch)
		}
	})
}

func TestVerifiedPGOCacheKeyIsolationAndLifecycle(t *testing.T) {
	dir := t.TempDir()
	cache := NewVerifiedPGOCache(dir)
	baseKey, baseProof, _ := validPGOTestFixture()

	variants := []struct {
		label string
		key   PGOCacheKey
	}{
		{label: "base", key: baseKey},
		{label: "arch", key: func() PGOCacheKey { k := baseKey; k.Arch = "sm_100a"; return k }()},
		{label: "driver", key: func() PGOCacheKey { k := baseKey; k.DriverVersion = "580.999.00"; return k }()},
		{label: "cuda", key: func() PGOCacheKey { k := baseKey; k.CUDAVersion = "13.5"; return k }()},
		{label: "weights", key: func() PGOCacheKey { k := baseKey; k.WeightDigest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"; return k }()},
		{label: "plan", key: func() PGOCacheKey { k := baseKey; k.PlanKey = "llama3_8b_layer_pre"; return k }()},
		{label: "bucket8", key: func() PGOCacheKey { k := baseKey; k.ShapeBucket = 8; return k }()},
		{label: "bucket64", key: func() PGOCacheKey { k := baseKey; k.ShapeBucket = 64; return k }()},
	}

	seenIDs := make(map[string]string)
	for _, v := range variants {
		id := v.key.CacheID()
		if prev, exists := seenIDs[id]; exists {
			t.Fatalf("CacheID collision between %s and %s: %s", prev, v.label, id)
		}
		seenIDs[id] = v.label

		cubin := []byte("cubin-for-" + v.label)
		proof := baseProof
		proof.CandidateName = "cand_" + v.label
		proof.CubinSHA256 = sha256.Sum256(cubin)
		if err := cache.Store(v.key, proof, cubin); err != nil {
			t.Fatalf("Store(%s): %v", v.label, err)
		}
	}

	listed, err := cache.List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(listed) != len(variants) {
		t.Fatalf("List count = %d, want %d", len(listed), len(variants))
	}

	for _, v := range variants {
		gotProof, gotCubin, err := cache.LoadVerified(v.key)
		if err != nil {
			t.Fatalf("LoadVerified(%s): %v", v.label, err)
		}
		wantCubin := []byte("cubin-for-" + v.label)
		if !bytes.Equal(gotCubin, wantCubin) {
			t.Errorf("LoadVerified(%s) cubin = %q, want %q", v.label, gotCubin, wantCubin)
		}
		if gotProof.CandidateName != "cand_"+v.label {
			t.Errorf("LoadVerified(%s) CandidateName = %q, want %q", v.label, gotProof.CandidateName, "cand_"+v.label)
		}
	}

	// Invalidate one key and verify other keys remain intact.
	if err := cache.Invalidate(baseKey); err != nil {
		t.Fatalf("Invalidate(base): %v", err)
	}
	if _, _, err := cache.LoadVerified(baseKey); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("LoadVerified after Invalidate = %v, want os.ErrNotExist", err)
	}
	listedAfter, err := cache.List()
	if err != nil {
		t.Fatalf("List after Invalidate: %v", err)
	}
	if len(listedAfter) != len(variants)-1 {
		t.Fatalf("List count after Invalidate = %d, want %d", len(listedAfter), len(variants)-1)
	}
}
