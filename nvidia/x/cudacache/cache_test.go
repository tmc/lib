package cudacache_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func newTestArtifact(t *testing.T, entryPoint string, target string) *cudakernel.Artifact {
	t.Helper()
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "data", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatalf("NewSchema: %v", err)
	}
	ptx := fmt.Sprintf(".version 8.0\n.target %s\n.address_size 64\n.entry %s(.param .u64 ptr) { ret; }\n", target, entryPoint)
	art, err := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		[]byte(ptx),
		entryPoint,
		schema,
		target,
	)
	if err != nil {
		t.Fatalf("NewArtifact: %v", err)
	}
	return art
}

// TestArtifactCacheRoundTrip tests storing and retrieving an artifact locally.
func TestArtifactCacheRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	art := newTestArtifact(t, "roundtrip_kernel", "sm_121a")
	ctx := context.Background()
	key := "test-roundtrip-key"

	if err := c.PutArtifact(ctx, key, art); err != nil {
		t.Fatalf("PutArtifact: %v", err)
	}

	got, err := c.GetArtifact(ctx, key)
	if err != nil {
		t.Fatalf("GetArtifact: %v", err)
	}

	if got.EntryPoint != art.EntryPoint {
		t.Errorf("entry point mismatch: got %q, want %q", got.EntryPoint, art.EntryPoint)
	}
	if got.Digest != art.Digest {
		t.Errorf("digest mismatch: got %q, want %q", got.Digest, art.Digest)
	}
	if got.Target != art.Target {
		t.Errorf("target mismatch: got %q, want %q", got.Target, art.Target)
	}
}

// TestTargetCacheInvalidation tests that compilerCacheKey hashing sm_121a vs sm_100a
// produces different keys and forces invalidation when target architecture changes.
func TestTargetCacheInvalidation(t *testing.T) {
	baseID := cudacache.CompilerIdentity{
		NVRTCVersion: "12.8",
		DriverSONAME: "libcuda.so.1",
		TargetArch:   "sm_121a",
		Flags:        []string{"--std=c++20"},
		OutputKind:   nvrtcx.OutputCUBIN,
	}

	inputKey := "source-hash-xyz"
	key121 := cudacache.CompilerCacheKey(inputKey, baseID)

	// Shift target arch to sm_100a
	sm100ID := baseID
	sm100ID.TargetArch = "sm_100a"
	key100 := cudacache.CompilerCacheKey(inputKey, sm100ID)

	if key121 == key100 {
		t.Fatalf("expected different cache keys for sm_121a vs sm_100a, got identical: %s", key121)
	}

	// Toolchain update (nvrtc.Version)
	toolchainID := baseID
	toolchainID.NVRTCVersion = "13.0"
	keyToolchain := cudacache.CompilerCacheKey(inputKey, toolchainID)
	if keyToolchain == key121 {
		t.Fatalf("expected different cache key on NVRTC version upgrade")
	}

	// Driver SONAME update
	driverID := baseID
	driverID.DriverSONAME = "libcuda.so.550.54.14"
	keyDriver := cudacache.CompilerCacheKey(inputKey, driverID)
	if keyDriver == key121 {
		t.Fatalf("expected different cache key on Driver SONAME change")
	}

	// Compilation flags change
	flagsID := baseID
	flagsID.Flags = []string{"--std=c++20", "--extra-device-vectorization"}
	keyFlags := cudacache.CompilerCacheKey(inputKey, flagsID)
	if keyFlags == key121 {
		t.Fatalf("expected different cache key on compilation flags change")
	}

	// Output kind change
	outputID := baseID
	outputID.OutputKind = nvrtcx.OutputPTX
	keyOutput := cudacache.CompilerCacheKey(inputKey, outputID)
	if keyOutput == key121 {
		t.Fatalf("expected different cache key on output kind change")
	}
}

// TestNumericalModeCacheInvalidation verifies that changing numerical modes
// (e.g. FP32 vs BF16 vs MXFP4) invalidates the workload identity key.
func TestNumericalModeCacheInvalidation(t *testing.T) {
	wFP32 := cudacache.WorkloadIdentity{
		ProgramID:     "matmul_layer_1",
		ShapeDims:     []int{16, 512, 512},
		NumericalMode: "fp32",
	}

	wBF16 := wFP32
	wBF16.NumericalMode = "bf16"

	wMXFP4 := wFP32
	wMXFP4.NumericalMode = "mxfp4"

	if wFP32.Key() == wBF16.Key() {
		t.Fatalf("workload keys identical for fp32 vs bf16")
	}
	if wFP32.Key() == wMXFP4.Key() {
		t.Fatalf("workload keys identical for fp32 vs mxfp4")
	}
}

// TestConcurrentPublication verifies that concurrent writers for the same or different keys
// execute safely with atomic renames without data corruption or partial file reads.
func TestConcurrentPublication(t *testing.T) {
	tmpDir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const numWriters = 20
	var wg sync.WaitGroup
	wg.Add(numWriters)

	for i := 0; i < numWriters; i++ {
		go func(idx int) {
			defer wg.Done()
			key := fmt.Sprintf("concurrent-key-%d", idx%3) // Writers contend on 3 keys
			art := newTestArtifact(t, fmt.Sprintf("kernel_%d", idx%3), "sm_121a")
			if err := c.PutArtifact(context.Background(), key, art); err != nil {
				t.Errorf("PutArtifact worker %d: %v", idx, err)
			}
			retrieved, err := c.GetArtifact(context.Background(), key)
			if err != nil {
				t.Errorf("GetArtifact worker %d: %v", idx, err)
			} else if retrieved == nil {
				t.Errorf("GetArtifact returned nil for worker %d", idx)
			}
		}(i)
	}

	wg.Wait()
}

// TestCorruptArtifactRejected verifies that corrupt or truncated artifacts fail SHA-256 validation
// and are isolated into the quarantine directory.
func TestCorruptArtifactRejected(t *testing.T) {
	tmpDir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	art := newTestArtifact(t, "corrupt_test_kernel", "sm_121a")
	key := "corrupt-artifact-key"
	ctx := context.Background()

	if err := c.PutArtifact(ctx, key, art); err != nil {
		t.Fatalf("PutArtifact: %v", err)
	}

	// Directly mutate content on disk under the old digest (Mutant negative control 1)
	artPath := filepath.Join(tmpDir, "artifacts", key+".artifact")
	data, err := os.ReadFile(artPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}

	// Tamper with bytes inside Raw
	var rawArt map[string]any
	if err := json.Unmarshal(data, &rawArt); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	rawArt["raw"] = "dGFtcGVyZWQtYnl0ZXM=" // Modified base64 content
	tamperedData, _ := json.Marshal(rawArt)
	if err := os.WriteFile(artPath, tamperedData, 0o644); err != nil {
		t.Fatalf("WriteFile tampered: %v", err)
	}

	// Evict from in-memory cache to force reading from disk
	cFresh, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		t.Fatalf("New fresh cache: %v", err)
	}

	_, err = cFresh.GetArtifact(ctx, key)
	if err == nil {
		t.Fatalf("expected ErrCorruptArtifact, got nil error")
	}
	if !errors.Is(err, cudacache.ErrCorruptArtifact) {
		t.Fatalf("expected ErrCorruptArtifact, got: %v", err)
	}

	// Verify quarantine directory contains the quarantined corrupt file
	quarantineEntries, err := os.ReadDir(filepath.Join(tmpDir, "quarantine"))
	if err != nil {
		t.Fatalf("ReadDir quarantine: %v", err)
	}
	if len(quarantineEntries) == 0 {
		t.Fatalf("expected quarantined file, got 0 entries")
	}
}

// TestReceiptRankingStale verifies that a search receipt evaluated in an older epoch
// is identified as stale and does not authorize silent promotion.
func TestReceiptRankingStale(t *testing.T) {
	policy := cudacache.SearchPolicyID{
		Strategy:     "beam",
		Options:      map[string]string{"beam_width": "4"},
		Seed:         42,
		ScoringEpoch: "epoch_1",
	}

	receipt := &cudacache.SearchReceipt{
		PolicyID:    policy,
		WorkloadKey: "workload_123",
		CandidateID: "candidate_abc",
		ArtifactKey: "artifact_xyz",
		Complete:    true,
		TimingKind:  cudatrial.TimingWallElapsed,
		Timings:     map[string]int64{string(cudatrial.TimingWallElapsed): 1000000},
		RankScore:   0.85,
		Epoch:       "epoch_1",
	}

	if receipt.IsStale("epoch_1") {
		t.Errorf("receipt should not be stale for matching epoch")
	}
	if !receipt.IsStale("epoch_2") {
		t.Errorf("receipt should be reported stale for new scoring epoch")
	}
}

// TestRemoteDownloadLimits verifies that downloads exceeding MaxRemoteDownloadBytes are rejected.
func TestRemoteDownloadLimits(t *testing.T) {
	reg := cudacache.NewMemoryRegistry()
	server := cudacache.NewTestServer(reg)
	defer server.Close()

	// Store an oversized payload in the remote server
	oversizedBytes := make([]byte, 5000)
	reg.PublishArtifact(context.Background(), "oversized-key", oversizedBytes)

	tmpDir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{
		RootDir:                tmpDir,
		MaxRemoteDownloadBytes: 1000, // 1000 bytes limit
		Remote:                 cudacache.NewHTTPRemoteSource(server.URL, server.Client()),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = c.GetArtifact(context.Background(), "oversized-key")
	if err == nil {
		t.Fatalf("expected error on oversized download, got nil")
	}
	if !errors.Is(err, cudacache.ErrDownloadLimitExceeded) {
		t.Fatalf("expected ErrDownloadLimitExceeded, got: %v", err)
	}
}

// TestSearchPolicyReceiptIdentity tests that changing search policy parameters (e.g. beam width or seed)
// creates a distinct receipt identity.
func TestSearchPolicyReceiptIdentity(t *testing.T) {
	p1 := cudacache.SearchPolicyID{
		Strategy: "beam",
		Options:  map[string]string{"beam_width": "4"},
		Seed:     100,
	}

	p2 := cudacache.SearchPolicyID{
		Strategy: "beam",
		Options:  map[string]string{"beam_width": "8"}, // Different beam width
		Seed:     100,
	}

	p3 := cudacache.SearchPolicyID{
		Strategy: "beam",
		Options:  map[string]string{"beam_width": "4"},
		Seed:     200, // Different seed
	}

	if p1.String() == p2.String() {
		t.Errorf("expected different SearchPolicyID for different beam widths")
	}
	if p1.String() == p3.String() {
		t.Errorf("expected different SearchPolicyID for different seeds")
	}

	workload := "workload-alpha"
	k1 := cudacache.SearchReceiptKey(p1, workload)
	k2 := cudacache.SearchReceiptKey(p2, workload)
	if k1 == k2 {
		t.Errorf("receipt keys should differ for different policies on same workload")
	}
}

// TestArtifactReusableAcrossSearchPolicies verifies that multiple search policies
// share the identical underlying compiled binary artifact without cache misses or collisions.
func TestArtifactReusableAcrossSearchPolicies(t *testing.T) {
	tmpDir := t.TempDir()
	c, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	art := newTestArtifact(t, "shared_kernel", "sm_121a")
	artKey := "shared-artifact-key"
	ctx := context.Background()

	// Store the binary artifact once
	if err := c.PutArtifact(ctx, artKey, art); err != nil {
		t.Fatalf("PutArtifact: %v", err)
	}

	// Policy 1 (Beam search)
	policy1 := cudacache.SearchPolicyID{Strategy: "beam", Options: map[string]string{"beam_width": "4"}, Seed: 1}
	receipt1 := &cudacache.SearchReceipt{
		PolicyID:    policy1,
		WorkloadKey: "workload_common",
		CandidateID: "cand_1",
		ArtifactKey: artKey,
		Complete:    true,
		TimingKind:  cudatrial.TimingWallElapsed,
		Timings:     map[string]int64{string(cudatrial.TimingWallElapsed): 500000},
		RankScore:   0.92,
		Epoch:       "epoch_1",
	}

	// Policy 2 (MCTS)
	policy2 := cudacache.SearchPolicyID{Strategy: "mcts", Options: map[string]string{"rollout_depth": "10"}, Seed: 2}
	receipt2 := &cudacache.SearchReceipt{
		PolicyID:    policy2,
		WorkloadKey: "workload_common",
		CandidateID: "cand_2",
		ArtifactKey: artKey,
		Complete:    true,
		TimingKind:  cudatrial.TimingWallElapsed,
		Timings:     map[string]int64{string(cudatrial.TimingWallElapsed): 480000},
		RankScore:   0.95,
		Epoch:       "epoch_1",
	}

	// Put receipts for both policies
	if err := c.PutReceipt(ctx, receipt1); err != nil {
		t.Fatalf("PutReceipt 1: %v", err)
	}
	if err := c.PutReceipt(ctx, receipt2); err != nil {
		t.Fatalf("PutReceipt 2: %v", err)
	}

	// Both policies can retrieve their receipts independently
	gotR1, err := c.GetReceipt(ctx, policy1, "workload_common")
	if err != nil {
		t.Fatalf("GetReceipt 1: %v", err)
	}
	gotR2, err := c.GetReceipt(ctx, policy2, "workload_common")
	if err != nil {
		t.Fatalf("GetReceipt 2: %v", err)
	}

	// Both receipts point to the same artifact
	if gotR1.ArtifactKey != artKey || gotR2.ArtifactKey != artKey {
		t.Fatalf("receipts do not reference shared artifact key")
	}

	// Retrieve underlying artifact
	gotArt, err := c.GetArtifact(ctx, artKey)
	if err != nil {
		t.Fatalf("GetArtifact shared: %v", err)
	}
	if gotArt.Digest != art.Digest {
		t.Fatalf("artifact digest mismatch: got %q, want %q", gotArt.Digest, art.Digest)
	}
}

// Negative Controls and Mutant Tests:
// 1. Truncate a write: partial/truncated file rejected
// 2. Modify content under old digest: SHA-256 verification fails
// 3. Omit compatibility field: unmarshal/validation fails
// 4. Replay receipt for different workload: fails lookup or reports mismatch
// 5. Return oversized remote data: rejected by download limit

func TestMutantsAndNegativeControls(t *testing.T) {
	t.Run("TruncatedWriteRejected", func(t *testing.T) {
		tmpDir := t.TempDir()
		c, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		// Write half of JSON payload directly into artifacts dir
		truncatedJSON := []byte(`{"kind": 1, "digest": "abcdef", "raw": "AA`)
		key := "truncated-key"
		artFile := filepath.Join(tmpDir, "artifacts", key+".artifact")
		_ = os.WriteFile(artFile, truncatedJSON, 0o644)

		_, err = c.GetArtifact(context.Background(), key)
		if err == nil {
			t.Fatalf("expected error on truncated JSON artifact")
		}
		if !errors.Is(err, cudacache.ErrCorruptArtifact) {
			t.Errorf("expected ErrCorruptArtifact, got: %v", err)
		}
	})

	t.Run("OmitCompatibilityField", func(t *testing.T) {
		tmpDir := t.TempDir()
		c, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		rawBytes := []byte("simple raw code")
		h := sha256.Sum256(rawBytes)
		digest := hex.EncodeToString(h[:])

		// Missing target architecture and entry point
		omittedFieldJSON := fmt.Sprintf(`{"kind": 1, "digest": %q, "raw": %q}`, digest, rawBytes)
		key := "omitted-field-key"
		artFile := filepath.Join(tmpDir, "artifacts", key+".artifact")
		_ = os.WriteFile(artFile, []byte(omittedFieldJSON), 0o644)

		_, err = c.GetArtifact(context.Background(), key)
		if err == nil {
			t.Fatalf("expected error when compatibility fields are omitted")
		}
	})

	t.Run("ReplayReceiptDifferentWorkload", func(t *testing.T) {
		tmpDir := t.TempDir()
		c, err := cudacache.New(cudacache.Config{RootDir: tmpDir})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		policy := cudacache.SearchPolicyID{Strategy: "beam", Seed: 42}
		receipt := &cudacache.SearchReceipt{
			PolicyID:    policy,
			WorkloadKey: "workload_A",
			CandidateID: "cand_1",
			ArtifactKey: "art_1",
			Complete:    true,
		}
		if err := c.PutReceipt(context.Background(), receipt); err != nil {
			t.Fatalf("PutReceipt: %v", err)
		}

		// Querying for workload_B should yield ErrNotFound, preventing replay
		_, err = c.GetReceipt(context.Background(), policy, "workload_B")
		if !errors.Is(err, cudacache.ErrNotFound) {
			t.Fatalf("expected ErrNotFound for replayed receipt on different workload, got: %v", err)
		}
	})

	t.Run("RemoteOversizedDataRejected", func(t *testing.T) {
		reg := cudacache.NewMemoryRegistry()
		server := cudacache.NewTestServer(reg)
		defer server.Close()

		bigData := bytes.Repeat([]byte("A"), 10000)
		reg.PublishArtifact(context.Background(), "big-key", bigData)

		tmpDir := t.TempDir()
		c, err := cudacache.New(cudacache.Config{
			RootDir:                tmpDir,
			MaxRemoteDownloadBytes: 500, // Small limit
			Remote:                 cudacache.NewHTTPRemoteSource(server.URL, server.Client()),
		})
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		_, err = c.GetArtifact(context.Background(), "big-key")
		if !errors.Is(err, cudacache.ErrDownloadLimitExceeded) {
			t.Fatalf("expected ErrDownloadLimitExceeded, got: %v", err)
		}
	})
}
