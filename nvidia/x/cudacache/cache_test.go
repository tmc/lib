package cudacache_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
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

		ctx := context.Background()
		rawBytes := []byte(".version 8.0\n.target sm_121a\n.address_size 64\n.entry compat_kernel(.param .u64 ptr) { ret; }\n")
		h := sha256.Sum256(rawBytes)
		digest := hex.EncodeToString(h[:])
		rawBase64 := base64.StdEncoding.EncodeToString(rawBytes)

		schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "data", Type: "ptr", IsPointer: true})
		if err != nil {
			t.Fatalf("NewSchema: %v", err)
		}

		baseMap := func() map[string]any {
			return map[string]any{
				"schema_version": cudacache.ArtifactSchemaVersion,
				"kind":           int(cudakernel.KindPTX),
				"digest":         digest,
				"raw":            rawBase64,
				"entry_point":    "compat_kernel",
				"schema":         schema,
				"target":         "sm_121a",
			}
		}

		// Passing twin: all required compatibility fields present with valid base64 raw.
		twinData, err := json.Marshal(baseMap())
		if err != nil {
			t.Fatalf("Marshal twin: %v", err)
		}
		twinKey := "passing-twin-key"
		twinFile := filepath.Join(tmpDir, "artifacts", twinKey+".artifact")
		if err := os.WriteFile(twinFile, twinData, 0o644); err != nil {
			t.Fatalf("WriteFile twin: %v", err)
		}
		got, err := c.GetArtifact(ctx, twinKey)
		if err != nil {
			t.Fatalf("passing twin rejected: %v", err)
		}
		if got.Target != "sm_121a" || got.EntryPoint != "compat_kernel" || !bytes.Equal(got.Raw, rawBytes) {
			t.Fatalf("passing twin mismatch: got target=%q entry=%q raw=%q", got.Target, got.EntryPoint, got.Raw)
		}

		// Single-field compatibility omissions derived from the passing twin.
		for _, omitted := range []string{"target", "entry_point"} {
			t.Run("omit_"+omitted, func(t *testing.T) {
				m := baseMap()
				delete(m, omitted)
				mutatedData, err := json.Marshal(m)
				if err != nil {
					t.Fatalf("Marshal mutated (%s): %v", omitted, err)
				}
				key := "omitted-" + omitted + "-key"
				artFile := filepath.Join(tmpDir, "artifacts", key+".artifact")
				if err := os.WriteFile(artFile, mutatedData, 0o644); err != nil {
					t.Fatalf("WriteFile mutated (%s): %v", omitted, err)
				}

				_, err = c.GetArtifact(ctx, key)
				if !errors.Is(err, cudacache.ErrCorruptArtifact) {
					t.Fatalf("omitting %q: expected ErrCorruptArtifact, got: %v", omitted, err)
				}
			})
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

func TestEntrySchemaVersion(t *testing.T) {
	ctx := context.Background()
	art := newTestArtifact(t, "versioned_kernel", "sm_121a")
	artBytes, err := cudacache.MarshalArtifact(art)
	if err != nil {
		t.Fatalf("MarshalArtifact: %v", err)
	}
	var baseArt map[string]any
	if err := json.Unmarshal(artBytes, &baseArt); err != nil {
		t.Fatalf("Unmarshal artifact map: %v", err)
	}
	if gotVer, ok := baseArt["schema_version"].(float64); !ok || int(gotVer) != cudacache.ArtifactSchemaVersion {
		t.Fatalf("persisted artifact schema_version = %v, want %d", baseArt["schema_version"], cudacache.ArtifactSchemaVersion)
	}

	policy := cudacache.SearchPolicyID{Strategy: "beam", Seed: 7, ScoringEpoch: "epoch_v1"}
	receipt := &cudacache.SearchReceipt{
		PolicyID:    policy,
		WorkloadKey: "workload_schema_v1",
		CandidateID: "cand_v1",
		ArtifactKey: art.Digest,
		Complete:    true,
		TimingKind:  cudatrial.TimingWallElapsed,
		Timings:     map[string]int64{string(cudatrial.TimingWallElapsed): 42000},
		RankScore:   0.91,
		Epoch:       "epoch_v1",
	}
	receiptBytes, err := cudacache.MarshalReceipt(receipt)
	if err != nil {
		t.Fatalf("MarshalReceipt: %v", err)
	}
	var baseReceipt map[string]any
	if err := json.Unmarshal(receiptBytes, &baseReceipt); err != nil {
		t.Fatalf("Unmarshal receipt map: %v", err)
	}
	if gotVer, ok := baseReceipt["schema_version"].(float64); !ok || int(gotVer) != cudacache.ReceiptSchemaVersion {
		t.Fatalf("persisted receipt schema_version = %v, want %d", baseReceipt["schema_version"], cudacache.ReceiptSchemaVersion)
	}

	intPtr := func(v int) *int { return &v }
	cases := []struct {
		name    string
		version *int
		wantOK  bool
	}{
		{name: "current_v1", version: intPtr(1), wantOK: true},
		{name: "missing_version", version: nil, wantOK: false},
		{name: "zero_version", version: intPtr(0), wantOK: false},
		{name: "negative_version", version: intPtr(-1), wantOK: false},
		{name: "unknown_future_v2", version: intPtr(2), wantOK: false},
		{name: "unknown_future_v99", version: intPtr(99), wantOK: false},
	}

	mutateVersion := func(src map[string]any, v *int) []byte {
		m := make(map[string]any, len(src))
		for k, val := range src {
			m[k] = val
		}
		if v == nil {
			delete(m, "schema_version")
		} else {
			m["schema_version"] = *v
		}
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatalf("Marshal mutated map: %v", err)
		}
		return b
	}

	for _, tc := range cases {
		t.Run("artifact_disk/"+tc.name, func(t *testing.T) {
			root := t.TempDir()
			c, err := cudacache.New(cudacache.Config{RootDir: root})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			key := "art-" + tc.name
			path := filepath.Join(root, "artifacts", key+".artifact")
			if err := os.WriteFile(path, mutateVersion(baseArt, tc.version), 0o644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}
			got, err := c.GetArtifact(ctx, key)
			if tc.wantOK {
				if err != nil || got == nil || got.Digest != art.Digest {
					t.Fatalf("expected valid v1 artifact load, got art=%v err=%v", got, err)
				}
				return
			}
			if !errors.Is(err, cudacache.ErrCorruptArtifact) {
				t.Fatalf("expected ErrCorruptArtifact for %s, got: %v", tc.name, err)
			}
			qEntries, err := os.ReadDir(filepath.Join(root, "quarantine"))
			if err != nil || len(qEntries) != 1 {
				t.Fatalf("expected 1 quarantined file for %s, got %d (err=%v)", tc.name, len(qEntries), err)
			}
		})

		t.Run("artifact_remote/"+tc.name, func(t *testing.T) {
			reg := cudacache.NewMemoryRegistry()
			key := "remote-art-" + tc.name
			_ = reg.PublishArtifact(ctx, key, mutateVersion(baseArt, tc.version))
			c, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: reg})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			got, err := c.GetArtifact(ctx, key)
			if tc.wantOK {
				if err != nil || got == nil || got.Digest != art.Digest {
					t.Fatalf("expected valid v1 remote artifact, got art=%v err=%v", got, err)
				}
				return
			}
			if !errors.Is(err, cudacache.ErrCorruptArtifact) {
				t.Fatalf("expected ErrCorruptArtifact for remote %s, got: %v", tc.name, err)
			}
		})

		t.Run("receipt_disk/"+tc.name, func(t *testing.T) {
			root := t.TempDir()
			c, err := cudacache.New(cudacache.Config{RootDir: root})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			rKey := cudacache.SearchReceiptKey(policy, receipt.WorkloadKey)
			path := filepath.Join(root, "receipts", rKey+".receipt")
			if err := os.WriteFile(path, mutateVersion(baseReceipt, tc.version), 0o644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}
			got, err := c.GetReceipt(ctx, policy, receipt.WorkloadKey)
			if tc.wantOK {
				if err != nil || got == nil || got.CandidateID != receipt.CandidateID {
					t.Fatalf("expected valid v1 receipt load, got receipt=%v err=%v", got, err)
				}
				return
			}
			if !errors.Is(err, cudacache.ErrCorruptReceipt) {
				t.Fatalf("expected ErrCorruptReceipt for %s, got: %v", tc.name, err)
			}
		})

		t.Run("receipt_remote/"+tc.name, func(t *testing.T) {
			reg := cudacache.NewMemoryRegistry()
			rKey := cudacache.SearchReceiptKey(policy, receipt.WorkloadKey)
			_ = reg.PublishReceipt(ctx, rKey, mutateVersion(baseReceipt, tc.version))
			c, err := cudacache.New(cudacache.Config{RootDir: t.TempDir(), Remote: reg})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			got, err := c.GetReceipt(ctx, policy, receipt.WorkloadKey)
			if tc.wantOK {
				if err != nil || got == nil || got.CandidateID != receipt.CandidateID {
					t.Fatalf("expected valid v1 remote receipt, got receipt=%v err=%v", got, err)
				}
				return
			}
			if !errors.Is(err, cudacache.ErrCorruptReceipt) {
				t.Fatalf("expected ErrCorruptReceipt for remote %s, got: %v", tc.name, err)
			}
		})
	}
}

func ExampleMarshalArtifact() {
	schema, _ := cudakernel.NewSchema(cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true})
	art, _ := cudakernel.NewArtifact(
		cudakernel.KindPTX,
		[]byte(".version 8.0\n.target sm_121a\n.address_size 64\n.entry k(.param .u64 p) { ret; }\n"),
		"k",
		schema,
		"sm_121a",
	)
	data, err := cudacache.MarshalArtifact(art)
	var envelope struct {
		SchemaVersion int `json:"schema_version"`
	}
	_ = json.Unmarshal(data, &envelope)
	fmt.Println(err == nil, envelope.SchemaVersion == cudacache.ArtifactSchemaVersion)
	// Output:
	// true true
}

func ExampleMarshalReceipt() {
	receipt := &cudacache.SearchReceipt{
		PolicyID:    cudacache.SearchPolicyID{Strategy: "beam", Seed: 1},
		WorkloadKey: "workload_1",
		CandidateID: "cand_1",
		ArtifactKey: "art_1",
	}
	data, err := cudacache.MarshalReceipt(receipt)
	var envelope struct {
		SchemaVersion int `json:"schema_version"`
	}
	_ = json.Unmarshal(data, &envelope)
	fmt.Println(err == nil, envelope.SchemaVersion == cudacache.ReceiptSchemaVersion)
	// Output:
	// true true
}
