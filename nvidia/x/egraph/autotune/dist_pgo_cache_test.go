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

func validDistPGOTestFixture(worldSize int) DistPGOCacheEntry {
	key := DistPGOCacheKey{
		Topology: DistTopologyKey{
			WorldSize:        worldSize,
			Arch:             "sm_100a",
			SwitchBWGBps:     900,
			SwitchLatencyUs:  0.8,
			TensorCoreTFLOPS: 2250,
			Interconnect:     "NVL72_NVLink5",
		},
		Arch:           "sm_100a",
		DriverVersion:  "580.173.02",
		CUDAVersion:    "13.4",
		WeightDigest:   "sha256:deepseek_v3_tp8_bf16",
		PlanKey:        "tp_down_proj_rmsnorm",
		ShapeBucket:    16,
		CollectiveRule: "allreduce_rmsnorm_to_multimem_ld_reduce_fused",
	}
	cubin := []byte("\x7fELF-synthetic-nvl72-multimem-ld-reduce-rmsnorm-cubin")
	source := []byte("__global__ void nvl72_multimem_rmsnorm() {}")
	traj := "sha256:greedy-tokens-128000-791-6864-315-9688-374-264-1176"

	votes := make([]RankVerificationVote, worldSize)
	for r := range worldSize {
		votes[r] = RankVerificationVote{
			Rank:              r,
			HoldoutVerified:   true,
			CoherenceClean:    true,
			TrajectoryDigest:  traj,
			LatencyUs:         18.2 + 0.1*float64(r),
			BaselineLatencyUs: 32.0,
			HoldoutRatio:      1.68,
			OneSidedPValue:    0.015625,
		}
	}
	return DistPGOCacheEntry{
		Key:               key,
		CandidateName:     "tp_down_proj_multimem_pull_v1",
		CollectiveRule:    "allreduce_rmsnorm_to_multimem_ld_reduce_fused",
		PlanSignature:     "sig_multimem_pull_tp8",
		ScheduleJSON:      []byte(`{"rule":"allreduce_rmsnorm_to_multimem_ld_reduce_fused","threads":128}`),
		PTX:               "\tmultimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%r0, %r1, %r2, %r3}, [%rd_mc];\n",
		Cubin:             cubin,
		CubinSHA256:       sha256.Sum256(cubin),
		SourceSHA256:      sha256.Sum256(source),
		BaselineLatencyUs: 32.0,
		MaxRankLatencyUs:  votes[worldSize-1].LatencyUs,
		TrajectoryDigest:  traj,
		RankVotes:         votes,
		PromotedAt:        time.Date(2026, 10, 31, 9, 0, 0, 0, time.UTC),
	}
}

func TestDistVerifiedPGOCacheQuorumPromotionAndReload(t *testing.T) {
	dir := t.TempDir()
	cache := NewDistVerifiedPGOCache(dir)
	entry := validDistPGOTestFixture(8)

	// Cold miss before recording.
	if _, hit, err := cache.LookupVerified(entry.Key); err != nil || hit {
		t.Fatalf("initial LookupVerified: hit=%v err=%v, want hit=false err=nil", hit, err)
	}

	if err := cache.RecordQuorumWinner(entry); err != nil {
		t.Fatalf("RecordQuorumWinner: %v", err)
	}

	// Reload from a fresh cache instance pointing at the same directory to
	// verify atomic disk persistence and full cryptographic + quorum re-verification.
	reloadedCache := NewDistVerifiedPGOCache(dir)
	got, hit, err := reloadedCache.LookupVerified(entry.Key)
	if err != nil {
		t.Fatalf("reloaded LookupVerified: %v", err)
	}
	if !hit {
		t.Fatal("expected hit=true after RecordQuorumWinner")
	}
	if got.CandidateName != entry.CandidateName {
		t.Errorf("CandidateName = %q, want %q", got.CandidateName, entry.CandidateName)
	}
	if got.CollectiveRule != entry.CollectiveRule {
		t.Errorf("CollectiveRule = %q, want %q", got.CollectiveRule, entry.CollectiveRule)
	}
	if !bytes.Equal(got.Cubin, entry.Cubin) {
		t.Errorf("Cubin mismatch: got %d bytes, want %d bytes", len(got.Cubin), len(entry.Cubin))
	}
	if got.CubinSHA256 != entry.CubinSHA256 {
		t.Errorf("CubinSHA256 = %x, want %x", got.CubinSHA256, entry.CubinSHA256)
	}
	if len(got.RankVotes) != 8 {
		t.Fatalf("len(RankVotes) = %d, want 8", len(got.RankVotes))
	}
	if !(got.MaxRankLatencyUs < got.BaselineLatencyUs) {
		t.Errorf("MaxRankLatencyUs=%v >= BaselineLatencyUs=%v", got.MaxRankLatencyUs, got.BaselineLatencyUs)
	}

	// Ensure no temporary files remain in dir after atomic write.
	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("dir file count = %d, want 2 (.dist.json and .dist.cubin)", len(files))
	}
}

func TestDistVerifiedPGOCacheQuorumRejections(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(e *DistPGOCacheEntry)
		wantErr error
	}{
		{
			name: "IncompleteRankQuorumMissingVote",
			mutate: func(e *DistPGOCacheEntry) {
				e.RankVotes = e.RankVotes[:len(e.RankVotes)-1]
			},
			wantErr: ErrDistPGOQuorumIncomplete,
		},
		{
			name: "ExtraRankVoteExceedsWorldSize",
			mutate: func(e *DistPGOCacheEntry) {
				extra := e.RankVotes[0]
				extra.Rank = len(e.RankVotes)
				e.RankVotes = append(slices.Clone(e.RankVotes), extra)
			},
			wantErr: ErrDistPGOQuorumIncomplete,
		},
		{
			name: "SingleRankHoldoutUnverified",
			mutate: func(e *DistPGOCacheEntry) {
				e.RankVotes[5].HoldoutVerified = false
			},
			wantErr: ErrDistPGOHoldoutUnverified,
		},
		{
			name: "SingleRankCoherenceDirty",
			mutate: func(e *DistPGOCacheEntry) {
				e.RankVotes[3].CoherenceClean = false
			},
			wantErr: ErrDistPGOCoherenceFailed,
		},
		{
			name: "EmptyTrajectoryDigestOnRank",
			mutate: func(e *DistPGOCacheEntry) {
				e.RankVotes[2].TrajectoryDigest = "   "
			},
			wantErr: ErrDistPGOTrajectoryMismatch,
		},
		{
			name: "DivergentTrajectoryDigestAcrossRanks",
			mutate: func(e *DistPGOCacheEntry) {
				e.RankVotes[7].TrajectoryDigest = "sha256:divergent-rank-7-tokens"
			},
			wantErr: ErrDistPGOTrajectoryMismatch,
		},
		{
			name: "StragglerMaxRankLatencyExceedsBaseline",
			mutate: func(e *DistPGOCacheEntry) {
				e.MaxRankLatencyUs = e.BaselineLatencyUs + 0.5
			},
			wantErr: ErrDistPGOStragglerRegression,
		},
		{
			name: "StragglerMaxRankLatencyEqualsBaseline",
			mutate: func(e *DistPGOCacheEntry) {
				e.MaxRankLatencyUs = e.BaselineLatencyUs
			},
			wantErr: ErrDistPGOStragglerRegression,
		},
		{
			name: "StragglerRankVoteLatencyExceedsBaseline",
			mutate: func(e *DistPGOCacheEntry) {
				// Even if MaxRankLatencyUs was optimistically set low, a single
				// straggler rank vote above baseline must be rejected.
				e.MaxRankLatencyUs = 19.0
				e.RankVotes[6].LatencyUs = e.BaselineLatencyUs + 1.2
			},
			wantErr: ErrDistPGOStragglerRegression,
		},
		{
			name: "CubinDigestMismatch",
			mutate: func(e *DistPGOCacheEntry) {
				e.Cubin = append([]byte(nil), e.Cubin...)
				e.Cubin[0] ^= 0xff
			},
			wantErr: ErrPGODigestMismatch,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cache := NewDistVerifiedPGOCache(dir)
			entry := validDistPGOTestFixture(8)
			tc.mutate(&entry)

			err := cache.RecordQuorumWinner(entry)
			if err == nil {
				t.Fatalf("RecordQuorumWinner succeeded, want error %v", tc.wantErr)
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("RecordQuorumWinner error = %v, want errors.Is(_, %v)", err, tc.wantErr)
			}

			keys, listErr := cache.List()
			if listErr != nil {
				t.Fatalf("List: %v", listErr)
			}
			if len(keys) != 0 {
				t.Fatalf("rejected RecordQuorumWinner left %d entries in cache", len(keys))
			}
		})
	}
}

func TestDistVerifiedPGOCacheDiskTamperAndTopologyIsolation(t *testing.T) {
	dir := t.TempDir()
	cache := NewDistVerifiedPGOCache(dir)

	tp8Entry := validDistPGOTestFixture(8)
	nvl72Entry := validDistPGOTestFixture(72)
	nvl72Entry.Key.Topology.SwitchBWGBps = 1800
	nvl72Entry.CandidateName = "tp72_rubin_fabric_push_v1"

	if tp8Entry.Key.CacheID() == nvl72Entry.Key.CacheID() {
		t.Fatal("expected distinct CacheIDs for 8-rank vs 72-rank topologies")
	}

	if err := cache.RecordQuorumWinner(tp8Entry); err != nil {
		t.Fatalf("RecordQuorumWinner(tp8): %v", err)
	}
	if err := cache.RecordQuorumWinner(nvl72Entry); err != nil {
		t.Fatalf("RecordQuorumWinner(nvl72): %v", err)
	}

	keys, err := cache.List()
	if err != nil || len(keys) != 2 {
		t.Fatalf("List: count=%d err=%v, want 2", len(keys), err)
	}

	// Tamper with the on-disk JSON of tp8Entry to introduce a trajectory mismatch
	// and verify LookupVerified rejects the corrupted entry.
	jsonPath := filepath.Join(dir, tp8Entry.Key.CacheID()+".dist.json")
	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	var tampered DistPGOCacheEntry
	if err := json.Unmarshal(raw, &tampered); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	tampered.RankVotes[4].TrajectoryDigest = "sha256:corrupted-on-disk"
	tamperedBytes, err := json.Marshal(tampered)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := os.WriteFile(jsonPath, tamperedBytes, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if _, _, err := cache.LookupVerified(tp8Entry.Key); !errors.Is(err, ErrDistPGOTrajectoryMismatch) {
		t.Fatalf("LookupVerified on tampered JSON = %v, want %v", err, ErrDistPGOTrajectoryMismatch)
	}

	// Verify the 72-rank entry remains intact and valid.
	got72, hit, err := cache.LookupVerified(nvl72Entry.Key)
	if err != nil || !hit || got72.CandidateName != nvl72Entry.CandidateName {
		t.Fatalf("LookupVerified(nvl72): hit=%v err=%v cand=%q", hit, err, got72.CandidateName)
	}

	if err := cache.Invalidate(nvl72Entry.Key); err != nil {
		t.Fatalf("Invalidate(nvl72): %v", err)
	}
	if _, hit, err := cache.LookupVerified(nvl72Entry.Key); err != nil || hit {
		t.Fatalf("LookupVerified after Invalidate: hit=%v err=%v, want false, nil", hit, err)
	}
}
