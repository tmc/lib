package autotune

import (
	"crypto/sha256"
	"encoding/json"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestFleetSurrogateFitAndCrossArchPrediction(t *testing.T) {
	surrogate := NewFleetThroughputSurrogate(0.05)

	samples := []FleetTelemetrySample{
		{
			Arch:                     hopper.TargetSM90A,
			BatchM:                   16,
			NumStages:                4,
			RegsPerThread:            112,
			SharedBytes:              32768,
			SpillBytes:               0,
			LongScoreboardStallRatio: 0.14,
			BarrierStallRatio:        0.08,
			HoldoutSpeedup:           1.42,
		},
		{
			Arch:                     hopper.TargetSM100A,
			BatchM:                   32,
			NumStages:                4,
			RegsPerThread:            128,
			SharedBytes:              49152,
			SpillBytes:               0,
			LongScoreboardStallRatio: 0.11,
			BarrierStallRatio:        0.07,
			HoldoutSpeedup:           1.68,
		},
		{
			Arch:                     hopper.TargetSM120A,
			BatchM:                   16,
			NumStages:                5,
			RegsPerThread:            120,
			SharedBytes:              32768,
			SpillBytes:               0,
			LongScoreboardStallRatio: 0.16,
			BarrierStallRatio:        0.09,
			HoldoutSpeedup:           1.38,
		},
		{
			Arch:                     hopper.TargetSM121A,
			BatchM:                   32,
			NumStages:                4,
			RegsPerThread:            120,
			SharedBytes:              32768,
			SpillBytes:               0,
			LongScoreboardStallRatio: 0.13,
			BarrierStallRatio:        0.08,
			HoldoutSpeedup:           1.51,
		},
		{
			Arch:                     hopper.TargetSM107A,
			BatchM:                   64,
			NumStages:                4,
			RegsPerThread:            128,
			SharedBytes:              49152,
			SpillBytes:               0,
			LongScoreboardStallRatio: 0.09,
			BarrierStallRatio:        0.06,
			HoldoutSpeedup:           1.84,
		},
		// High-spill, high-stall negative receipts across architectures.
		{
			Arch:                     hopper.TargetSM121A,
			BatchM:                   32,
			NumStages:                10,
			RegsPerThread:            240,
			SharedBytes:              98304,
			SpillBytes:               512,
			LongScoreboardStallRatio: 0.62,
			BarrierStallRatio:        0.44,
			HoldoutSpeedup:           0.76,
		},
		{
			Arch:                     hopper.TargetSM100A,
			BatchM:                   32,
			NumStages:                12,
			RegsPerThread:            248,
			SharedBytes:              131072,
			SpillBytes:               768,
			LongScoreboardStallRatio: 0.58,
			BarrierStallRatio:        0.48,
			HoldoutSpeedup:           0.81,
		},
	}

	if err := surrogate.Fit(samples); err != nil {
		t.Fatalf("surrogate.Fit: %v", err)
	}
	if got := len(surrogate.Samples()); got != len(samples) {
		t.Fatalf("len(surrogate.Samples()) = %d, want %d", got, len(samples))
	}

	fastPred, err := surrogate.PredictSpeedup(FleetTelemetrySample{
		Arch:                     hopper.TargetSM121A,
		BatchM:                   32,
		NumStages:                4,
		RegsPerThread:            116,
		SharedBytes:              32768,
		SpillBytes:               0,
		LongScoreboardStallRatio: 0.12,
		BarrierStallRatio:        0.07,
	})
	if err != nil {
		t.Fatalf("PredictSpeedup(fast): %v", err)
	}

	spillingPred, err := surrogate.PredictSpeedup(FleetTelemetrySample{
		Arch:                     hopper.TargetSM121A,
		BatchM:                   32,
		NumStages:                10,
		RegsPerThread:            240,
		SharedBytes:              98304,
		SpillBytes:               640,
		LongScoreboardStallRatio: 0.60,
		BarrierStallRatio:        0.45,
	})
	if err != nil {
		t.Fatalf("PredictSpeedup(spilling): %v", err)
	}

	if fastPred <= spillingPred {
		t.Fatalf("expected fastPred (%.3f) > spillingPred (%.3f)", fastPred, spillingPred)
	}

	latFast, err := surrogate.PredictLatencyUs(FleetTelemetrySample{
		Arch:                     hopper.TargetSM107A,
		BatchM:                   64,
		NumStages:                4,
		RegsPerThread:            128,
		SharedBytes:              49152,
		SpillBytes:               0,
		LongScoreboardStallRatio: 0.10,
		BarrierStallRatio:        0.06,
	}, 100.0)
	if err != nil {
		t.Fatalf("PredictLatencyUs: %v", err)
	}
	if latFast >= 100.0 || latFast <= 0 {
		t.Fatalf("PredictLatencyUs = %.3f us, want in (0, 100.0)", latFast)
	}
}

func TestFleetSurrogateRankCandidatesZeroShot(t *testing.T) {
	surrogate := NewFleetThroughputSurrogate(0.05)

	// Candidate 0: Unfused 6-stage plan with no prefetch and high split-K barrier overhead.
	candSlow := &hopper.Plan{
		Name: "cand_slow_unprefetched",
		Kernels: []hopper.PlanKernel{
			{M: 4, GEMM: &hopper.SkinnyGEMM{Name: "g0", N: 256, K: 256, MaxM: 4, Prefetch: 1, SplitK: 4}, Args: []string{"x", "w0", "t0", "ws0"}},
			{M: 4, GEMM: &hopper.SkinnyGEMM{Name: "g1", N: 256, K: 256, MaxM: 4, Prefetch: 1, SplitK: 4}, Args: []string{"t0", "w1", "t1", "ws1"}},
			{M: 4, GEMM: &hopper.SkinnyGEMM{Name: "g2", N: 256, K: 256, MaxM: 4, Prefetch: 1, SplitK: 4}, Args: []string{"t1", "w2", "t2", "ws2"}},
			{M: 4, GEMM: &hopper.SkinnyGEMM{Name: "g3", N: 256, K: 256, MaxM: 4, Prefetch: 1, SplitK: 4}, Args: []string{"t2", "w3", "y", "ws3"}},
		},
	}

	// Candidate 1: Compact fused 2-stage plan with 3-stage register prefetch and SplitK=1.
	candFast := &hopper.Plan{
		Name: "cand_fast_prefetched",
		Kernels: []hopper.PlanKernel{
			{M: 16, GEMM: &hopper.SkinnyGEMM{Name: "g0_fast", N: 256, K: 256, MaxM: 16, Prefetch: 3, SplitK: 1, BlackwellFP4: true}, Args: []string{"x", "w0", "t0"}},
			{M: 16, GEMM: &hopper.SkinnyGEMM{Name: "g1_fast", N: 256, K: 256, MaxM: 16, Prefetch: 3, SplitK: 1, BlackwellFP4: true}, Args: []string{"t0", "w1", "y"}},
		},
	}

	// Candidate 2: Intermediate 3-stage plan with moderate prefetch.
	candMid := &hopper.Plan{
		Name: "cand_mid",
		Kernels: []hopper.PlanKernel{
			{M: 8, GEMM: &hopper.SkinnyGEMM{Name: "g0_mid", N: 256, K: 256, MaxM: 8, Prefetch: 2, SplitK: 2}, Args: []string{"x", "w0", "t0", "ws0"}},
			{M: 8, GEMM: &hopper.SkinnyGEMM{Name: "g1_mid", N: 256, K: 256, MaxM: 8, Prefetch: 2, SplitK: 2}, Args: []string{"t0", "w1", "t1", "ws1"}},
			{M: 8, GEMM: &hopper.SkinnyGEMM{Name: "g2_mid", N: 256, K: 256, MaxM: 8, Prefetch: 2, SplitK: 1}, Args: []string{"t1", "w2", "y"}},
		},
	}

	candidates := []*hopper.Plan{candSlow, candFast, candMid}

	for _, arch := range []hopper.Arch{
		hopper.TargetSM90A,
		hopper.TargetSM100A,
		hopper.TargetSM120A,
		hopper.TargetSM121A,
		hopper.TargetSM107A,
	} {
		order, scores, err := surrogate.RankCandidatesZeroShot(arch, candidates)
		if err != nil {
			t.Fatalf("RankCandidatesZeroShot(%s): %v", arch, err)
		}
		if len(order) != 3 || len(scores) != 3 {
			t.Fatalf("arch %s: got order=%v scores=%v, want len 3", arch, order, scores)
		}
		if order[0] != 1 {
			t.Fatalf("arch %s: expected candFast (index 1) ranked first, got order=%v scores=%v", arch, order, scores)
		}
		if order[2] != 0 {
			t.Fatalf("arch %s: expected candSlow (index 0) ranked last, got order=%v scores=%v", arch, order, scores)
		}
		if !(scores[0] >= scores[1] && scores[1] >= scores[2]) {
			t.Fatalf("arch %s: scores %v not sorted descending", arch, scores)
		}
	}
}

func TestFleetSurrogateMergeFederatedPGOCaches(t *testing.T) {
	localDir := t.TempDir()
	remoteDir := t.TempDir()
	local := NewVerifiedPGOCache(localDir)
	remote := NewVerifiedPGOCache(remoteDir)

	keyShared := PGOCacheKey{
		Arch:          hopper.TargetSM121A,
		DriverVersion: "570.00",
		CUDAVersion:   "12.8",
		WeightDigest:  "sha256:weights_v1",
		PlanKey:       "cross_l0_l1",
		ShapeBucket:   16,
	}
	keyNewRemote := PGOCacheKey{
		Arch:          hopper.TargetSM100A,
		DriverVersion: "570.00",
		CUDAVersion:   "12.8",
		WeightDigest:  "sha256:weights_v1",
		PlanKey:       "cross_l0_l1",
		ShapeBucket:   32,
	}
	keyMissingSrcDigest := PGOCacheKey{
		Arch:          hopper.TargetSM90A,
		DriverVersion: "570.00",
		CUDAVersion:   "12.8",
		WeightDigest:  "sha256:weights_v1",
		PlanKey:       "cross_l0_l1",
		ShapeBucket:   16,
	}

	srcSumShared := sha256.Sum256([]byte("__global__ void cross_l0_l1_mega() {}"))
	srcSumNew := sha256.Sum256([]byte("__global__ void cross_l0_l1_sm100_mega() {}"))

	cubinLocal := []byte("ELF_CUBIN_LOCAL_V1")
	cubinRemoteFaster := []byte("ELF_CUBIN_REMOTE_FASTER_V2")
	cubinRemoteNew := []byte("ELF_CUBIN_REMOTE_SM100")
	cubinMissingSrc := []byte("ELF_CUBIN_NO_SOURCE_SHA")

	telJSON, err := json.Marshal(FleetTelemetrySample{
		Arch:                     hopper.TargetSM121A,
		BatchM:                   16,
		NumStages:                5,
		RegsPerThread:            112,
		SharedBytes:              32768,
		SpillBytes:               0,
		LongScoreboardStallRatio: 0.14,
		BarrierStallRatio:        0.07,
		HoldoutSpeedup:           1.48,
	})
	if err != nil {
		t.Fatalf("json.Marshal telemetry: %v", err)
	}

	// Store initial entry in local with HoldoutRatio = 1.20.
	if err := local.Store(keyShared, PGOPromotionProof{
		Key:            keyShared,
		CandidateName:  "local_v1",
		ScheduleJSON:   telJSON,
		CubinSHA256:    sha256.Sum256(cubinLocal),
		SourceSHA256:   srcSumShared,
		ScreenBlocks:   6,
		HoldoutBlocks:  8,
		ScreenRatio:    1.22,
		HoldoutRatio:   1.20,
		OneSidedPValue: 0.0039,
		TokensMatched:  true,
	}, cubinLocal); err != nil {
		t.Fatalf("local.Store: %v", err)
	}

	// Store faster entry for keyShared in remote (HoldoutRatio = 1.48 > 1.20).
	if err := remote.Store(keyShared, PGOPromotionProof{
		Key:            keyShared,
		CandidateName:  "remote_v2_faster",
		ScheduleJSON:   telJSON,
		CubinSHA256:    sha256.Sum256(cubinRemoteFaster),
		SourceSHA256:   srcSumShared,
		ScreenBlocks:   6,
		HoldoutBlocks:  8,
		ScreenRatio:    1.50,
		HoldoutRatio:   1.48,
		OneSidedPValue: 0.0039,
		TokensMatched:  true,
	}, cubinRemoteFaster); err != nil {
		t.Fatalf("remote.Store(keyShared): %v", err)
	}

	// Store brand-new entry for keyNewRemote with valid SourceSHA256.
	if err := remote.Store(keyNewRemote, PGOPromotionProof{
		Key:            keyNewRemote,
		CandidateName:  "remote_sm100_winner",
		ScheduleJSON:   telJSON,
		CubinSHA256:    sha256.Sum256(cubinRemoteNew),
		SourceSHA256:   srcSumNew,
		ScreenBlocks:   6,
		HoldoutBlocks:  8,
		ScreenRatio:    1.65,
		HoldoutRatio:   1.61,
		OneSidedPValue: 0.0039,
		TokensMatched:  true,
	}, cubinRemoteNew); err != nil {
		t.Fatalf("remote.Store(keyNewRemote): %v", err)
	}

	// Store entry with zero SourceSHA256 in remote (must be skipped during merge).
	if err := remote.Store(keyMissingSrcDigest, PGOPromotionProof{
		Key:            keyMissingSrcDigest,
		CandidateName:  "remote_no_src_digest",
		CubinSHA256:    sha256.Sum256(cubinMissingSrc),
		SourceSHA256:   [32]byte{},
		ScreenBlocks:   6,
		HoldoutBlocks:  8,
		ScreenRatio:    1.40,
		HoldoutRatio:   1.35,
		OneSidedPValue: 0.0039,
		TokensMatched:  true,
	}, cubinMissingSrc); err != nil {
		t.Fatalf("remote.Store(keyMissingSrcDigest): %v", err)
	}

	added, updated, err := MergeFederatedPGOCaches(local, remote)
	if err != nil {
		t.Fatalf("MergeFederatedPGOCaches: %v", err)
	}
	if added != 1 || updated != 1 {
		t.Fatalf("MergeFederatedPGOCaches added=%d updated=%d, want added=1 updated=1", added, updated)
	}

	// Verify keyShared was updated to remote_v2_faster (HoldoutRatio = 1.48).
	gotShared, _, err := local.LoadVerified(keyShared)
	if err != nil {
		t.Fatalf("local.LoadVerified(keyShared): %v", err)
	}
	if gotShared.CandidateName != "remote_v2_faster" || gotShared.HoldoutRatio != 1.48 {
		t.Fatalf("gotShared = %+v, want CandidateName=remote_v2_faster HoldoutRatio=1.48", gotShared)
	}

	// Verify keyMissingSrcDigest was NOT merged into local.
	if _, _, err := local.LoadVerified(keyMissingSrcDigest); err == nil {
		t.Fatalf("expected keyMissingSrcDigest with zero SourceSHA256 to be rejected during federated merge")
	}

	// Re-merging the same remote cache should be a no-op (added=0, updated=0).
	added2, updated2, err := MergeFederatedPGOCaches(local, remote)
	if err != nil {
		t.Fatalf("second MergeFederatedPGOCaches: %v", err)
	}
	if added2 != 0 || updated2 != 0 {
		t.Fatalf("second MergeFederatedPGOCaches added=%d updated=%d, want 0, 0", added2, updated2)
	}

	// Ingest merged local cache into FleetThroughputSurrogate.
	surrogate := NewFleetThroughputSurrogate(0.05)
	count, err := surrogate.IngestVerifiedCache(local)
	if err != nil {
		t.Fatalf("IngestVerifiedCache: %v", err)
	}
	if count != 2 {
		t.Fatalf("IngestVerifiedCache count = %d, want 2", count)
	}
}
