package model

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

func buildDistTPBlockGraph() (*egraph.EGraph, egraph.Id) {
	g := egraph.New()
	leaf := func(s string) egraph.Id { return g.Add(egraph.NewLeaf(s)) }
	mm := g.Add(egraph.NewNode(nvl72.DistGemm, leaf("x"), leaf("w")))
	ar := g.Add(egraph.NewNode(nvl72.DistAllReduceSum, mm))
	add := g.Add(egraph.NewNode(nvl72.DistAdd, ar, leaf("r")))
	root := g.Add(egraph.NewNode(nvl72.DistRMSNorm, add, leaf("g")))
	return g, root
}

func buildDistAGMatmulGraph() (*egraph.EGraph, egraph.Id) {
	g := egraph.New()
	leaf := func(s string) egraph.Id { return g.Add(egraph.NewLeaf(s)) }
	ag := g.Add(egraph.NewNode(nvl72.DistAllGather, leaf("x")))
	root := g.Add(egraph.NewNode(nvl72.DistGemm, ag, leaf("w")))
	return g, root
}

func buildDistMatmulRSGraph() (*egraph.EGraph, egraph.Id) {
	g := egraph.New()
	leaf := func(s string) egraph.Id { return g.Add(egraph.NewLeaf(s)) }
	mm := g.Add(egraph.NewNode(nvl72.DistGemm, leaf("x"), leaf("w")))
	root := g.Add(egraph.NewNode(nvl72.DistReduceScatter, mm))
	return g, root
}

func newTestMultiRankCoordinator(t *testing.T, ctx context.Context, worldSize int, kernelName string) (*MultiRankEpochCoordinator, []*cudahotswap.Service) {
	t.Helper()
	services := make([]*cudahotswap.Service, worldSize)
	trackers := make([]*MegakernelEpochTracker, worldSize)
	for r := range worldSize {
		svc, err := cudahotswap.NewService(&flightTestLoader{})
		if err != nil {
			t.Fatalf("cudahotswap.NewService(rank %d): %v", r, err)
		}
		t.Cleanup(func() {
			_ = svc.Close(ctx)
		})
		services[r] = svc
		trk, err := NewMegakernelEpochTracker(EpochConfig{}, svc, kernelName, 132)
		if err != nil {
			t.Fatalf("NewMegakernelEpochTracker(rank %d): %v", r, err)
		}
		trackers[r] = trk
	}
	coord, err := NewMultiRankEpochCoordinator(worldSize, kernelName, services, trackers)
	if err != nil {
		t.Fatalf("NewMultiRankEpochCoordinator: %v", err)
	}
	return coord, services
}

func TestDistAutonomousSearchTopologyAwareBeam(t *testing.T) {
	compG, compRoot, _, _ := buildAutonomousTestGraph(t)

	// 1. Decode regime (BatchTokens=16) vs Prefill regime (BatchTokens=4096) on
	// joint compute + TP collective graph.
	regimes := []struct {
		name      string
		batch     int
		target    string
		wantRule  nvl72.DistRewriteKind
		wantInPTX string
	}{
		{
			name:      "DecodeM16_PullMultimemLdReduce",
			batch:     16,
			target:    "sm_100a",
			wantRule:  nvl72.RulePullMultimemLdReduce,
			wantInPTX: "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2",
		},
		{
			name:      "PrefillM4096_PushTileFabricRed_Blackwell",
			batch:     4096,
			target:    "sm_100a",
			wantRule:  nvl72.RuleTilePushFabricRed,
			wantInPTX: "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group",
		},
		{
			name:      "PrefillM4096_PushTileFabricRed_Rubin",
			batch:     4096,
			target:    "sm_107a",
			wantRule:  nvl72.RuleTilePushFabricRed,
			wantInPTX: "fabric.try_red.async.tensor.2d.multimem",
		},
	}

	for _, tc := range regimes {
		t.Run(tc.name, func(t *testing.T) {
			distG, distRoot := buildDistTPBlockGraph()
			beam, err := SearchTopologyAwareBeam(TopologyBeamConfig{
				Name:         "tp_layer",
				ComputeGraph: compG,
				ComputeRoot:  compRoot,
				DistGraph:    distG,
				DistRoot:     distRoot,
				OutDType:     hopper.BF16,
				CostConfig:   hopper.CostConfig{Device: "NVIDIA B200", SMs: 148},
				DistParams: nvl72.DistFusionCostParams{
					WorldSize:        8,
					SwitchBWGBps:     nvl72.NVL72NVLink5UniGBps,
					SwitchLatencyUs:  0.8,
					TensorCoreTFLOPS: nvl72.NVL72BlackwellTensorTFLOPS,
					BatchTokens:      tc.batch,
					Target:           tc.target,
				},
				BeamWidth: 6,
			})
			if err != nil {
				t.Fatalf("SearchTopologyAwareBeam: %v", err)
			}
			if len(beam) < 2 {
				t.Fatalf("len(beam) = %d, want >= 2", len(beam))
			}

			top := beam[0]
			if top.CollectiveRule != tc.wantRule {
				t.Fatalf("top.CollectiveRule = %q, want %q", top.CollectiveRule, tc.wantRule)
			}
			if !top.NumericallyVerified || top.MaxRelDiff > 1e-4 {
				t.Fatalf("top candidate not numerically verified: verified=%v maxRelDiff=%g", top.NumericallyVerified, top.MaxRelDiff)
			}
			if !strings.Contains(top.PTX, tc.wantInPTX) {
				t.Fatalf("top.PTX missing %q:\n%s", tc.wantInPTX, top.PTX)
			}
			if !(top.DistLatencyUs < top.UnfusedDistLatencyUs) {
				t.Fatalf("expected fused DistLatencyUs %.2f < UnfusedDistLatencyUs %.2f", top.DistLatencyUs, top.UnfusedDistLatencyUs)
			}
			if top.ComputeCandidate.Plan == nil {
				t.Fatal("expected ComputeCandidate.Plan to be populated from ExtractStructuralPlanBeam")
			}
			for i := 1; i < len(beam); i++ {
				if beam[i].TotalDistLatencyUs < beam[i-1].TotalDistLatencyUs {
					t.Fatalf("beam not sorted by TotalDistLatencyUs at index %d: %.2f < %.2f",
						i, beam[i].TotalDistLatencyUs, beam[i-1].TotalDistLatencyUs)
				}
			}
		})
	}

	// 2. Verify RuleAGMatmulToRingPrefetch and RuleMatmulRSAssociative are
	// discovered, numerically verified via EvalDistExpr, and ranked above unfused.
	t.Run("AllGatherRingPrefetchAndReduceScatterTiled", func(t *testing.T) {
		agG, agRoot := buildDistAGMatmulGraph()
		agBeam, err := SearchTopologyAwareBeam(TopologyBeamConfig{
			Name:      "tp_ag_matmul",
			DistGraph: agG,
			DistRoot:  agRoot,
			DistParams: nvl72.DistFusionCostParams{
				WorldSize:        8,
				SwitchBWGBps:     900,
				SwitchLatencyUs:  0.8,
				TensorCoreTFLOPS: 2250,
				BatchTokens:      512,
			},
			BeamWidth: 4,
		})
		if err != nil {
			t.Fatalf("SearchTopologyAwareBeam(AG): %v", err)
		}
		if agBeam[0].CollectiveRule != nvl72.RuleAGMatmulToRingPrefetch {
			t.Fatalf("AG top rule = %q, want %q", agBeam[0].CollectiveRule, nvl72.RuleAGMatmulToRingPrefetch)
		}
		if !agBeam[0].NumericallyVerified {
			t.Fatal("expected AG candidate to be numerically verified")
		}

		rsG, rsRoot := buildDistMatmulRSGraph()
		rsBeam, err := SearchTopologyAwareBeam(TopologyBeamConfig{
			Name:      "tp_matmul_rs",
			DistGraph: rsG,
			DistRoot:  rsRoot,
			DistParams: nvl72.DistFusionCostParams{
				WorldSize:        8,
				SwitchBWGBps:     900,
				SwitchLatencyUs:  0.8,
				TensorCoreTFLOPS: 2250,
				BatchTokens:      512,
			},
			BeamWidth: 4,
		})
		if err != nil {
			t.Fatalf("SearchTopologyAwareBeam(RS): %v", err)
		}
		if rsBeam[0].CollectiveRule != nvl72.RuleMatmulRSAssociative {
			t.Fatalf("RS top rule = %q, want %q", rsBeam[0].CollectiveRule, nvl72.RuleMatmulRSAssociative)
		}
		if !rsBeam[0].NumericallyVerified {
			t.Fatal("expected RS candidate to be numerically verified")
		}
	})
}

func TestDistAutonomousMultiRankEpochCoordinator(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()
	const worldSize = 8
	const kernelName = "tp_decode_megakernel"

	coord, services := newTestMultiRankCoordinator(t, ctx, worldSize, kernelName)

	traj := "sha256:consistent-tokens-across-8-ranks"
	makeVotes := func(latUs float64) []autotune.RankVerificationVote {
		votes := make([]autotune.RankVerificationVote, worldSize)
		for r := range worldSize {
			votes[r] = autotune.RankVerificationVote{
				Rank:             r,
				HoldoutVerified:  true,
				CoherenceClean:   true,
				TrajectoryDigest: traj,
				LatencyUs:        latUs + 0.05*float64(r),
			}
		}
		return votes
	}

	// 1. Coordinate first barrier-aligned swap at epoch 5 (v1 baseline -> v2 pull multimem).
	swapV2, err := coord.CoordinateHotSwap(ctx, MultiRankSwapRequest{
		SwitchAtEpoch:     5,
		CandidateName:     "cand_pull_multimem_v2",
		CollectiveRule:    nvl72.RulePullMultimemLdReduce,
		Source:            cudahotswap.Source{Code: `extern "C" __global__ void tp_decode_megakernel(float* out) { if (out) out[0] = 2.0f; }`, Entry: kernelName},
		RankVotes:         makeVotes(16.0),
		BaselineLatencyUs: 28.0,
	})
	if err != nil {
		t.Fatalf("CoordinateHotSwap v2: %v", err)
	}
	if !swapV2.Committed || swapV2.RolledBack || swapV2.SwitchAtEpoch != 5 {
		t.Fatalf("unexpected swapV2 result: %+v", swapV2)
	}
	if coord.ActiveRule != nvl72.RulePullMultimemLdReduce || coord.Epoch != 5 {
		t.Fatalf("coord ActiveRule=%q Epoch=%d, want %q, 5", coord.ActiveRule, coord.Epoch, nvl72.RulePullMultimemLdReduce)
	}
	for r, svc := range services {
		st, sErr := svc.Status(kernelName)
		if sErr != nil || st.Version != 2 {
			t.Fatalf("rank %d version after v2 commit = %d (err=%v), want 2", r, st.Version, sErr)
		}
		if coord.RankEpochs[r] != 5 {
			t.Fatalf("rank %d epoch = %d, want 5", r, coord.RankEpochs[r])
		}
	}

	// 2. Reject a swap when rank 3 has already advanced past target switchAtEpoch.
	if err := coord.SetRankEpoch(3, 9); err != nil {
		t.Fatalf("SetRankEpoch: %v", err)
	}
	_, err = coord.CoordinateHotSwap(ctx, MultiRankSwapRequest{
		SwitchAtEpoch:  7, // < rank 3's epoch 9
		CandidateName:  "cand_stale_barrier",
		CollectiveRule: nvl72.RuleTilePushFabricRed,
	})
	if !errors.Is(err, ErrEpochBarrierMismatch) {
		t.Fatalf("expected ErrEpochBarrierMismatch, got %v", err)
	}
	_ = coord.SetRankEpoch(3, 5)

	// 3. Simulate a single-rank post-swap check failure on rank 6 at epoch 10:
	// verify atomic rollback restores ALL 8 ranks to v2 and keeps ActiveRule on
	// RulePullMultimemLdReduce so no rank executes a mismatched collective protocol.
	rank6Err := errors.New("rank 6 nvlink multicast barrier timeout")
	swapFailed, err := coord.CoordinateHotSwap(ctx, MultiRankSwapRequest{
		SwitchAtEpoch:     10,
		CandidateName:     "cand_push_tile_v3_broken_rank6",
		CollectiveRule:    nvl72.RuleTilePushFabricRed,
		Source:            cudahotswap.Source{Code: `extern "C" __global__ void tp_decode_megakernel(float* out) { if (out) out[0] = 3.0f; }`, Entry: kernelName},
		RankVotes:         makeVotes(14.0),
		BaselineLatencyUs: 28.0,
		PostSwapRankCheck: func(_ context.Context, rank int, _ cudahotswap.State) error {
			if rank == 6 {
				return rank6Err
			}
			return nil
		},
	})
	if !errors.Is(err, ErrMultiRankSwapVerificationFailed) || !errors.Is(err, rank6Err) {
		t.Fatalf("expected ErrMultiRankSwapVerificationFailed wrapping rank6Err, got %v", err)
	}
	if swapFailed == nil || swapFailed.Committed || !swapFailed.RolledBack || swapFailed.FailedRank != 6 {
		t.Fatalf("unexpected swapFailed result: %+v", swapFailed)
	}
	if coord.ActiveRule != nvl72.RulePullMultimemLdReduce || coord.Epoch != 5 {
		t.Fatalf("after rollback, coord ActiveRule=%q Epoch=%d, want %q, 5", coord.ActiveRule, coord.Epoch, nvl72.RulePullMultimemLdReduce)
	}
	for r, svc := range services {
		st, sErr := svc.Status(kernelName)
		if sErr != nil || st.Version != 2 {
			t.Fatalf("rank %d version after atomic rollback = %d (err=%v), want 2", r, st.Version, sErr)
		}
	}

	// 4. Commit a valid v3 upgrade at epoch 12, then test explicit RollbackLastCommit.
	swapV3, err := coord.CoordinateHotSwap(ctx, MultiRankSwapRequest{
		SwitchAtEpoch:     12,
		CandidateName:     "cand_push_tile_v3_clean",
		CollectiveRule:    nvl72.RuleTilePushFabricRed,
		Source:            cudahotswap.Source{Code: `extern "C" __global__ void tp_decode_megakernel(float* out) { if (out) out[0] = 30.0f; }`, Entry: kernelName},
		RankVotes:         makeVotes(13.5),
		BaselineLatencyUs: 28.0,
	})
	if err != nil || !swapV3.Committed {
		t.Fatalf("CoordinateHotSwap v3 clean: res=%+v err=%v", swapV3, err)
	}
	for r, svc := range services {
		st, _ := svc.Status(kernelName)
		if st.Version != 5 { // 1: baseline, 2: pull, 3: failed push, 4: rollback select rev, 5: clean push
			t.Fatalf("rank %d version = %d, want 5", r, st.Version)
		}
	}

	restoredStates, err := coord.RollbackLastCommit(ctx)
	if err != nil {
		t.Fatalf("RollbackLastCommit: %v", err)
	}
	for r, st := range restoredStates {
		if st.Version != 2 {
			t.Fatalf("rank %d restored version = %d, want immediate prior committed v2", r, st.Version)
		}
	}
	if coord.ActiveRule != nvl72.RulePullMultimemLdReduce || coord.Epoch != 5 {
		t.Fatalf("after RollbackLastCommit, ActiveRule=%q Epoch=%d, want %q, 5", coord.ActiveRule, coord.Epoch, nvl72.RulePullMultimemLdReduce)
	}
}

func TestDistAutonomousPipelineEndToEnd(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "gpu-lock"))
	ctx := context.Background()
	const worldSize = 8
	const kernelName = "llama_nvl72_tp8_decode"

	compG, compRoot, _, _ := buildAutonomousTestGraph(t)
	distG, distRoot := buildDistTPBlockGraph()

	cacheDir := filepath.Join(t.TempDir(), "dist-pgo-cache")
	distCache := autotune.NewDistVerifiedPGOCache(cacheDir)
	cacheKey := autotune.DistPGOCacheKey{
		Topology: autotune.DistTopologyKey{
			WorldSize:        worldSize,
			Arch:             "sm_100a",
			SwitchBWGBps:     nvl72.NVL72NVLink5UniGBps,
			SwitchLatencyUs:  0.8,
			TensorCoreTFLOPS: nvl72.NVL72BlackwellTensorTFLOPS,
			Interconnect:     "NVL72",
		},
		Arch:          "sm_100a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		WeightDigest:  "sha256:llama3_70b_tp8_bf16",
		PlanKey:       kernelName,
		ShapeBucket:   16,
	}

	coord, services := newTestMultiRankCoordinator(t, ctx, worldSize, kernelName)
	ptxasCompiler := &autonomousTestPTXASCompiler{}

	cfg := DistAutonomousConfig{
		DistCache:     distCache,
		CacheKey:      cacheKey,
		Device:        "NVIDIA B200",
		Arch:          "sm_100a",
		DeviceSMs:     148,
		BeamWidth:     4,
		DistParams: nvl72.DistFusionCostParams{
			WorldSize:        worldSize,
			SwitchBWGBps:     nvl72.NVL72NVLink5UniGBps,
			SwitchLatencyUs:  0.8,
			TensorCoreTFLOPS: nvl72.NVL72BlackwellTensorTFLOPS,
			BatchTokens:      16,
			Target:           "sm_100a",
		},
		PTXASCompiler: ptxasCompiler,
		PTXASGate:     qualify.StaticGate{NoSpills: true, MaxRegisters: 255, MinCTAsPerSM: 1},
		MaxPTXASIters: 4,
		Coordinator:   coord,
		KernelName:    kernelName,
	}

	id := qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("ctx:dist_autonomous:sm_100a:tp8")),
		Admission: sha256.Sum256([]byte("adm:dist_autonomous:trusted")),
	}
	ev := testWinningHoldoutEvidence(id)
	winningCubin := []byte("ELF-CUBIN-NVL72-TP8-MULTIMEM-VERIFIED")

	// 1. Cold-start run: Cache Miss -> SearchTopologyAwareBeam -> PTXAS spill cure ->
	// Holdout tournament -> 8-rank barrier hot-swap at epoch 3 -> Quorum cache commit.
	coldRes, err := RunDistAutonomousPipeline(ctx, kernelName, cfg, DistAutonomousRequest{
		ComputeGraph:    compG,
		ComputeRoot:     compRoot,
		DistGraph:       distG,
		DistRoot:        distRoot,
		OutDType:        hopper.BF16,
		Identity:        id,
		DefaultEvidence: &ev,
		SwitchAtEpoch:   3,
		WinningCubin:    winningCubin,
	})
	if err != nil {
		t.Fatalf("RunDistAutonomousPipeline (cold): %v", err)
	}
	if coldRes.CacheHit {
		t.Fatal("expected CacheHit=false on cold start")
	}
	if coldRes.Selected == nil || coldRes.Selected.CollectiveRule != nvl72.RulePullMultimemLdReduce {
		t.Fatalf("expected Selected rule %q, got %+v", nvl72.RulePullMultimemLdReduce, coldRes.Selected)
	}
	if len(coldRes.PTXASRefinements) == 0 {
		t.Fatal("expected PTXASRefinements from zero-GPU ptxas loop")
	}
	if coldRes.HoldoutWinner == nil || !coldRes.HoldoutWinner.Promoted {
		t.Fatalf("expected HoldoutWinner.Promoted=true, got %+v", coldRes.HoldoutWinner)
	}
	if coldRes.SwapResult == nil || !coldRes.SwapResult.Committed || coldRes.SwapResult.SwitchAtEpoch != 3 {
		t.Fatalf("unexpected SwapResult: %+v", coldRes.SwapResult)
	}
	if !coldRes.PersistedToCache || coldRes.CachedEntry == nil {
		t.Fatalf("expected PersistedToCache=true and non-nil CachedEntry, got %+v", coldRes)
	}
	for r, svc := range services {
		st, sErr := svc.Status(kernelName)
		if sErr != nil || st.Version != 2 {
			t.Fatalf("rank %d version = %d (err=%v), want 2", r, st.Version, sErr)
		}
	}

	// 2. Warm-start run on a fresh 8-rank coordinator: hits DistVerifiedPGOCache
	// without re-running beam search or ptxas refinement, and hot-swaps all 8 ranks
	// at SwitchAtEpoch=4.
	warmCoord, warmServices := newTestMultiRankCoordinator(t, ctx, worldSize, kernelName)
	warmCfg := cfg
	warmCfg.Coordinator = warmCoord
	warmCfg.PTXASCompiler = nil

	warmRes, err := RunDistAutonomousPipeline(ctx, kernelName, warmCfg, DistAutonomousRequest{
		SwitchAtEpoch: 4,
	})
	if err != nil {
		t.Fatalf("RunDistAutonomousPipeline (warm): %v", err)
	}
	if !warmRes.CacheHit || warmRes.CachedEntry == nil {
		t.Fatalf("expected CacheHit=true on warm start, got %+v", warmRes)
	}
	if !bytes.Equal(warmRes.CachedEntry.Cubin, winningCubin) {
		t.Fatalf("warm CachedEntry.Cubin mismatch: got %d bytes, want %d", len(warmRes.CachedEntry.Cubin), len(winningCubin))
	}
	if warmRes.SwapResult == nil || !warmRes.SwapResult.Committed || warmRes.SwapResult.SwitchAtEpoch != 4 {
		t.Fatalf("unexpected warm SwapResult: %+v", warmRes.SwapResult)
	}
	for r, svc := range warmServices {
		st, sErr := svc.Status(kernelName)
		if sErr != nil || st.Version != 2 {
			t.Fatalf("warm rank %d version = %d (err=%v), want 2", r, st.Version, sErr)
		}
	}
}
