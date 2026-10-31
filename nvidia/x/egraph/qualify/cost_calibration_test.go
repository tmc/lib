package qualify

import (
	"crypto/sha256"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func TestCalibratedCostMultiplierAndSmoothing(t *testing.T) {
	tests := []struct {
		name         string
		observations []CostObservation
		queryOp      string
		querySched   string
		wantLessOne  bool
		wantMoreOne  bool
	}{
		{
			name: "verified_holdout_win_lowers_cost",
			observations: []CostObservation{
				{
					Op:                   hopper.OpGEMM,
					ScheduleKey:          "wgmma=false",
					AnalyticalCost:       2000,
					MeasuredHoldoutRatio: 1.45,
					RegistersPerThread:   96,
					TotalSpillBytes:      0,
				},
				{
					Op:                   hopper.OpGEMM,
					ScheduleKey:          "wgmma=false",
					AnalyticalCost:       2000,
					MeasuredHoldoutRatio: 1.50,
					RegistersPerThread:   96,
					TotalSpillBytes:      0,
				},
			},
			queryOp:     hopper.OpGEMM,
			querySched:  "wgmma=false",
			wantLessOne: true,
		},
		{
			name: "spills_and_high_registers_penalize_cost",
			observations: []CostObservation{
				{
					Op:                   hopper.OpGEMM,
					ScheduleKey:          "wgmma=true",
					AnalyticalCost:       2000,
					MeasuredHoldoutRatio: 0.82,
					RegistersPerThread:   224,
					TotalSpillBytes:      64,
				},
			},
			queryOp:     hopper.OpGEMM,
			querySched:  "wgmma=true",
			wantMoreOne: true,
		},
		{
			name: "register_pressure_only_penalizes_cost",
			observations: []CostObservation{
				{
					Op:                   "tile_gemm",
					ScheduleKey:          "stages=4",
					AnalyticalCost:       1800,
					MeasuredHoldoutRatio: 1.0,
					RegistersPerThread:   216,
					TotalSpillBytes:      0,
				},
			},
			queryOp:     "tile_gemm",
			querySched:  "stages=4",
			wantMoreOne: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cal := NewCalibratedCostModel(nil)
			if got := cal.Multiplier(tc.queryOp, tc.querySched); got != 1.0 {
				t.Fatalf("initial multiplier = %v, want 1.0", got)
			}
			for _, obs := range tc.observations {
				cal.Observe(obs)
			}
			mult := cal.Multiplier(tc.queryOp, tc.querySched)
			if tc.wantLessOne && !(mult < 1.0) {
				t.Fatalf("Multiplier(%q, %q) = %v, want < 1.0", tc.queryOp, tc.querySched, mult)
			}
			if tc.wantMoreOne && !(mult > 1.0) {
				t.Fatalf("Multiplier(%q, %q) = %v, want > 1.0", tc.queryOp, tc.querySched, mult)
			}

			opFn := cal.OpCostFunc()
			node := egraph.ENode{
				Op:       tc.queryOp,
				Data:     tc.querySched,
				Children: []egraph.Id{1, 2},
			}
			baseCost := (hopper.Cost{}).NodeCost(node, []float64{100, 200})
			calCost := opFn(node, []float64{100, 200})
			if calCost < 200 {
				t.Fatalf("calibrated NodeCost = %v Violates monotone child bound 200", calCost)
			}
			if tc.wantLessOne && !(calCost < baseCost) {
				t.Fatalf("calibrated NodeCost = %v, want < baseCost %v", calCost, baseCost)
			}
			if tc.wantMoreOne && !(calCost > baseCost) {
				t.Fatalf("calibrated NodeCost = %v, want > baseCost %v", calCost, baseCost)
			}
		})
	}
}

func TestCalibratedCostSpillProportionality(t *testing.T) {
	calLow := NewCalibratedCostModel(nil)
	calLow.Observe(CostObservation{
		Op:                 "tile_gemm",
		ScheduleKey:        "sched_low_spill",
		RegistersPerThread: 140,
		TotalSpillBytes:    8,
	})

	calHigh := NewCalibratedCostModel(nil)
	calHigh.Observe(CostObservation{
		Op:                 "tile_gemm",
		ScheduleKey:        "sched_high_spill",
		RegistersPerThread: 220,
		TotalSpillBytes:    128,
	})

	lowMult := calLow.Multiplier("tile_gemm", "sched_low_spill")
	highMult := calHigh.Multiplier("tile_gemm", "sched_high_spill")
	if !(lowMult > 1.0 && highMult > lowMult) {
		t.Fatalf("expected proportional spill/register penalty (1.0 < low=%v < high=%v)", lowMult, highMult)
	}
}

func TestCalibratedCostBeamShiftFromVerifiedPGOCache(t *testing.T) {
	buildPrefillGraph := func(t *testing.T) (*egraph.EGraph, egraph.Id) {
		t.Helper()
		b, root := buildTransformerFFNGraph(64, 256, 256)
		return b.G, root
	}

	buildDecodeGraph := func(t *testing.T) (*egraph.EGraph, egraph.Id) {
		t.Helper()
		b, root := buildTransformerFFNGraph(4, 256, 256)
		return b.G, root
	}

	tests := []struct {
		name         string
		buildGraph   func(t *testing.T) (*egraph.EGraph, egraph.Id)
		selectTarget func(t *testing.T, uncal []StructuralBeamCandidate) (winnerIdx int, schedKey string, op string)
		holdoutRatio float64
		loserSpills  int
		loserRegs    int
	}{
		{
			name:       "prefill_tile_gemm_shifts_from_spilling_wgmma_to_verified_mma_sync",
			buildGraph: buildPrefillGraph,
			selectTarget: func(t *testing.T, uncal []StructuralBeamCandidate) (int, string, string) {
				t.Helper()
				for i, c := range uncal {
					if i == 0 || c.Plan == nil {
						continue
					}
					for _, st := range c.Plan.Kernels {
						if st.Tile != nil && !st.Tile.WGMMA {
							return i, "wgmma=false", "tile_gemm"
						}
					}
				}
				return 1, uncal[1].PlanSignature, "tile_gemm"
			},
			holdoutRatio: 1.65,
			loserSpills:  96,
			loserRegs:    232,
		},
		{
			name:       "decode_skinny_gemm_shifts_from_prefetch_to_verified_non_prefetch_or_unfused",
			buildGraph: buildDecodeGraph,
			selectTarget: func(t *testing.T, uncal []StructuralBeamCandidate) (int, string, string) {
				t.Helper()
				for i, c := range uncal {
					if i == 0 || c.Plan == nil {
						continue
					}
					for _, st := range c.Plan.Kernels {
						if st.GEMM != nil && st.GEMM.Prefetch == 0 {
							return i, "prefetch=false", "skinny_gemm"
						}
					}
				}
				return 1, uncal[1].PlanSignature, "skinny_gemm"
			},
			holdoutRatio: 1.58,
			loserSpills:  48,
			loserRegs:    196,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g, root := tc.buildGraph(t)

			uncalModel := NewCalibratedCostModel(nil)
			uncalBeam, err := ExtractCalibratedStructuralBeam(g, root, 8, uncalModel)
			if err != nil {
				t.Fatalf("ExtractCalibratedStructuralBeam (uncalibrated): %v", err)
			}
			if len(uncalBeam) < 2 {
				t.Fatalf("expected at least 2 beam candidates, got %d", len(uncalBeam))
			}
			uncalTop := uncalBeam[0]
			targetIdx, winnerSchedKey, opKind := tc.selectTarget(t, uncalBeam)
			targetCand := uncalBeam[targetIdx]
			if targetCand.PlanSignature == uncalTop.PlanSignature {
				t.Fatalf("target candidate signature matches uncalibrated top-1 %s", uncalTop.PlanSignature)
			}

			// Store the verified winner in an on-disk VerifiedPGOCache.
			cacheDir := t.TempDir()
			cache := autotune.NewVerifiedPGOCache(cacheDir)
			cubin := []byte("MOCK_VERIFIED_CUBIN_" + tc.name)
			cubinSum := sha256.Sum256(cubin)
			schedMeta, err := json.Marshal(map[string]any{
				"op":                   opKind,
				"schedule_key":         winnerSchedKey,
				"plan_signature":       targetCand.PlanSignature,
				"analytical_cost":      targetCand.EstimatedNs,
				"registers_per_thread": 72,
				"total_spill_bytes":    0,
			})
			if err != nil {
				t.Fatalf("json.Marshal schedule metadata: %v", err)
			}

			key := autotune.PGOCacheKey{
				Arch:          "sm_90a",
				DriverVersion: "570.124",
				CUDAVersion:   "12.8",
				WeightDigest:  "sha256:weights",
				PlanKey:       targetCand.PlanSignature,
				ShapeBucket:   64,
			}
			proof := autotune.PGOPromotionProof{
				Key:            key,
				CandidateName:  targetCand.Name,
				ScheduleJSON:   schedMeta,
				CubinSHA256:    cubinSum,
				ScreenBlocks:   6,
				HoldoutBlocks:  8,
				ScreenRatio:    tc.holdoutRatio,
				HoldoutRatio:   tc.holdoutRatio,
				AANullRatio:    1.0,
				OneSidedPValue: 0.00390625,
				TokensMatched:  true,
				PromotedAt:     time.Unix(1700000000, 0).UTC(),
			}
			if err := cache.Store(key, proof, cubin); err != nil {
				t.Fatalf("VerifiedPGOCache.Store: %v", err)
			}

			entries, err := LoadPGOCacheEntries(cache)
			if err != nil {
				t.Fatalf("LoadPGOCacheEntries: %v", err)
			}
			if len(entries) != 1 {
				t.Fatalf("LoadPGOCacheEntries count = %d, want 1", len(entries))
			}

			// Also attach a PTXAS spill/register penalty entry for the uncalibrated top-1 schedule.
			ptxOut := `ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'uncal_top' for 'sm_90a'
ptxas info    : Function properties for uncal_top
    128 bytes stack frame, 64 bytes spill stores, 32 bytes spill loads
ptxas info    : Used 224 registers, 32768 bytes smem, 384 bytes cmem[0]
`
			ptxRep, err := ptxasinfo.Parse(strings.NewReader(ptxOut))
			if err != nil {
				t.Fatalf("ptxasinfo.Parse: %v", err)
			}
			entries = append(entries, PGOCacheEntry{
				Op:                 opKind,
				ScheduleKey:        uncalTop.PlanSignature,
				AnalyticalCost:     uncalTop.EstimatedNs,
				RegistersPerThread: tc.loserRegs,
				TotalSpillBytes:    tc.loserSpills,
				PTXAS:              ptxRep,
			})

			calModel := NewCalibratedCostModel(nil)
			calModel.IngestPGOCache(entries)

			calBeam, err := ExtractCalibratedStructuralBeam(g, root, 8, calModel)
			if err != nil {
				t.Fatalf("ExtractCalibratedStructuralBeam (calibrated): %v", err)
			}
			if len(calBeam) == 0 {
				t.Fatal("calibrated beam is empty")
			}

			calTop := calBeam[0]
			if calTop.PlanSignature == uncalTop.PlanSignature {
				t.Fatalf("expected top-1 plan to shift away from uncalibrated %s after ingesting VerifiedPGOCache, still got %s",
					uncalTop.PlanSignature, calTop.PlanSignature)
			}
			if calTop.PlanSignature != targetCand.PlanSignature {
				t.Fatalf("calibrated top-1 signature = %s (%s), want empirical winner %s (%s)",
					calTop.PlanSignature, calTop.Name, targetCand.PlanSignature, targetCand.Name)
			}
			if !(calTop.EstimatedNs < uncalTop.EstimatedNs) {
				t.Fatalf("expected calibrated winner EstimatedNs (%v) < uncalibrated EstimatedNs (%v)",
					calTop.EstimatedNs, uncalTop.EstimatedNs)
			}
		})
	}
}
