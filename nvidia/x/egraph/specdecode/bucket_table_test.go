package specdecode

import (
	"bytes"
	"context"
	"crypto/sha256"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestMultiBucketMegakernelTablePreWarm(t *testing.T) {
	builder := DefaultCrossLayerBucketBuilder(256, 512, 256, "sm_121a")

	tbl, err := NewMultiBucketMegakernelTable(MultiBucketTableConfig{
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		WeightDigest:  "sha256_llama3_8b_weights_v1",
		PlanKeyPrefix: "llama_cross_l0_l1",
		Builder:       builder,
	})
	if err != nil {
		t.Fatalf("NewMultiBucketMegakernelTable: %v", err)
	}

	entries := tbl.Entries()
	wantBuckets := []struct {
		m          int
		wantRegime BucketRegime
	}{
		{m: 1, wantRegime: RegimeSkinnyGEMV},
		{m: 2, wantRegime: RegimeSkinnyGEMV},
		{m: 4, wantRegime: RegimeTensorCoreTile},
		{m: 8, wantRegime: RegimeTensorCoreTile},
	}
	if len(entries) != len(wantBuckets) {
		t.Fatalf("len(Entries()) = %d, want %d", len(entries), len(wantBuckets))
	}

	for i, wb := range wantBuckets {
		entry, ok := tbl.Entry(wb.m)
		if !ok || entry == nil {
			t.Fatalf("Entry(%d) missing", wb.m)
		}
		if entries[i] != entry {
			t.Fatalf("Entries()[%d] != Entry(%d)", i, wb.m)
		}
		if entry.Spec.BucketM != wb.m {
			t.Fatalf("Entry(%d).Spec.BucketM = %d, want %d", wb.m, entry.Spec.BucketM, wb.m)
		}
		if entry.Spec.Regime != wb.wantRegime {
			t.Fatalf("Entry(%d).Spec.Regime = %q, want %q", wb.m, entry.Spec.Regime, wb.wantRegime)
		}
		if entry.CacheHit {
			t.Fatalf("Entry(%d).CacheHit = true on cold start, want false", wb.m)
		}
		if entry.Plan == nil || len(entry.Plan.Kernels) == 0 {
			t.Fatalf("Entry(%d).Plan is empty", wb.m)
		}
		if len(entry.Megakernel.Stages) != 6 {
			t.Fatalf("Entry(%d).Megakernel has %d stages, want 6 (5 post + 1 nextAttnIn)", wb.m, len(entry.Megakernel.Stages))
		}
		if entry.Plan.Output.Shape[0] != wb.m || entry.Plan.Output.Shape[1] != 256 {
			t.Fatalf("Entry(%d).Plan.Output.Shape = %v, want [%d 256]", wb.m, entry.Plan.Output.Shape, wb.m)
		}
		if !strings.Contains(entry.CUDASource, "__global__ void __launch_bounds__(256)") {
			t.Fatalf("Entry(%d).CUDASource missing megakernel entry header", wb.m)
		}
		if !strings.Contains(entry.CUDASource, "eg_ready_wait") || !strings.Contains(entry.CUDASource, "eg_ready_signal") {
			t.Fatalf("Entry(%d).CUDASource missing ready counter synchronization", wb.m)
		}
		if err := entry.CacheKey.Validate(); err != nil {
			t.Fatalf("Entry(%d).CacheKey.Validate(): %v", wb.m, err)
		}
	}
}

func TestMultiBucketMegakernelTableTransitionsAndHotSwap(t *testing.T) {
	ctx := context.Background()
	builder := DefaultCrossLayerBucketBuilder(256, 512, 256, "sm_121a")

	var callbackTransitions []BucketTransition
	var callbackEntries []*BucketMegakernelEntry

	tbl, err := NewMultiBucketMegakernelTable(MultiBucketTableConfig{
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		WeightDigest:  "sha256_llama3_8b_weights_v1",
		PlanKeyPrefix: "llama_cross_l0_l1",
		Builder:       builder,
		OnTransition: func(_ context.Context, entry *BucketMegakernelEntry, tr BucketTransition) error {
			callbackEntries = append(callbackEntries, entry)
			callbackTransitions = append(callbackTransitions, tr)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("NewMultiBucketMegakernelTable: %v", err)
	}

	steps := []struct {
		name           string
		draftTokens    int
		acceptanceRate float64
		wantBucketM    int
		wantRegime     BucketRegime
		wantChanged    bool
		wantHotSwaps   int
	}{
		{
			name:           "initial_single_token_skinny_m1",
			draftTokens:    0,
			acceptanceRate: 0.20,
			wantBucketM:    1,
			wantRegime:     RegimeSkinnyGEMV,
			wantChanged:    false,
			wantHotSwaps:   0,
		},
		{
			name:           "draft3_medium_acceptance_transitions_to_m4_tensor_core",
			draftTokens:    3,
			acceptanceRate: 0.70, // EMA = 0.5*0.70 + 0.5*0.20 = 0.45 in [0.30, 0.65) -> M=4
			wantBucketM:    4,
			wantRegime:     RegimeTensorCoreTile,
			wantChanged:    true,
			wantHotSwaps:   1,
		},
		{
			name:           "draft7_high_acceptance_transitions_to_m8_tensor_core",
			draftTokens:    7,
			acceptanceRate: 0.95, // EMA = 0.5*0.95 + 0.5*0.45 = 0.70 >= 0.65 -> M=8
			wantBucketM:    8,
			wantRegime:     RegimeTensorCoreTile,
			wantChanged:    true,
			wantHotSwaps:   2,
		},
		{
			name:           "draft7_sustained_high_acceptance_stays_m8",
			draftTokens:    7,
			acceptanceRate: 0.90,
			wantBucketM:    8,
			wantRegime:     RegimeTensorCoreTile,
			wantChanged:    false,
			wantHotSwaps:   2,
		},
		{
			name:           "acceptance_collapse_step1",
			draftTokens:    7,
			acceptanceRate: 0.0, // EMA = 0.5*0.0 + 0.5*0.80 = 0.40 -> M=4
			wantBucketM:    4,
			wantRegime:     RegimeTensorCoreTile,
			wantChanged:    true,
			wantHotSwaps:   3,
		},
		{
			name:           "acceptance_drops_below_min_fallback_to_m1_skinny",
			draftTokens:    3,
			acceptanceRate: 0.0, // EMA = 0.5*0.0 + 0.5*0.40 = 0.20 < 0.30; step2 -> 0.10 < 0.15 -> M=1
			wantBucketM:    2,
			wantRegime:     RegimeSkinnyGEMV,
			wantChanged:    true,
			wantHotSwaps:   4,
		},
		{
			name:           "acceptance_collapses_fully_to_m1_skinny",
			draftTokens:    3,
			acceptanceRate: 0.0, // EMA = 0.5*0.0 + 0.5*0.20 = 0.10 < 0.15 -> M=1
			wantBucketM:    1,
			wantRegime:     RegimeSkinnyGEMV,
			wantChanged:    true,
			wantHotSwaps:   5,
		},
	}

	for _, tc := range steps {
		t.Run(tc.name, func(t *testing.T) {
			entry, tr, err := tbl.ObserveAndSelect(ctx, tc.draftTokens, tc.acceptanceRate)
			if err != nil {
				t.Fatalf("ObserveAndSelect(%d, %v): %v", tc.draftTokens, tc.acceptanceRate, err)
			}
			if entry == nil {
				t.Fatal("ObserveAndSelect returned nil entry")
			}
			if entry.Spec.BucketM != tc.wantBucketM {
				t.Fatalf("entry.Spec.BucketM = %d, want %d (EMA=%v)", entry.Spec.BucketM, tc.wantBucketM, tbl.Controller().AcceptanceRate())
			}
			if entry.Spec.Regime != tc.wantRegime {
				t.Fatalf("entry.Spec.Regime = %q, want %q", entry.Spec.Regime, tc.wantRegime)
			}
			if tr.Changed() != tc.wantChanged {
				t.Fatalf("tr.Changed() = %v, want %v (tr=%+v)", tr.Changed(), tc.wantChanged, tr)
			}
			if tr.To().BucketM != tc.wantBucketM {
				t.Fatalf("tr.To().BucketM = %d, want %d", tr.To().BucketM, tc.wantBucketM)
			}
			if tbl.HotSwaps() != tc.wantHotSwaps {
				t.Fatalf("tbl.HotSwaps() = %d, want %d", tbl.HotSwaps(), tc.wantHotSwaps)
			}
			if len(callbackTransitions) != tc.wantHotSwaps || len(callbackEntries) != tc.wantHotSwaps {
				t.Fatalf("callback count = %d, want %d", len(callbackTransitions), tc.wantHotSwaps)
			}
		})
	}
}

func TestMultiBucketMegakernelTablePersistAndWarmStart(t *testing.T) {
	cacheDir := t.TempDir()
	cache := autotune.NewVerifiedPGOCache(cacheDir)
	builder := DefaultCrossLayerBucketBuilder(256, 512, 256, "sm_121a")

	buildCallsCold := make(map[int]int)
	coldBuilder := func(bucketM int, regime BucketRegime) (*hopper.Plan, hopper.Megakernel, error) {
		buildCallsCold[bucketM]++
		return builder(bucketM, regime)
	}

	baseCfg := MultiBucketTableConfig{
		Arch:          "sm_121a",
		DriverVersion: "580.173.02",
		CUDAVersion:   "13.4",
		WeightDigest:  "a3f19b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a",
		PlanKeyPrefix: "llama_cross_l0_l1",
		Cache:         cache,
		Builder:       coldBuilder,
	}

	tbl1, err := NewMultiBucketMegakernelTable(baseCfg)
	if err != nil {
		t.Fatalf("NewMultiBucketMegakernelTable (cold): %v", err)
	}

	buckets := []int{1, 2, 4, 8}
	wantCubins := make(map[int][]byte, len(buckets))
	for _, m := range buckets {
		if buildCallsCold[m] != 1 {
			t.Fatalf("cold buildCalls[%d] = %d, want 1", m, buildCallsCold[m])
		}
		entry, ok := tbl1.Entry(m)
		if !ok || entry == nil {
			t.Fatalf("cold Entry(%d) missing", m)
		}
		if entry.CacheHit {
			t.Fatalf("cold Entry(%d).CacheHit = true, want false", m)
		}

		cubin := []byte("ELF-CUBIN-SM121A-CROSS-LAYER-BUCKET-M" + string(rune('0'+m)))
		wantCubins[m] = cubin
		proof := autotune.PGOPromotionProof{
			CandidateName:  entry.Megakernel.Name,
			CubinSHA256:    sha256.Sum256(cubin),
			SourceSHA256:   sha256.Sum256([]byte(entry.CUDASource)),
			ScreenBlocks:   3,
			HoldoutBlocks:  6,
			ScreenRatio:    1.06,
			HoldoutRatio:   1.05,
			AANullRatio:    1.00,
			OneSidedPValue: 0.015625,
			TokensMatched:  true,
			PromotedAt:     time.Date(2026, 10, 31, 9, 30, 0, 0, time.UTC),
		}
		if err := tbl1.PersistEntry(m, cubin, proof); err != nil {
			t.Fatalf("PersistEntry(%d): %v", m, err)
		}
	}

	// Construct a second MultiBucketMegakernelTable backed by the same VerifiedPGOCache.
	// Because all 4 buckets hit the cache with serialized Plan + Megakernel + CUDASource,
	// warmBuilder must not be invoked.
	buildCallsWarm := 0
	warmCfg := baseCfg
	warmCfg.Builder = func(bucketM int, regime BucketRegime) (*hopper.Plan, hopper.Megakernel, error) {
		buildCallsWarm++
		return builder(bucketM, regime)
	}

	tbl2, err := NewMultiBucketMegakernelTable(warmCfg)
	if err != nil {
		t.Fatalf("NewMultiBucketMegakernelTable (warm): %v", err)
	}
	if buildCallsWarm != 0 {
		t.Fatalf("warm builder was called %d times, want 0 (all restored from VerifiedPGOCache)", buildCallsWarm)
	}

	for _, m := range buckets {
		coldEntry, _ := tbl1.Entry(m)
		warmEntry, ok := tbl2.Entry(m)
		if !ok || warmEntry == nil {
			t.Fatalf("warm Entry(%d) missing", m)
		}
		if !warmEntry.CacheHit {
			t.Fatalf("warm Entry(%d).CacheHit = false, want true", m)
		}
		if !bytes.Equal(warmEntry.Cubin, wantCubins[m]) {
			t.Fatalf("warm Entry(%d).Cubin = %q, want %q", m, warmEntry.Cubin, wantCubins[m])
		}
		if warmEntry.CUDASource != coldEntry.CUDASource {
			t.Fatalf("warm Entry(%d).CUDASource mismatch", m)
		}
		if warmEntry.Megakernel.Name != coldEntry.Megakernel.Name || len(warmEntry.Megakernel.Stages) != len(coldEntry.Megakernel.Stages) {
			t.Fatalf("warm Entry(%d).Megakernel = %+v, want %+v", m, warmEntry.Megakernel, coldEntry.Megakernel)
		}
		if warmEntry.Plan == nil || warmEntry.Plan.Name != coldEntry.Plan.Name || len(warmEntry.Plan.Kernels) != len(coldEntry.Plan.Kernels) {
			t.Fatalf("warm Entry(%d).Plan = %+v, want %+v", m, warmEntry.Plan, coldEntry.Plan)
		}
	}
}
