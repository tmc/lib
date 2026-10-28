// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"math"
	"reflect"
	"testing"
)

func TestResolveNumPhysicalExperts(t *testing.T) {
	tests := []struct {
		name         string
		numRouted    int
		numRedundant int
		epSize       int
		want         int
		wantErr      bool
	}{
		{name: "exact_multiple", numRouted: 256, numRedundant: 32, epSize: 32, want: 288},
		{name: "rounds_up_to_ep_multiple", numRouted: 256, numRedundant: 10, epSize: 32, want: 288},
		{name: "zero_redundant_already_aligned", numRouted: 64, numRedundant: 0, epSize: 8, want: 64},
		{name: "zero_redundant_needs_padding", numRouted: 65, numRedundant: 0, epSize: 8, want: 72},
		{name: "ep_size_1", numRouted: 17, numRedundant: 3, epSize: 1, want: 20},
		{name: "invalid_routed", numRouted: 0, numRedundant: 4, epSize: 4, wantErr: true},
		{name: "negative_redundant", numRouted: 8, numRedundant: -1, epSize: 4, wantErr: true},
		{name: "invalid_ep_size", numRouted: 8, numRedundant: 2, epSize: 0, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveNumPhysicalExperts(tc.numRouted, tc.numRedundant, tc.epSize)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ResolveNumPhysicalExperts(%d, %d, %d) succeeded with %d, want error", tc.numRouted, tc.numRedundant, tc.epSize, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveNumPhysicalExperts(%d, %d, %d) unexpected error: %v", tc.numRouted, tc.numRedundant, tc.epSize, err)
			}
			if got != tc.want {
				t.Fatalf("ResolveNumPhysicalExperts(%d, %d, %d) = %d, want %d", tc.numRouted, tc.numRedundant, tc.epSize, got, tc.want)
			}
		})
	}
}

func checkEPLBPlacementInvariants(t *testing.T, p EPLBPlacement, cfg EPLBConfig, loads []float64) {
	t.Helper()
	if len(p.PhysicalToLogical) != cfg.NumPhysicalExperts {
		t.Fatalf("len(PhysicalToLogical) = %d, want %d", len(p.PhysicalToLogical), cfg.NumPhysicalExperts)
	}
	if len(p.LogicalToPhysical) != cfg.NumRoutedExperts {
		t.Fatalf("len(LogicalToPhysical) = %d, want %d", len(p.LogicalToPhysical), cfg.NumRoutedExperts)
	}
	if len(p.ExpertReplicaCount) != cfg.NumRoutedExperts {
		t.Fatalf("len(ExpertReplicaCount) = %d, want %d", len(p.ExpertReplicaCount), cfg.NumRoutedExperts)
	}
	if len(p.RankLoads) != cfg.EPSize {
		t.Fatalf("len(RankLoads) = %d, want %d", len(p.RankLoads), cfg.EPSize)
	}

	var totalReplicas int32
	seenPhysical := make([]bool, cfg.NumPhysicalExperts)
	for e := 0; e < cfg.NumRoutedExperts; e++ {
		if p.ExpertReplicaCount[e] < 1 {
			t.Fatalf("expert %d has replica count %d < 1", e, p.ExpertReplicaCount[e])
		}
		if len(p.LogicalToPhysical[e]) != int(p.ExpertReplicaCount[e]) {
			t.Fatalf("expert %d len(LogicalToPhysical)=%d != ExpertReplicaCount=%d", e, len(p.LogicalToPhysical[e]), p.ExpertReplicaCount[e])
		}
		totalReplicas += p.ExpertReplicaCount[e]
		for _, slot := range p.LogicalToPhysical[e] {
			if slot < 0 || int(slot) >= cfg.NumPhysicalExperts {
				t.Fatalf("expert %d has out-of-range physical slot %d", e, slot)
			}
			if seenPhysical[slot] {
				t.Fatalf("physical slot %d assigned multiple times", slot)
			}
			seenPhysical[slot] = true
			if p.PhysicalToLogical[slot] != int32(e) {
				t.Fatalf("PhysicalToLogical[%d] = %d, want %d", slot, p.PhysicalToLogical[slot], e)
			}
		}
	}
	if int(totalReplicas) != cfg.NumPhysicalExperts {
		t.Fatalf("sum(ExpertReplicaCount) = %d, want %d", totalReplicas, cfg.NumPhysicalExperts)
	}

	expertsPerRank := cfg.NumPhysicalExperts / cfg.EPSize
	for r := 0; r < cfg.EPSize; r++ {
		var wantRankLoad float64
		for s := 0; s < expertsPerRank; s++ {
			e := p.PhysicalToLogical[r*expertsPerRank+s]
			wantRankLoad += loads[e] / float64(p.ExpertReplicaCount[e])
		}
		if math.Abs(p.RankLoads[r]-wantRankLoad) > 1e-9 {
			t.Fatalf("RankLoads[%d] = %v, want %v", r, p.RankLoads[r], wantRankLoad)
		}
	}
}

func TestRebalanceExpertsGlobal(t *testing.T) {
	t.Run("skewed_hot_expert_gets_replicas_and_spreads_ranks", func(t *testing.T) {
		// 4 logical experts, 6 physical slots across 3 ranks (2 slots/rank).
		// Expert 0 has load 90 (gets 3 replicas of weight 30 each -> 1 on each of the 3 ranks).
		// Experts 1, 2, 3 have loads 25, 20, 15 (1 replica each).
		loads := []float64{90, 25, 20, 15}
		cfg := EPLBConfig{
			NumRoutedExperts:   4,
			NumPhysicalExperts: 6,
			EPSize:             3,
			Decay:              0.9,
		}
		p, err := RebalanceExpertsGlobal(loads, cfg)
		if err != nil {
			t.Fatalf("RebalanceExpertsGlobal unexpected error: %v", err)
		}
		checkEPLBPlacementInvariants(t, p, cfg, loads)

		wantReplicas := []int32{3, 1, 1, 1}
		if !reflect.DeepEqual(p.ExpertReplicaCount, wantReplicas) {
			t.Fatalf("ExpertReplicaCount = %v, want %v", p.ExpertReplicaCount, wantReplicas)
		}

		// Verify the 3 replicas of expert 0 are placed on 3 distinct ranks (0, 1, 2).
		expertsPerRank := cfg.NumPhysicalExperts / cfg.EPSize
		ranksOfE0 := make(map[int]bool)
		for _, slot := range p.LogicalToPhysical[0] {
			ranksOfE0[int(slot)/expertsPerRank] = true
		}
		if len(ranksOfE0) != 3 {
			t.Fatalf("expert 0 replicas placed on %d distinct ranks (%v), want 3", len(ranksOfE0), p.LogicalToPhysical[0])
		}

		// Verify min-peak packing balances the remaining experts [25, 20, 15] across the 3 ranks:
		// Each rank starts with 30 from expert 0, then gets one of {25, 20, 15}.
		wantLoads := []float64{55, 50, 45}
		for r, want := range wantLoads {
			if math.Abs(p.RankLoads[r]-want) > 1e-9 {
				t.Fatalf("RankLoads[%d] = %v, want %v (all RankLoads=%v)", r, p.RankLoads[r], want, p.RankLoads)
			}
		}
	})

	t.Run("second_hot_expert_splits_redundant_slots", func(t *testing.T) {
		// Expert 1 has 90 (gets 1st redundant slot -> 90/2 = 45), then Expert 0 has 50 > 45 (gets 2nd redundant slot -> 50/2 = 25).
		loads := []float64{50, 90, 20, 15}
		cfg := EPLBConfig{
			NumRoutedExperts:   4,
			NumPhysicalExperts: 6,
			EPSize:             2,
		}
		p, err := RebalanceExpertsGlobal(loads, cfg)
		if err != nil {
			t.Fatalf("RebalanceExpertsGlobal unexpected error: %v", err)
		}
		checkEPLBPlacementInvariants(t, p, cfg, loads)
		wantReplicas := []int32{2, 2, 1, 1}
		if !reflect.DeepEqual(p.ExpertReplicaCount, wantReplicas) {
			t.Fatalf("ExpertReplicaCount = %v, want %v", p.ExpertReplicaCount, wantReplicas)
		}
	})

	t.Run("tie_breaking_determinism_on_equal_loads", func(t *testing.T) {
		loads := []float64{10, 10, 10, 10}
		cfg := EPLBConfig{
			NumRoutedExperts:   4,
			NumPhysicalExperts: 6,
			EPSize:             2,
		}
		p1, err := RebalanceExpertsGlobal(loads, cfg)
		if err != nil {
			t.Fatalf("first RebalanceExpertsGlobal error: %v", err)
		}
		p2, err := RebalanceExpertsGlobal(loads, cfg)
		if err != nil {
			t.Fatalf("second RebalanceExpertsGlobal error: %v", err)
		}
		if !reflect.DeepEqual(p1, p2) {
			t.Fatalf("non-deterministic placement: %+v vs %+v", p1, p2)
		}
		// Tie break by lowest logical expert ID: experts 0 and 1 receive the 2 redundant slots.
		wantReplicas := []int32{2, 2, 1, 1}
		if !reflect.DeepEqual(p1.ExpertReplicaCount, wantReplicas) {
			t.Fatalf("ExpertReplicaCount on equal loads = %v, want %v", p1.ExpertReplicaCount, wantReplicas)
		}
		checkEPLBPlacementInvariants(t, p1, cfg, loads)
	})

	t.Run("validation_errors", func(t *testing.T) {
		validLoads := []float64{1, 2, 3, 4}
		badCfgs := []EPLBConfig{
			{NumRoutedExperts: 0, NumPhysicalExperts: 4, EPSize: 2},
			{NumRoutedExperts: 4, NumPhysicalExperts: 2, EPSize: 2},
			{NumRoutedExperts: 4, NumPhysicalExperts: 6, EPSize: 4},
			{NumRoutedExperts: 4, NumPhysicalExperts: 4, EPSize: 0},
		}
		for i, cfg := range badCfgs {
			if _, err := RebalanceExpertsGlobal(validLoads, cfg); err == nil {
				t.Fatalf("case %d: expected error for invalid cfg %+v", i, cfg)
			}
		}
		if _, err := RebalanceExpertsGlobal([]float64{1, -1, 2, 3}, EPLBConfig{NumRoutedExperts: 4, NumPhysicalExperts: 4, EPSize: 2}); err == nil {
			t.Fatalf("expected error for negative load")
		}
	})
}
