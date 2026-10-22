// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"reflect"
	"testing"
)

// simulateShufflePlan applies LocalCopies, P2PWaves, and ReplicaFills to an
// initial per-rank slot buffer holding the logical expert IDs of oldMapping,
// and verifies wave contention-freedom and exact final agreement with newMapping.
func simulateShufflePlan(t *testing.T, oldMapping, newMapping []int32, epSize int, plan ExpertShufflePlan) {
	t.Helper()
	numPhysical := len(oldMapping)
	numLocal := numPhysical / epSize

	// Initial state on each rank and staging buffer after Phase 1 & Phase 2.
	origin := make([][]int32, epSize)
	staged := make([][]int32, epSize)
	for r := 0; r < epSize; r++ {
		origin[r] = make([]int32, numLocal)
		staged[r] = make([]int32, numLocal)
		for s := 0; s < numLocal; s++ {
			e := oldMapping[r*numLocal+s]
			origin[r][s] = e
			if newMapping[r*numLocal+s] == e {
				staged[r][s] = e
			} else {
				staged[r][s] = -1
			}
		}
	}

	// Phase 1: LocalCopies read from origin[r][src] and write to staged[r][dst].
	for _, cp := range plan.LocalCopies {
		if cp.Rank < 0 || cp.Rank >= epSize {
			t.Fatalf("LocalCopyOp invalid rank %d", cp.Rank)
		}
		got := origin[cp.Rank][cp.SrcLocalIdx]
		if got != cp.LogicalExpert {
			t.Fatalf("LocalCopyOp on rank %d src %d has expert %d, want %d", cp.Rank, cp.SrcLocalIdx, got, cp.LogicalExpert)
		}
		staged[cp.Rank][cp.DstLocalIdx] = got
	}

	// Phase 2: P2PWaves. Verify every wave has disjoint ranks and apply transfers.
	for wIdx, wave := range plan.P2PWaves {
		if len(wave) == 0 {
			t.Fatalf("wave %d is empty", wIdx)
		}
		busyRank := make(map[int]bool)
		for sIdx, step := range wave {
			if step.RankA == step.RankB {
				t.Fatalf("wave %d step %d pairs rank %d with itself", wIdx, sIdx, step.RankA)
			}
			if busyRank[step.RankA] || busyRank[step.RankB] {
				t.Fatalf("wave %d has rank contention at step %d (%d <-> %d)", wIdx, sIdx, step.RankA, step.RankB)
			}
			busyRank[step.RankA] = true
			busyRank[step.RankB] = true

			if step.Bidirectional {
				// Both A->B and B->A transfer in this step.
				if origin[step.RankA][step.SlotA] != step.ExpertA {
					t.Fatalf("wave %d step %d RankA %d SlotA %d holds %d, want ExpertA %d", wIdx, sIdx, step.RankA, step.SlotA, origin[step.RankA][step.SlotA], step.ExpertA)
				}
				if origin[step.RankB][step.SlotB] != step.ExpertB {
					t.Fatalf("wave %d step %d RankB %d SlotB %d holds %d, want ExpertB %d", wIdx, sIdx, step.RankB, step.SlotB, origin[step.RankB][step.SlotB], step.ExpertB)
				}
				// Place ExpertA into the first unsatisfied slot for ExpertA on RankB, and ExpertB on RankA.
				placedAB := false
				for dst := 0; dst < numLocal; dst++ {
					if staged[step.RankB][dst] == -1 && newMapping[step.RankB*numLocal+dst] == step.ExpertA {
						staged[step.RankB][dst] = step.ExpertA
						placedAB = true
						break
					}
				}
				if !placedAB {
					t.Fatalf("wave %d step %d could not place ExpertA %d on RankB %d", wIdx, sIdx, step.ExpertA, step.RankB)
				}
				placedBA := false
				for dst := 0; dst < numLocal; dst++ {
					if staged[step.RankA][dst] == -1 && newMapping[step.RankA*numLocal+dst] == step.ExpertB {
						staged[step.RankA][dst] = step.ExpertB
						placedBA = true
						break
					}
				}
				if !placedBA {
					t.Fatalf("wave %d step %d could not place ExpertB %d on RankA %d", wIdx, sIdx, step.ExpertB, step.RankA)
				}
			} else {
				if origin[step.RankA][step.SlotA] != step.ExpertA {
					t.Fatalf("wave %d step %d RankA %d SlotA %d holds %d, want %d", wIdx, sIdx, step.RankA, step.SlotA, origin[step.RankA][step.SlotA], step.ExpertA)
				}
				staged[step.RankB][step.SlotB] = step.ExpertB
			}
		}
	}

	// Phase 3: ReplicaFills copy from already-populated staged[r][src] to staged[r][dst].
	for _, fill := range plan.ReplicaFills {
		got := staged[fill.Rank][fill.SrcLocalIdx]
		if got != fill.LogicalExpert {
			t.Fatalf("ReplicaFill on rank %d src %d has expert %d, want %d", fill.Rank, fill.SrcLocalIdx, got, fill.LogicalExpert)
		}
		staged[fill.Rank][fill.DstLocalIdx] = got
	}

	// Final check: flattened staged matches newMapping and plan.UpdatedMapping.
	var finalFlat []int32
	for r := 0; r < epSize; r++ {
		finalFlat = append(finalFlat, staged[r]...)
	}
	if !reflect.DeepEqual(finalFlat, newMapping) {
		t.Fatalf("simulated final mapping = %v, want %v", finalFlat, newMapping)
	}
	if !reflect.DeepEqual(plan.UpdatedMapping, newMapping) {
		t.Fatalf("UpdatedMapping = %v, want %v", plan.UpdatedMapping, newMapping)
	}
}

func TestPlanExpertShuffle(t *testing.T) {
	t.Run("local_reuse_p2p_waves_and_replica_fills", func(t *testing.T) {
		// 4 ranks, 3 slots per rank (12 physical slots total).
		oldMap := []int32{
			0, 1, 2, // rank 0
			3, 4, 5, // rank 1
			6, 7, 0, // rank 2
			1, 2, 3, // rank 3
		}
		newMap := []int32{
			2, 3, 3, // rank 0: reuses 2 locally (slot 2->0), receives 3 remotely (slot 1), replica-fills 3 (slot 1->2)
			0, 4, 4, // rank 1: keeps 4 in slot 1, replica-fills 4 (slot 1->2), receives 0 remotely (slot 0)
			7, 1, 5, // rank 2: reuses 7 locally (slot 1->0), receives 1 and 5 remotely
			6, 6, 2, // rank 3: reuses 2 locally (slot 1->2), receives 6 remotely (slot 0), replica-fills 6 (slot 0->1)
		}

		plan, err := PlanExpertShuffle(oldMap, newMap, 4)
		if err != nil {
			t.Fatalf("PlanExpertShuffle unexpected error: %v", err)
		}
		if len(plan.LocalCopies) == 0 {
			t.Fatalf("expected non-empty LocalCopies")
		}
		if len(plan.P2PWaves) == 0 {
			t.Fatalf("expected non-empty P2PWaves")
		}
		if len(plan.ReplicaFills) != 3 {
			t.Fatalf("len(ReplicaFills) = %d, want 3 (got %+v)", len(plan.ReplicaFills), plan.ReplicaFills)
		}
		simulateShufflePlan(t, oldMap, newMap, 4, plan)
	})

	t.Run("identity_mapping_produces_zero_ops", func(t *testing.T) {
		mapping := []int32{0, 1, 2, 3, 4, 5, 6, 7}
		plan, err := PlanExpertShuffle(mapping, mapping, 4)
		if err != nil {
			t.Fatalf("PlanExpertShuffle error: %v", err)
		}
		if len(plan.LocalCopies) != 0 || len(plan.P2PWaves) != 0 || len(plan.ReplicaFills) != 0 {
			t.Fatalf("identity shuffle should produce 0 ops, got %+v", plan)
		}
		simulateShufflePlan(t, mapping, mapping, 4, plan)
	})

	t.Run("odd_ep_size_circle_pairing", func(t *testing.T) {
		oldMap := []int32{0, 1, 2, 3, 4, 5}
		newMap := []int32{2, 4, 0, 5, 1, 3}
		plan, err := PlanExpertShuffle(oldMap, newMap, 3)
		if err != nil {
			t.Fatalf("PlanExpertShuffle error: %v", err)
		}
		simulateShufflePlan(t, oldMap, newMap, 3, plan)
	})

	t.Run("validation_errors", func(t *testing.T) {
		if _, err := PlanExpertShuffle([]int32{0, 1}, []int32{0, 1}, 0); err == nil {
			t.Fatalf("expected error for epSize=0")
		}
		if _, err := PlanExpertShuffle([]int32{0, 1, 2}, []int32{0, 1, 2}, 2); err == nil {
			t.Fatalf("expected error when physical count not divisible by epSize")
		}
		if _, err := PlanExpertShuffle([]int32{0, 1}, []int32{0, 99}, 2); err == nil {
			t.Fatalf("expected error when target expert missing from origin")
		}
	})
}
