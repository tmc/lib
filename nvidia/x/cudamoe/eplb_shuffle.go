// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"fmt"
	"sort"
)

// LocalCopyOp describes an intra-rank expert weight copy between local slots.
type LocalCopyOp struct {
	Rank          int
	SrcLocalIdx   int
	DstLocalIdx   int
	LogicalExpert int32
}

// P2PTransferStep describes a cross-rank point-to-point expert weight transfer
// within a contention-free communication wave.
type P2PTransferStep struct {
	RankA         int
	SlotA         int
	ExpertA       int32
	RankB         int
	SlotB         int
	ExpertB       int32
	Bidirectional bool
}

// ExpertShufflePlan describes a zero-allocation expert weight shuffle across
// three ordered phases: intra-rank local copies, contention-free cross-rank
// P2P transfer waves, and intra-rank duplicate replica fills.
type ExpertShufflePlan struct {
	LocalCopies    []LocalCopyOp
	P2PWaves       [][]P2PTransferStep
	ReplicaFills   []LocalCopyOp
	UpdatedMapping []int32
}

func pairedSender(receiverIdx, numSenders, numReceivers int) int {
	n := numReceivers / numSenders
	if n > 0 && receiverIdx < n*numSenders {
		return receiverIdx / n
	}
	return (receiverIdx - n*numSenders) % numSenders
}

type directedP2P struct {
	srcRank int
	srcSlot int
	dstRank int
	dstSlot int
	expert  int32
}

// PlanExpertShuffle computes the zero-allocation transition plan from
// oldPhysicalToLogical to newPhysicalToLogical across epSize ranks:
//  1. Intra-rank local reuse where a logical expert already exists on the rank.
//  2. Cross-rank P2P wave scheduling using circle-method round-robin pairing so
//     every wave consists of disjoint rank pairs with zero contention.
//  3. Final local replica fill for any duplicate slots on the same rank.
func PlanExpertShuffle(oldPhysicalToLogical []int32, newPhysicalToLogical []int32, epSize int) (ExpertShufflePlan, error) {
	if epSize <= 0 {
		return ExpertShufflePlan{}, fmt.Errorf("cudamoe: epSize %d must be positive", epSize)
	}
	numPhysical := len(oldPhysicalToLogical)
	if numPhysical == 0 {
		return ExpertShufflePlan{}, fmt.Errorf("cudamoe: oldPhysicalToLogical must not be empty")
	}
	if len(newPhysicalToLogical) != numPhysical {
		return ExpertShufflePlan{}, fmt.Errorf("cudamoe: len(newPhysicalToLogical) %d must equal len(oldPhysicalToLogical) %d", len(newPhysicalToLogical), numPhysical)
	}
	if numPhysical%epSize != 0 {
		return ExpertShufflePlan{}, fmt.Errorf("cudamoe: physical expert count %d must be divisible by epSize %d", numPhysical, epSize)
	}

	numLocal := numPhysical / epSize

	// Map each logical expert to the ranks (and first local slot on that rank) holding it in oldPhysicalToLogical.
	type rankSlot struct {
		rank int
		slot int
	}
	sendersByExpert := make(map[int32][]rankSlot)
	for r := 0; r < epSize; r++ {
		base := r * numLocal
		seenOnRank := make(map[int32]bool)
		for s := 0; s < numLocal; s++ {
			e := oldPhysicalToLogical[base+s]
			if e < 0 {
				return ExpertShufflePlan{}, fmt.Errorf("cudamoe: negative expert ID %d at old slot %d", e, base+s)
			}
			if !seenOnRank[e] {
				seenOnRank[e] = true
				sendersByExpert[e] = append(sendersByExpert[e], rankSlot{rank: r, slot: s})
			}
		}
	}
	for i, e := range newPhysicalToLogical {
		if e < 0 {
			return ExpertShufflePlan{}, fmt.Errorf("cudamoe: negative expert ID %d at new slot %d", e, i)
		}
		if len(sendersByExpert[e]) == 0 {
			return ExpertShufflePlan{}, fmt.Errorf("cudamoe: target logical expert %d at slot %d is not present in oldPhysicalToLogical", e, i)
		}
	}

	received := make([][]bool, epSize)
	firstSlotOnRank := make([]map[int32]int, epSize)
	var localCopies []LocalCopyOp

	// Phase 1: Intra-rank local reuse.
	for r := 0; r < epSize; r++ {
		received[r] = make([]bool, numLocal)
		firstSlotOnRank[r] = make(map[int32]int)
		base := r * numLocal
		oldLocal := oldPhysicalToLogical[base : base+numLocal]
		newLocal := newPhysicalToLogical[base : base+numLocal]

		// Slots where the expert is unchanged stay in place without a copy.
		for s := 0; s < numLocal; s++ {
			if oldLocal[s] == newLocal[s] {
				received[r][s] = true
				if _, exists := firstSlotOnRank[r][newLocal[s]]; !exists {
					firstSlotOnRank[r][newLocal[s]] = s
				}
			}
		}

		// Satisfy the first needed slot for any expert already present on this rank in oldLocal.
		for src := 0; src < numLocal; src++ {
			e := oldLocal[src]
			if _, alreadyHave := firstSlotOnRank[r][e]; alreadyHave {
				continue
			}
			for dst := 0; dst < numLocal; dst++ {
				if !received[r][dst] && newLocal[dst] == e {
					received[r][dst] = true
					firstSlotOnRank[r][e] = dst
					if src != dst {
						localCopies = append(localCopies, LocalCopyOp{
							Rank:          r,
							SrcLocalIdx:   src,
							DstLocalIdx:   dst,
							LogicalExpert: e,
						})
					}
					break
				}
			}
		}
	}

	// Collect unique missing experts per rank (first local destination slot per missing expert).
	type recvNeed struct {
		rank    int
		dstSlot int
	}
	receiversByExpert := make(map[int32][]recvNeed)
	var activeExperts []int32
	seenMissingExpert := make(map[int32]bool)

	for r := 0; r < epSize; r++ {
		base := r * numLocal
		newLocal := newPhysicalToLogical[base : base+numLocal]
		seenNeedOnRank := make(map[int32]bool)
		for dst := 0; dst < numLocal; dst++ {
			if received[r][dst] {
				continue
			}
			e := newLocal[dst]
			if _, ok := firstSlotOnRank[r][e]; ok {
				continue
			}
			if !seenNeedOnRank[e] {
				seenNeedOnRank[e] = true
				receiversByExpert[e] = append(receiversByExpert[e], recvNeed{rank: r, dstSlot: dst})
				if !seenMissingExpert[e] {
					seenMissingExpert[e] = true
					activeExperts = append(activeExperts, e)
				}
			}
		}
	}
	sort.Slice(activeExperts, func(i, j int) bool { return activeExperts[i] < activeExperts[j] })

	// Pair each receiver with a sender via _paired_sender.
	pairQueues := make(map[[2]int][]directedP2P)
	totalDirected := 0
	for _, e := range activeExperts {
		senders := sendersByExpert[e]
		recvs := receiversByExpert[e]
		for j, req := range recvs {
			sIdx := pairedSender(j, len(senders), len(recvs))
			sender := senders[sIdx]
			pairQueues[[2]int{sender.rank, req.rank}] = append(pairQueues[[2]int{sender.rank, req.rank}], directedP2P{
				srcRank: sender.rank,
				srcSlot: sender.slot,
				dstRank: req.rank,
				dstSlot: req.dstSlot,
				expert:  e,
			})
			received[req.rank][req.dstSlot] = true
			firstSlotOnRank[req.rank][e] = req.dstSlot
			totalDirected++
		}
	}

	// Phase 2: Circle-method round-robin P2P wave scheduling.
	var p2pWaves [][]P2PTransferStep
	if totalDirected > 0 && epSize > 1 {
		rEven := epSize
		if rEven%2 != 0 {
			rEven++
		}
		numRounds := rEven - 1
		scheduled := 0
		for scheduled < totalDirected {
			for s := 0; s < numRounds && scheduled < totalDirected; s++ {
				pairs := make([][2]int, 0, rEven/2)
				u0, v0 := s, rEven-1
				if u0 > v0 {
					u0, v0 = v0, u0
				}
				if v0 < epSize {
					pairs = append(pairs, [2]int{u0, v0})
				}
				for k := 1; k < rEven/2; k++ {
					a := (s + k) % (rEven - 1)
					b := (s - k + rEven - 1) % (rEven - 1)
					if a > b {
						a, b = b, a
					}
					if b < epSize {
						pairs = append(pairs, [2]int{a, b})
					}
				}
				sort.Slice(pairs, func(i, j int) bool {
					if pairs[i][0] != pairs[j][0] {
						return pairs[i][0] < pairs[j][0]
					}
					return pairs[i][1] < pairs[j][1]
				})

				var wave []P2PTransferStep
				for _, p := range pairs {
					u, v := p[0], p[1]
					qUV := pairQueues[[2]int{u, v}]
					qVU := pairQueues[[2]int{v, u}]
					if len(qUV) > 0 && len(qVU) > 0 {
						opUV := qUV[0]
						opVU := qVU[0]
						pairQueues[[2]int{u, v}] = qUV[1:]
						pairQueues[[2]int{v, u}] = qVU[1:]
						wave = append(wave, P2PTransferStep{
							RankA:         u,
							SlotA:         opUV.srcSlot,
							ExpertA:       opUV.expert,
							RankB:         v,
							SlotB:         opVU.srcSlot,
							ExpertB:       opVU.expert,
							Bidirectional: true,
						})
						scheduled += 2
					} else if len(qUV) > 0 {
						opUV := qUV[0]
						pairQueues[[2]int{u, v}] = qUV[1:]
						wave = append(wave, P2PTransferStep{
							RankA:         u,
							SlotA:         opUV.srcSlot,
							ExpertA:       opUV.expert,
							RankB:         v,
							SlotB:         opUV.dstSlot,
							ExpertB:       opUV.expert,
							Bidirectional: false,
						})
						scheduled++
					} else if len(qVU) > 0 {
						opVU := qVU[0]
						pairQueues[[2]int{v, u}] = qVU[1:]
						wave = append(wave, P2PTransferStep{
							RankA:         v,
							SlotA:         opVU.srcSlot,
							ExpertA:       opVU.expert,
							RankB:         u,
							SlotB:         opVU.dstSlot,
							ExpertB:       opVU.expert,
							Bidirectional: false,
						})
						scheduled++
					}
				}
				if len(wave) > 0 {
					p2pWaves = append(p2pWaves, wave)
				}
			}
		}
	}

	// Phase 3: Final intra-rank replica fills for duplicate slots on the same rank.
	var replicaFills []LocalCopyOp
	for r := 0; r < epSize; r++ {
		base := r * numLocal
		newLocal := newPhysicalToLogical[base : base+numLocal]
		for dst := 0; dst < numLocal; dst++ {
			if !received[r][dst] {
				e := newLocal[dst]
				src := firstSlotOnRank[r][e]
				received[r][dst] = true
				replicaFills = append(replicaFills, LocalCopyOp{
					Rank:          r,
					SrcLocalIdx:   src,
					DstLocalIdx:   dst,
					LogicalExpert: e,
				})
			}
		}
	}

	updated := make([]int32, numPhysical)
	copy(updated, newPhysicalToLogical)

	return ExpertShufflePlan{
		LocalCopies:    localCopies,
		P2PWaves:       p2pWaves,
		ReplicaFills:   replicaFills,
		UpdatedMapping: updated,
	}, nil
}
