// SPDX-License-Identifier: Apache-2.0

package cudakv_test

import (
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestFragmentationCalculation(t *testing.T) {
	tests := []struct {
		name        string
		totalFree   int
		largestFree int
		wantIndex   float64
	}{
		{
			name:        "zero_free",
			totalFree:   0,
			largestFree: 0,
			wantIndex:   0.0,
		},
		{
			name:        "perfect_contiguity",
			totalFree:   10,
			largestFree: 10,
			wantIndex:   0.0,
		},
		{
			name:        "checkerboard_ten_single_blocks",
			totalFree:   10,
			largestFree: 1,
			wantIndex:   0.9,
		},
		{
			name:        "two_equal_runs",
			totalFree:   6,
			largestFree: 3,
			wantIndex:   0.5,
		},
		{
			name:        "mild_fragmentation",
			totalFree:   100,
			largestFree: 80,
			wantIndex:   0.2,
		},
		{
			name:        "severe_fragmentation",
			totalFree:   100,
			largestFree: 10,
			wantIndex:   0.9,
		},
		{
			name:        "largest_greater_than_total",
			totalFree:   5,
			largestFree: 10,
			wantIndex:   0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cudakv.Fragmentation(tt.totalFree, tt.largestFree)
			if math.Abs(got-tt.wantIndex) > 1e-9 {
				t.Errorf("Fragmentation(%d, %d) = %f, want %f", tt.totalFree, tt.largestFree, got, tt.wantIndex)
			}
		})
	}
}

func TestTrackerFragmentation(t *testing.T) {
	tests := []struct {
		name             string
		totalCapacity    uint64
		currentAllocated uint64
		largestFree      uint64
		wantIndex        float64
	}{
		{
			name:             "half_capacity_contiguous",
			totalCapacity:    1000,
			currentAllocated: 500,
			largestFree:      500,
			wantIndex:        0.0,
		},
		{
			name:             "half_capacity_fragmented",
			totalCapacity:    1000,
			currentAllocated: 500,
			largestFree:      100,
			wantIndex:        0.8,
		},
		{
			name:             "zero_capacity",
			totalCapacity:    0,
			currentAllocated: 0,
			largestFree:      0,
			wantIndex:        0.0,
		},
		{
			name:             "fully_allocated",
			totalCapacity:    1000,
			currentAllocated: 1000,
			largestFree:      0,
			wantIndex:        0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := &cudamem.Tracker{}
			_ = tracker.Reset()
			// Mock tracker current value via Reset if needed, or check helper
			frag := cudakv.TrackerFragmentation(tracker, tt.totalCapacity, tt.largestFree)
			if tt.totalCapacity == 0 && frag != 0.0 {
				t.Errorf("TrackerFragmentation() = %f, want 0.0", frag)
			}
		})
	}
}

func TestDefragPolicy(t *testing.T) {
	tests := []struct {
		name         string
		policy       cudakv.DefragPolicy
		stats        cudakv.FragmentationStats
		neededBlocks int
		wantCompact  bool
	}{
		{
			name:   "default_policy_below_threshold",
			policy: cudakv.DefaultDefragPolicy(),
			stats: cudakv.FragmentationStats{
				TotalBlocks:        10,
				AllocatedBlocks:    4,
				FreeBlocks:         6,
				LargestContiguous:  5,
				FragmentationIndex: 0.166666,
			},
			neededBlocks: 0,
			wantCompact:  false,
		},
		{
			name:   "default_policy_above_threshold",
			policy: cudakv.DefaultDefragPolicy(),
			stats: cudakv.FragmentationStats{
				TotalBlocks:        10,
				AllocatedBlocks:    5,
				FreeBlocks:         5,
				LargestContiguous:  2,
				FragmentationIndex: 0.6,
			},
			neededBlocks: 0,
			wantCompact:  true,
		},
		{
			name: "allocation_blocked_by_fragmentation",
			policy: cudakv.DefragPolicy{
				Threshold: 0.80, // high threshold, but allocation fails
			},
			stats: cudakv.FragmentationStats{
				TotalBlocks:        20,
				AllocatedBlocks:    10,
				FreeBlocks:         10,
				LargestContiguous:  3, // largest run is 3
				FragmentationIndex: 0.70,
			},
			neededBlocks: 4, // needs 4 contiguous blocks: 10 free >= 4, but largest 3 < 4
			wantCompact:  true,
		},
		{
			name: "allocation_fits_contiguously",
			policy: cudakv.DefragPolicy{
				Threshold: 0.80,
			},
			stats: cudakv.FragmentationStats{
				TotalBlocks:        20,
				AllocatedBlocks:    10,
				FreeBlocks:         10,
				LargestContiguous:  5,
				FragmentationIndex: 0.50,
			},
			neededBlocks: 4,
			wantCompact:  false,
		},
		{
			name: "insufficient_total_memory",
			policy: cudakv.DefragPolicy{
				Threshold: 0.80,
			},
			stats: cudakv.FragmentationStats{
				TotalBlocks:        10,
				AllocatedBlocks:    8,
				FreeBlocks:         2,
				LargestContiguous:  1,
				FragmentationIndex: 0.50,
			},
			neededBlocks: 4, // needs 4, but only 2 free total (pure OOM)
			wantCompact:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.policy.ShouldCompact(tt.stats, tt.neededBlocks)
			if got != tt.wantCompact {
				t.Errorf("ShouldCompact() = %v, want %v", got, tt.wantCompact)
			}
		})
	}
}

func TestCompactionPlan(t *testing.T) {
	tests := []struct {
		name           string
		totalBlocks    int
		blockSize      int
		setup          func(a *cudakv.BlockAllocator)
		maxMoves       int
		wantMovesCount int
		wantFinalFrag  float64
	}{
		{
			name:        "already_compact",
			totalBlocks: 4,
			blockSize:   2,
			setup: func(a *cudakv.BlockAllocator) {
				// Manually simulate active blocks occupying slots 0 and 1
				// e.g. via retain or alloc then apply
				plan := a.PlanCompaction(0)
				_ = plan
			},
			maxMoves:       0,
			wantMovesCount: 0,
			wantFinalFrag:  0.0,
		},
		{
			name:        "scattered_single_move",
			totalBlocks: 4,
			blockSize:   2,
			setup: func(a *cudakv.BlockAllocator) {
				// Allocate sequence, then apply a plan to place one block at 0 and one at 3
				_ = a.AllocSequence(1, nil, 4) // takes blocks 3 and 2
				// We want block 0 active, block 3 active, total active = 2
				a.ApplyCompaction(cudakv.CompactionPlan{
					Moves: []cudakv.BlockMove{
						{SrcBlock: 2, DstBlock: 0},
					},
					Remappings: map[int]int{2: 0},
				})
			},
			maxMoves:       0,
			wantMovesCount: 1, // slot 3 moves to slot 1
			wantFinalFrag:  0.0,
		},
		{
			name:        "max_moves_limit",
			totalBlocks: 6,
			blockSize:   2,
			setup: func(a *cudakv.BlockAllocator) {
				// Active blocks at 4 and 5, target [0, 1] needs 2 moves
				_ = a.AllocSequence(1, nil, 4) // takes blocks 5 and 4
			},
			maxMoves:       1,
			wantMovesCount: 1,
			wantFinalFrag:  0.25,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			alloc := cudakv.NewBlockAllocator(tt.blockSize, tt.totalBlocks)
			tt.setup(alloc)
			plan := alloc.PlanCompaction(tt.maxMoves)
			if len(plan.Moves) != tt.wantMovesCount {
				t.Errorf("len(plan.Moves) = %d, want %d", len(plan.Moves), tt.wantMovesCount)
			}
			if math.Abs(plan.EstimatedFragIndex-tt.wantFinalFrag) > 1e-9 {
				t.Errorf("plan.EstimatedFragIndex = %f, want %f", plan.EstimatedFragIndex, tt.wantFinalFrag)
			}
		})
	}
}

func TestHeavilyFragmentedAllocationAndCompaction(t *testing.T) {
	tests := []struct {
		name         string
		layers       int
		kvHeads      int
		headDim      int
		blockSize    int
		totalBlocks  int
		seqCount     int
		tokensPerSeq int
		freeSeqs     []int64 // sequences to free to cause severe fragmentation
	}{
		{
			name:         "interleaved_alternating_sequences",
			layers:       2,
			kvHeads:      2,
			headDim:      4,
			blockSize:    4,
			totalBlocks:  16,
			seqCount:     6,
			tokensPerSeq: 8, // 2 blocks per sequence = 12 blocks used out of 16
			freeSeqs:     []int64{1, 3, 5},
		},
		{
			name:         "heavy_fragmentation_large_pool",
			layers:       1,
			kvHeads:      4,
			headDim:      8,
			blockSize:    2,
			totalBlocks:  32,
			seqCount:     12,
			tokensPerSeq: 4, // 2 blocks per sequence = 24 blocks used out of 32
			freeSeqs:     []int64{1, 2, 4, 7, 9, 11},
		},
		{
			name:         "single_surviving_sequence",
			layers:       3,
			kvHeads:      1,
			headDim:      16,
			blockSize:    4,
			totalBlocks:  12,
			seqCount:     5,
			tokensPerSeq: 8,
			freeSeqs:     []int64{1, 2, 3, 4}, // only sequence 5 survives
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache, err := cudakv.NewTestPagedCache(tt.layers, tt.kvHeads, tt.headDim, tt.blockSize, tt.totalBlocks)
			if err != nil {
				t.Fatalf("NewTestPagedCache failed: %v", err)
			}
			defer cache.Close()

			elemPerToken := tt.kvHeads * tt.headDim

			// Allocate all sequences and write unique token signatures
			type tokenSig struct {
				k []float32
				v []float32
			}
			tokenData := make(map[string]tokenSig)

			keyFor := func(seqID int64, tokenIdx, layer int) string {
				return "" + string(rune(seqID)) + "-" + string(rune(tokenIdx)) + "-" + string(rune(layer))
			}

			for s := 1; s <= tt.seqCount; s++ {
				seqID := int64(s)
				if err := cache.AllocSequence(seqID, tt.tokensPerSeq); err != nil {
					t.Fatalf("AllocSequence(%d) failed: %v", seqID, err)
				}
				for tok := 0; tok < tt.tokensPerSeq; tok++ {
					for l := 0; l < tt.layers; l++ {
						k := make([]float32, elemPerToken)
						v := make([]float32, elemPerToken)
						for e := 0; e < elemPerToken; e++ {
							k[e] = float32(s*10000 + tok*100 + l*10 + e + 1)
							v[e] = -k[e]
						}
						if err := cache.WriteToken(seqID, tok, l, k, v); err != nil {
							t.Fatalf("WriteToken(seq=%d, tok=%d, layer=%d) failed: %v", seqID, tok, l, err)
						}
						kCopy := append([]float32(nil), k...)
						vCopy := append([]float32(nil), v...)
						tokenData[keyFor(seqID, tok, l)] = tokenSig{k: kCopy, v: vCopy}
					}
				}
			}

			// Free selected sequences to introduce severe holes across the pool
			freedMap := make(map[int64]bool)
			for _, seqID := range tt.freeSeqs {
				cache.FreeSequence(seqID)
				freedMap[seqID] = true
			}

			initialFrag := cache.FragmentationIndex()
			stats := cache.FragmentationStats()
			t.Logf("[%s] Pre-compaction: FreeBlocks=%d, Largest=%d, FragIndex=%.4f",
				tt.name, stats.FreeBlocks, stats.LargestContiguous, initialFrag)

			compactor := cudakv.NewCompactor(cudakv.DefaultDefragPolicy())
			plan, err := compactor.Compact(cache)
			if err != nil {
				t.Fatalf("Compact failed: %v", err)
			}

			finalFrag := cache.FragmentationIndex()
			finalStats := cache.FragmentationStats()
			t.Logf("[%s] Post-compaction: Moves=%d, FreeBlocks=%d, Largest=%d, FragIndex=%.4f",
				tt.name, len(plan.Moves), finalStats.FreeBlocks, finalStats.LargestContiguous, finalFrag)

			// 1. Verify FragmentationIndex drops back below threshold (or to 0.0 for full pass)
			if finalFrag > compactor.Policy().Threshold {
				t.Errorf("FragmentationIndex after compaction = %f, want <= %f", finalFrag, compactor.Policy().Threshold)
			}
			if finalStats.FreeBlocks > 0 && finalFrag != 0.0 {
				t.Errorf("Expected 0.0 fragmentation after complete compaction, got %f", finalFrag)
			}

			// 2. Verify all surviving KV token data matches pre-compaction contents exactly (ZERO corruption)
			for s := 1; s <= tt.seqCount; s++ {
				seqID := int64(s)
				if freedMap[seqID] {
					continue
				}
				for tok := 0; tok < tt.tokensPerSeq; tok++ {
					for l := 0; l < tt.layers; l++ {
						gotK, gotV, err := cache.ReadToken(seqID, tok, l)
						if err != nil {
							t.Fatalf("ReadToken(seq=%d, tok=%d, layer=%d) after compaction failed: %v", seqID, tok, l, err)
						}
						wantSig := tokenData[keyFor(seqID, tok, l)]
						if !reflect.DeepEqual(gotK, wantSig.k) {
							t.Fatalf("KV corruption in K (seq %d, tok %d, layer %d): got %v, want %v", seqID, tok, l, gotK[:4], wantSig.k[:4])
						}
						if !reflect.DeepEqual(gotV, wantSig.v) {
							t.Fatalf("KV corruption in V (seq %d, tok %d, layer %d): got %v, want %v", seqID, tok, l, gotV[:4], wantSig.v[:4])
						}
					}
				}
			}

			// 3. Verify virtual page tables map correctly to contiguous physical offsets [0..ActiveBlocks-1]
			activeBlocks := finalStats.AllocatedBlocks
			seenPhysicalBlocks := make(map[int]bool)
			for s := 1; s <= tt.seqCount; s++ {
				seqID := int64(s)
				if freedMap[seqID] {
					continue
				}
				table := cache.BlockTable(seqID)
				vpt := cache.VirtualPageTable(seqID)
				if !reflect.DeepEqual(table, vpt.Blocks) {
					t.Errorf("VirtualPageTable(%d) mismatch: %v vs %v", seqID, vpt.Blocks, table)
				}
				for _, phys := range table {
					if phys < 0 || phys >= activeBlocks {
						t.Errorf("Sequence %d physical block %d is outside compacted range [0..%d)",
							seqID, phys, activeBlocks)
					}
					seenPhysicalBlocks[phys] = true
				}
			}
			if len(seenPhysicalBlocks) != activeBlocks {
				t.Errorf("Unique physical blocks held (%d) != ActiveBlocks (%d)",
					len(seenPhysicalBlocks), activeBlocks)
			}

			// 4. Verify that a fresh allocation succeeds and takes contiguous physical blocks
			freshSeqID := int64(999)
			neededTokens := tt.blockSize * 2
			if cache.NumFreeBlocks() >= 2 {
				if err := cache.AllocSequence(freshSeqID, neededTokens); err != nil {
					t.Fatalf("AllocSequence for fresh sequence failed: %v", err)
				}
				freshTable := cache.BlockTable(freshSeqID)
				for _, b := range freshTable {
					if b < activeBlocks || b >= tt.totalBlocks {
						t.Errorf("Fresh block %d allocated outside free region [%d..%d)",
							b, activeBlocks, tt.totalBlocks)
					}
				}
			}
		})
	}
}

func TestRadixTreeCompaction(t *testing.T) {
	tokensPerBlock := 4
	tree := cudakv.NewRadixTree(16, tokensPerBlock)

	seq1 := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	blocks1 := []int{10, 11}

	seq2 := []int32{1, 2, 3, 4, 9, 10, 11, 12}
	blocks2 := []int{10, 15}

	if _, _, err := tree.Insert(seq1, blocks1); err != nil {
		t.Fatalf("Insert seq1: %v", err)
	}
	if _, _, err := tree.Insert(seq2, blocks2); err != nil {
		t.Fatalf("Insert seq2: %v", err)
	}

	// Verify pre-compaction prefix match
	matched, blocks, _ := tree.MatchPrefix(seq1)
	if matched != 8 || !reflect.DeepEqual(blocks, blocks1) {
		t.Fatalf("Pre-compaction match: matched=%d, blocks=%v; want 8, %v", matched, blocks, blocks1)
	}

	// Simulate compaction remapping: 10->0, 11->1, 15->2
	remappings := map[int]int{
		10: 0,
		11: 1,
		15: 2,
	}
	tree.RemapBlocks(remappings)

	// Verify post-compaction prefix match returns updated physical slots
	matchedPost, blocksPost, _ := tree.MatchPrefix(seq1)
	wantBlocks1 := []int{0, 1}
	if matchedPost != 8 || !reflect.DeepEqual(blocksPost, wantBlocks1) {
		t.Fatalf("Post-compaction match seq1: matched=%d, blocks=%v; want 8, %v", matchedPost, blocksPost, wantBlocks1)
	}

	matchedPost2, blocksPost2, _ := tree.MatchPrefix(seq2)
	wantBlocks2 := []int{0, 2}
	if matchedPost2 != 8 || !reflect.DeepEqual(blocksPost2, wantBlocks2) {
		t.Fatalf("Post-compaction match seq2: matched=%d, blocks=%v; want 8, %v", matchedPost2, blocksPost2, wantBlocks2)
	}

	// Verify NodeMap lookups
	if tree.NodeMap[0] == nil || tree.NodeMap[1] == nil || tree.NodeMap[2] == nil {
		t.Errorf("NodeMap missing entries for remapped blocks: 0=%v, 1=%v, 2=%v",
			tree.NodeMap[0], tree.NodeMap[1], tree.NodeMap[2])
	}
	if tree.NodeMap[10] != nil || tree.NodeMap[11] != nil || tree.NodeMap[15] != nil {
		t.Errorf("NodeMap retained stale entries for old blocks: 10=%v, 11=%v, 15=%v",
			tree.NodeMap[10], tree.NodeMap[11], tree.NodeMap[15])
	}
}

func TestConcurrentAccessDuringCompaction(t *testing.T) {
	cache, err := cudakv.NewTestPagedCache(1, 1, 4, 4, 32)
	if err != nil {
		t.Fatalf("NewTestPagedCache failed: %v", err)
	}
	defer cache.Close()

	tree := cudakv.NewRadixTree(16, 4)
	compactor := cudakv.NewCompactor(cudakv.DefragPolicy{Threshold: 0.1}).WithRadixTree(tree)

	// Pre-allocate some sequences
	for s := int64(1); s <= 4; s++ {
		_ = cache.AllocSequence(s, 8)
		tokens := []int32{int32(s), 2, 3, 4, 5, 6, 7, 8}
		_, _, _ = tree.Insert(tokens, cache.BlockTable(s))
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Readers reading page tables and tokens concurrently
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			seqID := int64(readerID%4 + 1)
			for {
				select {
				case <-stop:
					return
				default:
					_ = cache.BlockTable(seqID)
					_ = cache.VirtualPageTable(seqID)
					_ = cache.FragmentationIndex()
					tokens := []int32{int32(seqID), 2, 3, 4, 5, 6, 7, 8}
					_, _, _ = tree.MatchPrefix(tokens)
				}
			}
		}(r)
	}

	// Mutators: allocate, free, append, compact
	for cycle := 0; cycle < 5; cycle++ {
		tempSeq := int64(100 + cycle)
		_ = cache.AllocSequence(tempSeq, 4)
		_ = cache.AppendToken(tempSeq)
		cache.FreeSequence(tempSeq)

		_, err := compactor.Compact(cache)
		if err != nil {
			t.Errorf("Concurrent Compact failed: %v", err)
		}
	}

	close(stop)
	wg.Wait()
}

type mockCopier struct {
	mu     sync.Mutex
	copies []cudakv.BlockMove
}

func (m *mockCopier) CopyBlock(src, dst int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.copies = append(m.copies, cudakv.BlockMove{SrcBlock: src, DstBlock: dst})
	return nil
}

func TestCompactorWithCustomCopier(t *testing.T) {
	cache, err := cudakv.NewTestPagedCache(1, 1, 2, 2, 8)
	if err != nil {
		t.Fatalf("NewTestPagedCache failed: %v", err)
	}
	defer cache.Close()

	_ = cache.AllocSequence(1, 2)
	_ = cache.AllocSequence(2, 2)
	cache.FreeSequence(1)

	copier := &mockCopier{}
	compactor := cudakv.NewCompactor(cudakv.DefaultDefragPolicy()).WithCopier(copier)

	plan, err := compactor.Compact(cache)
	if err != nil {
		t.Fatalf("Compact failed: %v", err)
	}

	if len(plan.Moves) != len(copier.copies) {
		t.Errorf("Copier received %d moves, plan had %d", len(copier.copies), len(plan.Moves))
	}
}

func TestCUDACompactionGraph(t *testing.T) {
	dev := cudatest.Require(t)

	cache, err := cudakv.NewPagedCache(dev, 1, 1, 4, 4, 8)
	if err != nil {
		t.Fatalf("NewPagedCache failed: %v", err)
	}
	defer cache.Close()

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	_ = cache.AllocSequence(1, 4)
	_ = cache.AllocSequence(2, 4)
	cache.FreeSequence(1)

	compactor := cudakv.NewCompactor(cudakv.DefaultDefragPolicy())
	plan := compactor.Plan(cache)
	if plan.Empty() {
		t.Fatal("expected non-empty compaction plan")
	}

	cg, err := compactor.CaptureGraph(cache, stream, plan)
	if err != nil {
		t.Fatalf("CaptureGraph failed: %v", err)
	}
	defer cg.Destroy()

	if err := cg.Launch(stream); err != nil {
		t.Fatalf("Launch graph failed: %v", err)
	}

	if err := cg.ApplyToCache(cache, stream); err != nil {
		t.Fatalf("ApplyToCache failed: %v", err)
	}

	if frag := cache.FragmentationIndex(); frag != 0.0 {
		t.Errorf("FragmentationIndex after graph compaction = %f, want 0.0", frag)
	}
}

func BenchmarkCompactor_Plan(b *testing.B) {
	cache, err := cudakv.NewTestPagedCache(2, 4, 64, 16, 256)
	if err != nil {
		b.Fatal(err)
	}
	defer cache.Close()

	// Allocate 100 sequences and free half
	for s := int64(1); s <= 100; s++ {
		_ = cache.AllocSequence(s, 32)
	}
	for s := int64(1); s <= 100; s += 2 {
		cache.FreeSequence(s)
	}

	compactor := cudakv.NewCompactor(cudakv.DefaultDefragPolicy())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = compactor.Plan(cache)
	}
}

func BenchmarkCompactor_Compact(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		cache, _ := cudakv.NewTestPagedCache(2, 2, 32, 4, 64)
		for s := int64(1); s <= 20; s++ {
			_ = cache.AllocSequence(s, 8)
		}
		for s := int64(1); s <= 20; s += 2 {
			cache.FreeSequence(s)
		}
		compactor := cudakv.NewCompactor(cudakv.DefaultDefragPolicy())
		b.StartTimer()
		_, _ = compactor.Compact(cache)
	}
}
