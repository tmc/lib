// SPDX-License-Identifier: Apache-2.0

package kvcache_test

import (
	"testing"

	"github.com/tmc/lib/lm/kvcache"
)

func TestRadixTreePrefixMatch(t *testing.T) {
	tree := kvcache.NewRadixTree(100)
	tree.Insert([]int32{1, 2, 3, 4}, []int{10, 11})

	matched, blocks, _ := tree.MatchPrefix([]int32{1, 2, 3, 4, 5, 6})
	if matched != 4 {
		t.Errorf("matched = %d, want 4", matched)
	}
	if len(blocks) != 2 || blocks[0] != 10 || blocks[1] != 11 {
		t.Errorf("matched blocks = %v, want [10 11]", blocks)
	}
}

func TestRollingKVCacheSlotMapping(t *testing.T) {
	cache, err := kvcache.NewRollingKVCache(kvcache.SlidingWindowConfig{
		SinkTokens: 4,
		WindowSize: 8,
	})
	if err != nil {
		t.Fatalf("NewRollingKVCache failed: %v", err)
	}

	for i := 0; i < 4; i++ {
		if slot := cache.PhysicalSlot(i); slot != i {
			t.Errorf("sink token %d slot = %d, want %d", i, slot, i)
		}
	}

	if slot := cache.PhysicalSlot(12); slot != 4 {
		t.Errorf("rolled token 12 slot = %d, want 4", slot)
	}
}

func TestChunkedPrefillScheduler(t *testing.T) {
	sched := kvcache.NewChunkedPrefillScheduler(512, 1024)

	prefills := []*kvcache.ChunkedPrefillRequest{
		{
			SeqID:        "long_prompt",
			PromptTokens: make([]int32, 1200),
		},
	}
	decodes := []string{"req_decode_1"}

	batch := sched.Schedule(prefills, decodes)
	if len(batch.DecodeSeqIDs) != 1 {
		t.Errorf("expected 1 decode request scheduled")
	}
	if len(batch.PrefillChunks) != 1 || len(batch.PrefillChunks[0].Tokens) != 512 {
		t.Errorf("expected chunk of 512 tokens scheduled")
	}
}
