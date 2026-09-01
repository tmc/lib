// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"testing"
)

func TestRadixTreePrefixMatch(t *testing.T) {
	tree := NewRadixTree(100)
	tree.Insert([]int32{10, 20, 30, 40}, []int{100, 101})

	matched, blocks, _ := tree.MatchPrefix([]int32{10, 20, 30, 40, 50})
	if matched != 4 {
		t.Errorf("matched = %d, want 4", matched)
	}
	if len(blocks) != 2 || blocks[0] != 100 || blocks[1] != 101 {
		t.Errorf("matched blocks = %v, want [100 101]", blocks)
	}
}

func TestChunkedPrefillScheduler(t *testing.T) {
	sched := NewChunkedPrefillScheduler(512, 1024)

	prefills := []*ChunkedPrefillRequest{
		{
			SeqID:        "cuda_long_prompt",
			PromptTokens: make([]int32, 1000),
		},
	}
	decodes := []string{"decode_req_0"}

	batch := sched.Schedule(prefills, decodes)
	if len(batch.DecodeSeqIDs) != 1 {
		t.Errorf("expected 1 decode request scheduled")
	}
	if len(batch.PrefillChunks) != 1 || len(batch.PrefillChunks[0].Tokens) != 512 {
		t.Errorf("expected 512 prefill chunk scheduled")
	}
}
