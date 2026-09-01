// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"testing"
)

func TestRadixTreePrefixMatch(t *testing.T) {
	tree := NewRadixTree(100)
	tree.Insert([]int32{1, 2, 3, 4}, []int{10, 11})

	matched, blocks, _ := tree.MatchPrefix([]int32{1, 2, 3, 4, 5, 6})
	if matched != 4 {
		t.Errorf("matched = %d, want 4", matched)
	}
	if len(blocks) != 2 || blocks[0] != 10 || blocks[1] != 11 {
		t.Errorf("matched blocks = %v, want [10 11]", blocks)
	}
}

func TestChunkedPrefillScheduler(t *testing.T) {
	sched := NewChunkedPrefillScheduler(512, 1024)

	prefills := []*ChunkedPrefillRequest{
		{
			SeqID:        "long_prompt",
			PromptTokens: make([]int32, 1200),
		},
	}
	decodes := []string{"req_decode_1", "req_decode_2"}

	batch1 := sched.Schedule(prefills, decodes)
	if len(batch1.DecodeSeqIDs) != 2 {
		t.Errorf("expected 2 decode requests scheduled")
	}
	if len(batch1.PrefillChunks) != 1 || len(batch1.PrefillChunks[0].Tokens) != 512 {
		t.Errorf("expected chunk of 512 tokens scheduled")
	}

	batch2 := sched.Schedule(prefills, nil)
	if len(batch2.PrefillChunks) != 1 || len(batch2.PrefillChunks[0].Tokens) != 512 {
		t.Errorf("expected second chunk of 512 tokens scheduled")
	}

	batch3 := sched.Schedule(prefills, nil)
	if len(batch3.PrefillChunks) != 1 || len(batch3.PrefillChunks[0].Tokens) != 176 {
		t.Errorf("expected final chunk of 176 tokens scheduled, got %d", len(batch3.PrefillChunks[0].Tokens))
	}
	if !batch3.PrefillChunks[0].IsLast {
		t.Errorf("expected final chunk to be marked IsLast")
	}
}
