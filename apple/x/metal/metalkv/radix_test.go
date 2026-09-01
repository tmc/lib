// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"reflect"
	"testing"
)

// insert stores a sequence and fails the test if the tree refuses it.
func insert(t *testing.T, tree *RadixTree, tokens []int32, blocks []int) {
	t.Helper()
	if err := tree.Insert(tokens, blocks); err != nil {
		t.Fatalf("Insert(%v, %v): %v", tokens, blocks, err)
	}
}

// checkMatch asserts on what a lookup returns: the matched token count and the
// blocks a caller would reuse.
func checkMatch(t *testing.T, tree *RadixTree, query []int32, wantLen int, wantBlocks []int) {
	t.Helper()
	gotLen, gotBlocks, _ := tree.MatchPrefix(query)
	if gotLen != wantLen || !reflect.DeepEqual(gotBlocks, wantBlocks) {
		t.Errorf("MatchPrefix(%v) = %d tokens, blocks %v; want %d tokens, blocks %v",
			query, gotLen, gotBlocks, wantLen, wantBlocks)
	}
}

func TestRadixTreePrefixMatch(t *testing.T) {
	tree := NewRadixTree(100, 2)
	insert(t, tree, []int32{10, 20, 30, 40}, []int{100, 101})

	// A longer query reuses the whole stored sequence.
	checkMatch(t, tree, []int32{10, 20, 30, 40, 50, 60}, 4, []int{100, 101})
	// The exact sequence matches exactly.
	checkMatch(t, tree, []int32{10, 20, 30, 40}, 4, []int{100, 101})
	// A shorter query matches the blocks it covers.
	checkMatch(t, tree, []int32{10, 20}, 2, []int{100})
	// An unrelated sequence matches nothing.
	checkMatch(t, tree, []int32{99, 98, 97, 96}, 0, []int{})
}

// TestRadixTreeSharedPrefix is the point of the structure: two sequences that
// share a prefix must return the same blocks for the shared part, and their own
// blocks past the divergence.
func TestRadixTreeSharedPrefix(t *testing.T) {
	tree := NewRadixTree(100, 2)
	insert(t, tree, []int32{1, 2, 3, 4, 5, 6}, []int{10, 11, 12})
	insert(t, tree, []int32{1, 2, 3, 4, 7, 8}, []int{10, 11, 20})

	// The shared four tokens come back from the first insert's blocks.
	checkMatch(t, tree, []int32{1, 2, 3, 4}, 4, []int{10, 11})
	// Each full sequence reaches its own tail block.
	checkMatch(t, tree, []int32{1, 2, 3, 4, 5, 6}, 6, []int{10, 11, 12})
	checkMatch(t, tree, []int32{1, 2, 3, 4, 7, 8}, 6, []int{10, 11, 20})
	// A third continuation reuses the prefix and stops at the divergence.
	checkMatch(t, tree, []int32{1, 2, 3, 4, 9, 9}, 4, []int{10, 11})

	// The shared prefix is stored once: the two sequences and the node they
	// diverge at make three segments.
	if got := tree.Len(); got != 3 {
		t.Errorf("Len() = %d, want 3 segments for two sequences sharing a prefix", got)
	}
}

// TestRadixTreeDivergenceWithinABlock covers two sequences whose first block
// differs, which must not be treated as a shared prefix even though they begin
// with the same token.
func TestRadixTreeDivergenceWithinABlock(t *testing.T) {
	tree := NewRadixTree(100, 2)
	insert(t, tree, []int32{1, 2, 3, 4}, []int{10, 11})
	insert(t, tree, []int32{1, 5, 3, 4}, []int{20, 21})

	checkMatch(t, tree, []int32{1, 2, 3, 4}, 4, []int{10, 11})
	checkMatch(t, tree, []int32{1, 5, 3, 4}, 4, []int{20, 21})
	// A query agreeing on only the first token of the block reuses nothing:
	// the block it would need was never computed.
	checkMatch(t, tree, []int32{1, 9, 3, 4}, 0, []int{})
}

// TestRadixTreePartialBlock covers a query that diverges partway through a
// block. The match stops at the last whole block, because a caller cannot reuse
// half of one.
func TestRadixTreePartialBlock(t *testing.T) {
	tree := NewRadixTree(100, 4)
	insert(t, tree, []int32{1, 2, 3, 4, 5, 6, 7, 8}, []int{10, 11})

	checkMatch(t, tree, []int32{1, 2, 3, 4, 5, 6, 9, 9}, 4, []int{10})
	checkMatch(t, tree, []int32{1, 2, 3, 9}, 0, []int{})
	checkMatch(t, tree, []int32{1, 2, 3, 4, 5, 6, 7, 8}, 8, []int{10, 11})
}

func TestRadixTreeInsertRejectsUnalignedInput(t *testing.T) {
	tree := NewRadixTree(100, 4)
	for _, tt := range []struct {
		name   string
		tokens []int32
		blocks []int
	}{
		{"empty", nil, nil},
		{"partial block", []int32{1, 2, 3}, []int{10}},
		{"too few blocks", []int32{1, 2, 3, 4, 5, 6, 7, 8}, []int{10}},
		{"too many blocks", []int32{1, 2, 3, 4}, []int{10, 11}},
	} {
		if err := tree.Insert(tt.tokens, tt.blocks); err == nil {
			t.Errorf("Insert(%s) was accepted", tt.name)
		}
	}
	if got := tree.Len(); got != 0 {
		t.Errorf("Len() = %d after only refused inserts, want 0", got)
	}
}

// TestRadixTreeEvictionFreesBlocks checks that eviction reports the blocks it
// frees and that the evicted sequence stops matching, while the sequence that
// was matched more recently still does.
func TestRadixTreeEvictionFreesBlocks(t *testing.T) {
	tree := NewRadixTree(100, 2)
	insert(t, tree, []int32{1, 2}, []int{10})
	insert(t, tree, []int32{3, 4}, []int{20})

	// Touch the first sequence so the second is the least recently used.
	checkMatch(t, tree, []int32{1, 2}, 2, []int{10})

	freed := tree.Evict(1)
	if !reflect.DeepEqual(freed, []int{20}) {
		t.Errorf("Evict(1) freed %v, want [20]", freed)
	}
	checkMatch(t, tree, []int32{3, 4}, 0, []int{})
	checkMatch(t, tree, []int32{1, 2}, 2, []int{10})
}

// TestRadixTreeEvictionKeepsLiveDescendants checks the reference counting: a
// prefix whose continuation is cached, or which a caller is holding, is not
// dropped out from under it.
func TestRadixTreeEvictionKeepsLiveDescendants(t *testing.T) {
	tree := NewRadixTree(100, 2)
	insert(t, tree, []int32{1, 2, 3, 4}, []int{10, 11})
	insert(t, tree, []int32{1, 2, 5, 6}, []int{10, 21})

	// Evicting everything unpinned removes the two tails but never the
	// shared prefix while they are there.
	freed := tree.Evict(1)
	if len(freed) != 1 {
		t.Fatalf("Evict(1) freed %v, want one block", freed)
	}
	// The shared prefix is still matchable after one tail went.
	checkMatch(t, tree, []int32{1, 2}, 2, []int{10})

	// Hold whichever tail survived; it and its prefix must now survive
	// eviction entirely.
	survivor := []int32{1, 2, 3, 4}
	if n, _, _ := tree.MatchPrefix(survivor); n != 4 {
		survivor = []int32{1, 2, 5, 6}
	}
	matched, blocks, node := tree.Acquire(survivor)
	if node == nil {
		t.Fatal("Acquire matched nothing; one of the two sequences should remain")
	}
	if matched != 4 || len(blocks) != 2 {
		t.Fatalf("Acquire(%v) matched %d tokens and blocks %v, want 4 tokens and two blocks",
			survivor, matched, blocks)
	}
	if freed := tree.Evict(10); len(freed) != 0 {
		t.Errorf("Evict freed %v while the sequence was held", freed)
	}
	checkMatch(t, tree, []int32{1, 2}, 2, []int{10})

	// Once released, it can go.
	tree.Release(node)
	if freed := tree.Evict(10); len(freed) == 0 {
		t.Error("Evict freed nothing after the sequence was released")
	}
	checkMatch(t, tree, []int32{1, 2}, 0, []int{})
}

// TestRadixTreeEvictsToStayWithinMaxNodes checks that a bounded tree drops its
// coldest entries rather than growing, and that what it keeps still matches.
func TestRadixTreeEvictsToStayWithinMaxNodes(t *testing.T) {
	tree := NewRadixTree(2, 2)
	insert(t, tree, []int32{1, 1}, []int{10})
	insert(t, tree, []int32{2, 2}, []int{20})
	insert(t, tree, []int32{3, 3}, []int{30})

	if got := tree.Len(); got > 2 {
		t.Errorf("Len() = %d, want at most the 2 segments the tree was built for", got)
	}
	checkMatch(t, tree, []int32{3, 3}, 2, []int{30})
	checkMatch(t, tree, []int32{1, 1}, 0, []int{})
}

// TestRadixTreeNodeMap checks the block index, which lets a caller go from a
// block id back to what is cached in it.
func TestRadixTreeNodeMap(t *testing.T) {
	tree := NewRadixTree(100, 2)
	insert(t, tree, []int32{1, 2, 3, 4}, []int{10, 11})

	node, ok := tree.NodeMap[11]
	if !ok {
		t.Fatal("block 11 is not indexed")
	}
	if !reflect.DeepEqual(node.Tokens, []int32{1, 2, 3, 4}) {
		t.Errorf("block 11 belongs to a node holding %v, want the whole sequence", node.Tokens)
	}

	// Splitting the node moves the tail block to the new node.
	insert(t, tree, []int32{1, 2, 5, 6}, []int{10, 21})
	if got := tree.NodeMap[11].Tokens; !reflect.DeepEqual(got, []int32{3, 4}) {
		t.Errorf("after the split, block 11 belongs to a node holding %v, want [3 4]", got)
	}
	if got := tree.NodeMap[10].Tokens; !reflect.DeepEqual(got, []int32{1, 2}) {
		t.Errorf("after the split, block 10 belongs to a node holding %v, want [1 2]", got)
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
