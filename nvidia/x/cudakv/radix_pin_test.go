// SPDX-License-Identifier: Apache-2.0

package cudakv

import "testing"

// Splitting a node a caller holds must leave the caller holding only what it
// acquired: releasing it has to make the whole path evictable again.
func TestRadixTreeReleaseAfterSplitUnpinsEverything(t *testing.T) {
	r := NewRadixTree(64, 2)
	if _, err := r.Insert([]int32{1, 2, 3, 4}, []int{10, 11}); err != nil {
		t.Fatal(err)
	}
	_, _, node := r.Acquire([]int32{1, 2, 3, 4})
	if node == nil {
		t.Fatal("Acquire matched nothing")
	}
	// Diverges inside the acquired segment, so the node is split under it.
	if _, err := r.Insert([]int32{1, 2, 7, 8}, []int{10, 12}); err != nil {
		t.Fatal(err)
	}
	r.Release(node)

	n := r.Len()
	r.Evict(n)
	if got := r.Len(); got != 0 {
		t.Errorf("after Release, Evict left %d of %d nodes; want 0", got, n)
	}
}

// Releasing more often than acquiring must not drive the held count below
// zero: a later split would then hand the tail more holders than exist and
// pin it for good.
func TestRadixTreeExtraReleaseDoesNotOverPin(t *testing.T) {
	r := NewRadixTree(64, 2)
	if _, err := r.Insert([]int32{1, 2, 3, 4}, []int{10, 11}); err != nil {
		t.Fatal(err)
	}
	_, _, node := r.Acquire([]int32{1, 2, 3, 4})
	if node == nil {
		t.Fatal("Acquire matched nothing")
	}
	r.Release(node)
	r.Release(node)

	if _, err := r.Insert([]int32{1, 2, 7, 8}, []int{10, 12}); err != nil {
		t.Fatal(err)
	}
	n := r.Len()
	r.Evict(n)
	if got := r.Len(); got != 0 {
		t.Errorf("after an extra Release, Evict left %d of %d nodes; want 0", got, n)
	}
}

// Splitting a node a caller holds must not release it: the caller acquired the
// whole segment, so every block of it stays live until the caller releases.
func TestRadixTreeSplitKeepsHeldBlocks(t *testing.T) {
	r := NewRadixTree(64, 2)
	if _, err := r.Insert([]int32{1, 2, 3, 4}, []int{10, 11}); err != nil {
		t.Fatal(err)
	}
	_, blocks, node := r.Acquire([]int32{1, 2, 3, 4})
	if node == nil {
		t.Fatal("Acquire matched nothing")
	}
	if _, err := r.Insert([]int32{1, 2, 7, 8}, []int{10, 12}); err != nil {
		t.Fatal(err)
	}
	held := map[int]bool{}
	for _, b := range blocks {
		held[b] = true
	}
	for _, b := range r.Evict(r.Len()) {
		if held[b] {
			t.Errorf("Evict freed block %d while it was held", b)
		}
	}
	// The held sequence must still match in full.
	if n, _, _ := r.MatchPrefix([]int32{1, 2, 3, 4}); n != 4 {
		t.Errorf("MatchPrefix after eviction = %d tokens, want 4", n)
	}
	r.Release(node)
}

// Splitting a node below the root has to leave the new prefix under that
// node's own parent, not under the root: reparenting it to the root would
// detach a whole subtree from the sequences that reach it.
func TestRadixTreeSplitBelowRoot(t *testing.T) {
	r := NewRadixTree(64, 2)
	// Two levels: {1,2} is shared, then {3,4,5,6} hangs below it.
	if _, err := r.Insert([]int32{1, 2, 3, 4, 5, 6}, []int{10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Insert([]int32{1, 2, 9, 9}, []int{10, 19}); err != nil {
		t.Fatal(err)
	}
	// Now split the second-level node {3,4,5,6} after its first block.
	if _, err := r.Insert([]int32{1, 2, 3, 4, 7, 7}, []int{10, 11, 17}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tokens []int32
		want   []int
	}{
		{[]int32{1, 2, 3, 4, 5, 6}, []int{10, 11, 12}},
		{[]int32{1, 2, 3, 4, 7, 7}, []int{10, 11, 17}},
		{[]int32{1, 2, 9, 9}, []int{10, 19}},
	} {
		n, blocks, _ := r.MatchPrefix(tc.tokens)
		if n != len(tc.tokens) || !equalInts(blocks, tc.want) {
			t.Errorf("MatchPrefix(%v) = %d tokens, blocks %v; want %d tokens, blocks %v",
				tc.tokens, n, blocks, len(tc.tokens), tc.want)
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
