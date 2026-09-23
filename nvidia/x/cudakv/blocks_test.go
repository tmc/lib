// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"reflect"
	"testing"
)

func TestBlockAllocatorSharedPrefixOutlivesSequences(t *testing.T) {
	a := newBlockAllocator(16, 8)

	// Sequence 1 writes 40 tokens: three blocks, the first two full.
	if err := a.allocSequence(1, nil, 40); err != nil {
		t.Fatal(err)
	}
	table := a.table(1)
	if len(table) != 3 || len(a.free) != 5 {
		t.Fatalf("table %v, free %d; want 3 blocks and 5 free", table, len(a.free))
	}

	// A prefix cache keeps the two full blocks; the sequence then ends.
	prefix := table[:2]
	if err := a.retain(prefix); err != nil {
		t.Fatal(err)
	}
	a.freeSequence(1)
	if len(a.free) != 6 {
		t.Fatalf("free %d after freeing a sequence with 2 retained blocks; want 6", len(a.free))
	}
	for _, b := range prefix {
		if a.holders[b] != 1 {
			t.Fatalf("block %d holders %d after sequence end; want 1 (the cache)", b, a.holders[b])
		}
	}

	// Sequence 2 starts on the shared prefix and needs one fresh block for
	// tokens 32..39.
	if err := a.allocSequence(2, prefix, 40); err != nil {
		t.Fatal(err)
	}
	if got := a.table(2)[:2]; !reflect.DeepEqual(got, prefix) {
		t.Fatalf("table(2) starts %v; want the shared prefix %v", got, prefix)
	}
	if len(a.free) != 5 {
		t.Fatalf("free %d; want 5 (one fresh block)", len(a.free))
	}
	for _, b := range prefix {
		if a.holders[b] != 2 {
			t.Fatalf("shared block %d holders %d; want 2", b, a.holders[b])
		}
	}

	// The cache evicts the prefix while sequence 2 still reads it: the blocks
	// must not become free.
	if err := a.release(prefix); err != nil {
		t.Fatal(err)
	}
	if len(a.free) != 5 {
		t.Fatalf("free %d after evicting a prefix a live sequence shares; want 5", len(a.free))
	}
	for _, b := range prefix {
		if a.holders[b] != 1 {
			t.Fatalf("block %d holders %d; want 1 (sequence 2)", b, a.holders[b])
		}
	}

	// Sequence 2 ends: everything is free again, each block exactly once.
	a.freeSequence(2)
	if len(a.free) != 8 {
		t.Fatalf("free %d at the end; want 8", len(a.free))
	}
	seen := make(map[int]bool)
	for _, b := range a.free {
		if seen[b] {
			t.Fatalf("block %d is on the free list twice", b)
		}
		seen[b] = true
	}
}

func TestBlockAllocatorRejects(t *testing.T) {
	tests := []struct {
		name string
		run  func(a *blockAllocator) error
	}{
		{"retain free block", func(a *blockAllocator) error { return a.retain([]int{3}) }},
		{"release free block", func(a *blockAllocator) error { return a.release([]int{3}) }},
		{"retain out of range", func(a *blockAllocator) error { return a.retain([]int{8}) }},
		{"share free block", func(a *blockAllocator) error { return a.allocSequence(9, []int{3}, 16) }},
		{"prefix longer than prompt", func(a *blockAllocator) error {
			if err := a.allocSequence(1, nil, 32); err != nil {
				return nil
			}
			return a.allocSequence(2, a.table(1), 31)
		}},
		{"double alloc", func(a *blockAllocator) error {
			if err := a.allocSequence(1, nil, 1); err != nil {
				return nil
			}
			return a.allocSequence(1, nil, 1)
		}},
		{"append unknown", func(a *blockAllocator) error { return a.appendToken(7) }},
		{"pool exhausted", func(a *blockAllocator) error { return a.allocSequence(1, nil, 16*8+1) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := newBlockAllocator(16, 8)
			if err := tt.run(a); err == nil {
				t.Fatal("accepted")
			}
			if len(a.free) != 8 && tt.name != "prefix longer than prompt" && tt.name != "double alloc" {
				t.Fatalf("a rejected call changed the free list: %d free", len(a.free))
			}
		})
	}
}

func TestBlockAllocatorAppendCrossesBoundary(t *testing.T) {
	a := newBlockAllocator(16, 4)
	if err := a.allocSequence(1, nil, 30); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ { // 31, 32: still in block two
		if err := a.appendToken(1); err != nil {
			t.Fatal(err)
		}
	}
	if len(a.table(1)) != 2 {
		t.Fatalf("table %v after 32 tokens; want 2 blocks", a.table(1))
	}
	if err := a.appendToken(1); err != nil { // 33: third block
		t.Fatal(err)
	}
	if len(a.table(1)) != 3 || a.lens[1] != 33 {
		t.Fatalf("table %v, len %d after 33 tokens; want 3 blocks", a.table(1), a.lens[1])
	}
}

// TestBlockAllocatorWithRadixTree is the join this package's two halves exist
// for: the tree decides what to keep, the allocator decides when a block is
// free, and a sequence must never be handed a block another sequence's prefix
// still lives in.
func TestBlockAllocatorWithRadixTree(t *testing.T) {
	const bs = 4
	a := newBlockAllocator(bs, 6)
	tree := NewRadixTree(16, bs)

	seq := func(n int32) []int32 {
		s := make([]int32, 0, 12)
		for i := int32(0); i < 8; i++ { // shared 8-token prefix
			s = append(s, i)
		}
		return append(s, n, n, n, n) // distinct last block
	}

	// Sequence 1 computes 12 tokens and, when it finishes, publishes all three
	// blocks to the tree.
	tokens1 := seq(100)
	if err := a.allocSequence(1, nil, len(tokens1)); err != nil {
		t.Fatal(err)
	}
	adopted, _, err := tree.Insert(tokens1, a.table(1))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(adopted, a.table(1)) {
		t.Fatalf("tree adopted %v of a fresh insert; want all of %v", adopted, a.table(1))
	}
	if err := a.retain(adopted); err != nil {
		t.Fatal(err)
	}
	tail1 := a.table(1)
	a.freeSequence(1)
	if len(a.free) != 3 {
		t.Fatalf("free %d after publishing; want 3", len(a.free))
	}

	// Sequence 2 shares the prefix (two blocks) and computes its own last block.
	tokens2 := seq(200)
	matched, shared, node := tree.Acquire(tokens2)
	if matched != 8 || len(shared) != 2 {
		t.Fatalf("Acquire matched %d tokens, %v blocks; want 8 and two blocks", matched, shared)
	}
	if err := a.allocSequence(2, shared, len(tokens2)); err != nil {
		t.Fatal(err)
	}
	own := a.table(2)[2]

	// Sequence 2 publishes. The prefix is already stored, so the tree adopts
	// only the new last block; the sequence's shared blocks are not doubly
	// counted.
	adopted, _, err = tree.Insert(tokens2, a.table(2))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(adopted, []int{own}) {
		t.Fatalf("tree adopted %v; want only the new block %d", adopted, own)
	}
	if err := a.retain(adopted); err != nil {
		t.Fatal(err)
	}
	tree.Release(node)
	a.freeSequence(2)
	if len(a.free) != 2 {
		t.Fatalf("free %d with two published sequences sharing a prefix; want 2", len(a.free))
	}

	// The tree is squeezed and evicts the older leaf: sequence 1's last block
	// becomes free, the shared prefix does not.
	freed := tree.Evict(1)
	if !reflect.DeepEqual(freed, []int{tail1[2]}) {
		t.Fatalf("Evict(1) freed %v; want sequence 1's last block %d", freed, tail1[2])
	}
	if err := a.release(freed); err != nil {
		t.Fatal(err)
	}
	if len(a.free) != 3 {
		t.Fatalf("free %d after evicting the unshared leaf; want 3", len(a.free))
	}

	// Everything left allocated is held exactly once, by the tree.
	for b := range a.holders {
		want := 0
		if b == own || b == shared[0] || b == shared[1] {
			want = 1
		}
		if a.holders[b] != want {
			t.Fatalf("block %d holders %d; want %d", b, a.holders[b], want)
		}
	}

	// A third sequence must be handed only free blocks.
	if err := a.allocSequence(3, nil, 3*bs); err != nil {
		t.Fatal(err)
	}
	for _, b := range a.table(3) {
		if b == own || b == shared[0] || b == shared[1] {
			t.Fatalf("sequence 3 was handed block %d, which the tree still holds", b)
		}
	}
}
