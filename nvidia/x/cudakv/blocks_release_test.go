// SPDX-License-Identifier: Apache-2.0

package cudakv_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func TestBlockAllocatorReleaseRefusalIsTransactional(t *testing.T) {
	tests := []struct {
		name   string
		blocks func(held, other, free int) []int
	}{
		{"duplicate overdraw", func(b, other, free int) []int { return []int{b, b} }},
		{"duplicate after valid block", func(b, other, free int) []int { return []int{other, b, b} }},
		{"negative after valid block", func(b, other, free int) []int { return []int{b, -1} }},
		{"out of range after valid block", func(b, other, free int) []int { return []int{b, 3} }},
		{"free after valid block", func(b, other, free int) []int { return []int{b, free} }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := cudakv.NewBlockAllocator(1, 3)
			if err := a.AllocSequence(1, nil, 2); err != nil {
				t.Fatal(err)
			}
			table := a.Table(1)
			free := -1
			before := make([]int, a.TotalBlocks())
			for b := range before {
				var err error
				before[b], err = a.Holders(b)
				if err != nil {
					t.Fatal(err)
				}
				if before[b] == 0 {
					free = b
				}
			}
			if err := a.Release(tt.blocks(table[0], table[1], free)); err == nil {
				t.Fatal("Release accepted invalid block counts")
			}
			if a.NumFreeBlocks() != 1 || !reflect.DeepEqual(a.Table(1), table) {
				t.Fatal("refusal changed free blocks or live table")
			}
			if n, ok := a.SequenceLength(1); !ok || n != 2 {
				t.Fatalf("refusal changed sequence length: %d, %v", n, ok)
			}
			for b, want := range before {
				if got, err := a.Holders(b); err != nil || got != want {
					t.Fatalf("block %d after refusal: holders %d, error %v; want %d", b, got, err, want)
				}
			}
			a.FreeSequence(1)
			if a.NumFreeBlocks() != 3 {
				t.Fatal("refusal corrupted later sequence cleanup")
			}
		})
	}
}

func TestBlockAllocatorReleaseMultiplicity(t *testing.T) {
	a := cudakv.NewBlockAllocator(1, 2)
	if err := a.AllocSequence(1, nil, 2); err != nil {
		t.Fatal(err)
	}
	table := a.Table(1)
	duplicates := []int{table[0], table[1], table[0], table[1]}
	if err := a.Retain(duplicates); err != nil {
		t.Fatal(err)
	}
	if err := a.Release(duplicates); err != nil {
		t.Fatalf("balanced duplicate release: %v", err)
	}
	for _, b := range table {
		if n, _ := a.Holders(b); n != 1 {
			t.Fatalf("block %d has %d holders; want sequence's one holder", b, n)
		}
	}
	if a.NumFreeBlocks() != 0 {
		t.Fatal("balanced release freed live sequence storage")
	}
	a.FreeSequence(1)
	if a.NumFreeBlocks() != 2 {
		t.Fatal("sequence cleanup did not free each block once")
	}
	if err := a.Release(nil); err != nil {
		t.Fatalf("empty release: %v", err)
	}
}

func TestBlockAllocatorReleaseFinalDuplicateHolders(t *testing.T) {
	a := cudakv.NewBlockAllocator(4, 1)
	if err := a.AllocSequence(1, nil, 1); err != nil {
		t.Fatal(err)
	}
	b := a.Table(1)[0]
	if err := a.Retain([]int{b, b}); err != nil {
		t.Fatal(err)
	}
	a.FreeSequence(1)
	if err := a.Release([]int{b, b}); err != nil {
		t.Fatal(err)
	}
	if n, _ := a.Holders(b); n != 0 || a.NumFreeBlocks() != 1 {
		t.Fatalf("final duplicate release: holders %d, free %d; want 0, 1", n, a.NumFreeBlocks())
	}
	if err := a.AllocSequence(2, nil, 1); err != nil {
		t.Fatal(err)
	}
	if err := a.AllocSequence(3, nil, 1); err == nil {
		t.Fatal("final duplicate release put the same block on the free list twice")
	}
}

func TestBlockAllocatorReleasePreservesSharedSequence(t *testing.T) {
	a := cudakv.NewBlockAllocator(4, 3)
	if err := a.AllocSequence(1, nil, 8); err != nil {
		t.Fatal(err)
	}
	b := a.Table(1)[0]
	if err := a.Retain([]int{b, b}); err != nil {
		t.Fatal(err)
	}
	if err := a.AllocSequence(2, []int{b}, 8); err != nil {
		t.Fatal(err)
	}
	if err := a.Release([]int{b, b}); err != nil {
		t.Fatal(err)
	}
	if n, _ := a.Holders(b); n != 2 || a.NumFreeBlocks() != 0 {
		t.Fatalf("shared release: holders %d, free %d; want 2, 0", n, a.NumFreeBlocks())
	}
	a.FreeSequence(1)
	if n, _ := a.Holders(b); n != 1 || a.NumFreeBlocks() != 1 {
		t.Fatalf("original sequence ended: holders %d, free %d; want 1, 1", n, a.NumFreeBlocks())
	}
	a.FreeSequence(2)
	if a.NumFreeBlocks() != 3 {
		t.Fatal("last sequence did not release all blocks")
	}
}

func ExampleBlockAllocator_Release() {
	a := cudakv.NewBlockAllocator(4, 1)
	if err := a.AllocSequence(1, nil, 1); err != nil {
		fmt.Println(err)
		return
	}
	b := a.Table(1)[0]
	if err := a.Retain([]int{b, b}); err != nil {
		fmt.Println(err)
		return
	}
	err := a.Release([]int{b, b})
	holders, _ := a.Holders(b)
	fmt.Println(err == nil, holders, a.NumFreeBlocks())
	// Output: true 1 0
}
