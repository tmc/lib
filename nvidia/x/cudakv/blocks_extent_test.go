// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"errors"
	"reflect"
	"testing"
)

func TestBlockAllocatorSafeCeilingAtIntLimit(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	a := NewBlockAllocator(maxInt, 1)
	if err := a.AllocSequence(1, nil, maxInt); err != nil {
		t.Fatalf("AllocSequence(maxInt) = %v; want success with one block", err)
	}
	if got := a.Table(1); !reflect.DeepEqual(got, []int{0}) {
		t.Fatalf("Table(1) = %v; want [0]", got)
	}
	if got, ok := a.SequenceLength(1); !ok || got != maxInt {
		t.Fatalf("SequenceLength(1) = %d, %v; want %d, true", got, ok, maxInt)
	}
}

func TestBlockAllocatorRejectsSharedExtentOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	blockSize := maxInt/2 + 1
	a := NewBlockAllocator(blockSize, 2)
	for seqID := int64(1); seqID <= 2; seqID++ {
		if err := a.AllocSequence(seqID, nil, 1); err != nil {
			t.Fatal(err)
		}
	}
	shared := []int{a.Table(1)[0], a.Table(2)[0]}
	beforeHolders := append([]int(nil), a.holders...)
	beforeFree := make([]int, len(a.free))
	copy(beforeFree, a.free)
	if err := a.AllocSequence(3, shared, maxInt); err == nil {
		t.Fatal("AllocSequence accepted shared extent larger than maxInt")
	}
	if !reflect.DeepEqual(a.holders, beforeHolders) {
		t.Fatalf("holders changed after refusal: got %v, want %v", a.holders, beforeHolders)
	}
	if !reflect.DeepEqual(a.free, beforeFree) {
		t.Fatalf("free list changed after refusal: got %v, want %v", a.free, beforeFree)
	}
	if _, ok := a.SequenceLength(3); ok {
		t.Fatal("sequence 3 was allocated after refusal")
	}
}

func TestBlockAllocatorRejectsUnrepresentableBlockTables(t *testing.T) {
	t.Run("constructor byte extent", func(t *testing.T) {
		totalBlocks := maxBlockSliceLen() + 1
		a := NewBlockAllocator(1, totalBlocks)
		err := a.AllocSequence(1, nil, 1)
		if err == nil {
			t.Fatalf("NewBlockAllocator(1, %d) admitted an unrepresentable block slice", totalBlocks)
		}
	})

	t.Run("sequence table extent", func(t *testing.T) {
		a := NewBlockAllocator(1, 1)
		tokens := maxBlockSliceLen() + 1
		if err := a.AllocSequence(1, nil, tokens); err == nil {
			t.Fatalf("AllocSequence(%d) admitted an unrepresentable block table", tokens)
		}
		if len(a.holders) != 1 || len(a.free) != 1 || len(a.tables) != 0 || len(a.lens) != 0 {
			t.Fatalf("allocator changed after refused extent: holders=%v free=%v tables=%v lens=%v", a.holders, a.free, a.tables, a.lens)
		}
	})
}

func TestBlockAllocatorRejectsNegativeTokenExtent(t *testing.T) {
	a := NewBlockAllocator(4, 2)
	beforeFree := make([]int, len(a.free))
	copy(beforeFree, a.free)
	if err := a.AllocSequence(1, nil, -1); err == nil {
		t.Fatal("AllocSequence(-1) succeeded")
	}
	if len(a.tables) != 0 || len(a.lens) != 0 || !reflect.DeepEqual(a.free, beforeFree) {
		t.Fatal("allocator state changed after refusing a negative token extent")
	}
}

func TestBlockAllocatorZeroValueNeedsConstructor(t *testing.T) {
	var a BlockAllocator
	err := a.AllocSequence(1, nil, 1)
	if !errors.Is(err, ErrBlockAllocatorUninitialized) {
		t.Fatalf("zero-value AllocSequence error = %v; want ErrBlockAllocatorUninitialized", err)
	}
	if err := a.AppendToken(1); !errors.Is(err, ErrBlockAllocatorUninitialized) {
		t.Fatalf("zero-value AppendToken error = %v; want ErrBlockAllocatorUninitialized", err)
	}
}

func TestBlockAllocatorAppendTokenCountOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	a := NewBlockAllocator(2, 1)
	if err := a.AllocSequence(1, nil, 0); err != nil {
		t.Fatal(err)
	}
	// Seed the stored length directly; reaching maxInt with public appends would
	// require an unbounded number of calls and blocks.
	a.lens[1] = maxInt
	beforeTable := a.Table(1)
	beforeHolders := append([]int(nil), a.holders...)
	beforeFree := append([]int(nil), a.free...)
	if err := a.AppendToken(1); err == nil {
		t.Fatal("AppendToken at maxInt succeeded")
	}
	if got := a.lens[1]; got != maxInt {
		t.Fatalf("sequence length changed to %d; want %d", got, maxInt)
	}
	if !reflect.DeepEqual(a.Table(1), beforeTable) || !reflect.DeepEqual(a.holders, beforeHolders) || !reflect.DeepEqual(a.free, beforeFree) {
		t.Fatal("allocator state changed after token-count overflow refusal")
	}
}

func TestBlockAllocatorZeroTokenSequenceStillWorks(t *testing.T) {
	a := NewBlockAllocator(4, 1)
	if err := a.AllocSequence(1, nil, 0); err != nil {
		t.Fatal(err)
	}
	if got := a.Table(1); len(got) != 0 {
		t.Fatalf("Table(1) = %v; want empty", got)
	}
	if err := a.AppendToken(1); err != nil {
		t.Fatalf("AppendToken after zero-token allocation: %v", err)
	}
	if got, ok := a.SequenceLength(1); !ok || got != 1 {
		t.Fatalf("SequenceLength(1) = %d, %v; want 1, true", got, ok)
	}
}
