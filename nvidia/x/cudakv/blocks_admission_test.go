// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"errors"
	"reflect"
	"testing"
)

func TestBlockAllocatorRejectsInvalidArguments(t *testing.T) {
	tests := []struct {
		name        string
		blockSize   int
		totalBlocks int
	}{
		{"zero block size", 0, 4},
		{"negative block size", -1, 4},
		{"negative total blocks", 16, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := NewBlockAllocator(tt.blockSize, tt.totalBlocks)
			allocErr := a.AllocSequence(1, nil, 1)
			if allocErr == nil {
				t.Fatalf("NewBlockAllocator(%d, %d).AllocSequence succeeded; want error", tt.blockSize, tt.totalBlocks)
			}
			if err := a.AppendToken(1); !errors.Is(err, allocErr) {
				t.Errorf("AppendToken = %v; want %v", err, allocErr)
			}
			if n := a.NumFreeBlocks(); n != 0 {
				t.Errorf("NumFreeBlocks = %d; want 0", n)
			}
		})
	}
}

func TestBlockAllocatorRejectsDuplicateShare(t *testing.T) {
	a := NewBlockAllocator(4, 8)
	if err := a.AllocSequence(1, nil, 8); err != nil {
		t.Fatal(err)
	}
	b := a.Table(1)[0]
	holders := append([]int(nil), a.holders...)
	free := append([]int(nil), a.free...)

	if err := a.AllocSequence(2, []int{b, b}, 8); err == nil {
		t.Fatalf("AllocSequence shared [%d %d] succeeded; table %v", b, b, a.Table(2))
	}
	if !reflect.DeepEqual(a.holders, holders) {
		t.Errorf("holders = %v after refusal; want %v", a.holders, holders)
	}
	if !reflect.DeepEqual(a.free, free) {
		t.Errorf("free = %v after refusal; want %v", a.free, free)
	}
	if _, ok := a.SequenceLength(2); ok {
		t.Errorf("sequence 2 allocated after refusal")
	}
}
