// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"math"
	"testing"
)

func TestRollingKVCacheSlotMapping(t *testing.T) {
	cache, err := NewRollingKVCache(SlidingWindowConfig{
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

func TestRollingKVCacheAdvanceRejectsInvalid(t *testing.T) {
	cache, err := NewRollingKVCache(SlidingWindowConfig{SinkTokens: 4, WindowSize: 8})
	if err != nil {
		t.Fatalf("NewRollingKVCache failed: %v", err)
	}
	if err := cache.Advance(-1); err == nil {
		t.Error("Advance(-1) = nil, want error")
	}
	if got := cache.TotalSeen(); got != 0 {
		t.Errorf("TotalSeen = %d after rejected advance, want 0", got)
	}
	if err := cache.Advance(math.MaxInt); err != nil {
		t.Fatal(err)
	}
	if err := cache.Advance(1); err == nil {
		t.Error("overflowing Advance = nil, want error")
	}
	if got := cache.TotalSeen(); got != math.MaxInt {
		t.Errorf("TotalSeen = %d after rejected advance, want %d", got, math.MaxInt)
	}
}
