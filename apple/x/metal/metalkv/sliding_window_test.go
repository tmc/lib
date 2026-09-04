// SPDX-License-Identifier: Apache-2.0

package metalkv

import (
	"math"
	"testing"
)

func TestRollingKVCacheSlotMapping(t *testing.T) {
	// SinkTokens: 4, WindowSize: 8 -> Capacity: 12
	cache, err := NewRollingKVCache(SlidingWindowConfig{
		SinkTokens: 4,
		WindowSize: 8,
	})
	if err != nil {
		t.Fatalf("NewRollingKVCache failed: %v", err)
	}

	// Sink tokens 0..3 stay in slots 0..3
	for i := 0; i < 4; i++ {
		if slot := cache.PhysicalSlot(i); slot != i {
			t.Errorf("sink token %d slot = %d, want %d", i, slot, i)
		}
	}

	// Window tokens 4..11 map to slots 4..11
	for i := 4; i < 12; i++ {
		if slot := cache.PhysicalSlot(i); slot != i {
			t.Errorf("window token %d slot = %d, want %d", i, slot, i)
		}
	}

	// Token 12 rolls into slot 4 (first window slot)
	if slot := cache.PhysicalSlot(12); slot != 4 {
		t.Errorf("rolled token 12 slot = %d, want 4", slot)
	}

	// Token 13 rolls into slot 5
	if slot := cache.PhysicalSlot(13); slot != 5 {
		t.Errorf("rolled token 13 slot = %d, want 5", slot)
	}

	if err := cache.Advance(20); err != nil {
		t.Fatal(err)
	}
	if cache.TotalSeen() != 20 {
		t.Errorf("TotalSeen = %d, want 20", cache.TotalSeen())
	}
	if cache.ActiveTokens() != 12 {
		t.Errorf("ActiveTokens = %d, want 12", cache.ActiveTokens())
	}
}

func TestRollingKVCacheRejectsNegativeAdvance(t *testing.T) {
	cache, err := NewRollingKVCache(SlidingWindowConfig{SinkTokens: 1, WindowSize: 4})
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Advance(-1); err == nil {
		t.Fatal("Advance(-1) succeeded")
	}
	if got := cache.TotalSeen(); got != 0 {
		t.Fatalf("TotalSeen after rejected advance = %d, want 0", got)
	}
	cache.totalSeen = math.MaxInt
	if err := cache.Advance(1); err == nil {
		t.Fatal("Advance past MaxInt succeeded")
	}
	if got := cache.TotalSeen(); got != math.MaxInt {
		t.Fatalf("TotalSeen after rejected overflow = %d, want %d", got, math.MaxInt)
	}
}
