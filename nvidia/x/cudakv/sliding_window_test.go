// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
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
