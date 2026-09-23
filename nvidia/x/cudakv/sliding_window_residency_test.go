// SPDX-License-Identifier: Apache-2.0

package cudakv

import "testing"

// TestRollingKVCacheResidency pins down which logical positions a slot actually
// holds. PhysicalSlot alone is many-to-one: with a window of 8, positions 4 and
// 12 both map to slot 4, so the mapping cannot tell a caller which one is there.
func TestRollingKVCacheResidency(t *testing.T) {
	cache, err := NewRollingKVCache(SlidingWindowConfig{SinkTokens: 4, WindowSize: 8})
	if err != nil {
		t.Fatalf("NewRollingKVCache failed: %v", err)
	}

	if cache.PhysicalSlot(4) != cache.PhysicalSlot(12) {
		t.Fatal("positions 4 and 12 should collide in an 8-slot window")
	}

	// Nothing is resident before anything is written.
	for _, pos := range []int{0, 4, 12} {
		if cache.Resident(pos) {
			t.Errorf("Resident(%d) = true on an empty cache", pos)
		}
	}

	// After 12 tokens the whole capacity is live and nothing has been lapped.
	if err := cache.Advance(12); err != nil {
		t.Fatal(err)
	}
	if got := cache.WindowStart(); got != 4 {
		t.Errorf("WindowStart = %d, want 4", got)
	}
	for pos := 0; pos < 12; pos++ {
		if !cache.Resident(pos) {
			t.Errorf("Resident(%d) = false, want true", pos)
		}
	}
	if cache.Resident(12) {
		t.Error("Resident(12) = true before position 12 was written")
	}

	// Writing position 12 overwrites position 4's slot. The sink is untouched.
	if err := cache.Advance(1); err != nil {
		t.Fatal(err)
	}
	if got := cache.WindowStart(); got != 5 {
		t.Errorf("WindowStart = %d, want 5", got)
	}
	if cache.Resident(4) {
		t.Error("Resident(4) = true after position 12 lapped its slot")
	}
	if _, ok := cache.Slot(4); ok {
		t.Error("Slot(4) reported a usable slot after it was overwritten")
	}
	for pos := 0; pos < 4; pos++ {
		if !cache.Resident(pos) {
			t.Errorf("sink position %d was evicted", pos)
		}
	}
	for pos := 5; pos <= 12; pos++ {
		slot, ok := cache.Slot(pos)
		if !ok {
			t.Errorf("Resident(%d) = false, want true", pos)
			continue
		}
		if want := cache.PhysicalSlot(pos); slot != want {
			t.Errorf("Slot(%d) = %d, want %d", pos, slot, want)
		}
	}

	// Exactly capacity positions are resident once the stream is long.
	if err := cache.Advance(1000); err != nil {
		t.Fatal(err)
	}
	live := 0
	for pos := 0; pos < cache.TotalSeen(); pos++ {
		if cache.Resident(pos) {
			live++
		}
	}
	if want := 4 + 8; live != want {
		t.Errorf("%d resident positions, want %d", live, want)
	}
	if live != cache.ActiveTokens() {
		t.Errorf("resident count %d disagrees with ActiveTokens %d", live, cache.ActiveTokens())
	}

	// Every resident position occupies a distinct slot in range.
	seen := make(map[int]int)
	for pos := 0; pos < cache.TotalSeen(); pos++ {
		slot, ok := cache.Slot(pos)
		if !ok {
			continue
		}
		if slot < 0 || slot >= 12 {
			t.Errorf("Slot(%d) = %d, out of range [0,12)", pos, slot)
		}
		if prev, dup := seen[slot]; dup {
			t.Errorf("positions %d and %d both claim slot %d", prev, pos, slot)
		}
		seen[slot] = pos
	}
}

func TestRollingKVCacheNegativePosition(t *testing.T) {
	cache, err := NewRollingKVCache(SlidingWindowConfig{SinkTokens: 4, WindowSize: 8})
	if err != nil {
		t.Fatalf("NewRollingKVCache failed: %v", err)
	}
	if err := cache.Advance(100); err != nil {
		t.Fatal(err)
	}

	if got := cache.PhysicalSlot(-1); got != -1 {
		t.Errorf("PhysicalSlot(-1) = %d, want -1", got)
	}
	if cache.Resident(-1) {
		t.Error("Resident(-1) = true")
	}
}

func TestRollingKVCacheNoSink(t *testing.T) {
	cache, err := NewRollingKVCache(SlidingWindowConfig{SinkTokens: 0, WindowSize: 4})
	if err != nil {
		t.Fatalf("NewRollingKVCache failed: %v", err)
	}
	if err := cache.Advance(10); err != nil {
		t.Fatal(err)
	}

	if got := cache.WindowStart(); got != 6 {
		t.Errorf("WindowStart = %d, want 6", got)
	}
	for pos := 0; pos < 6; pos++ {
		if cache.Resident(pos) {
			t.Errorf("Resident(%d) = true, want false", pos)
		}
	}
	for pos := 6; pos < 10; pos++ {
		if !cache.Resident(pos) {
			t.Errorf("Resident(%d) = false, want true", pos)
		}
	}
}
