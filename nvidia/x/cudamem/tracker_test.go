package cudamem

import (
	"sync"
	"testing"
)

func TestTrackerInterval(t *testing.T) {
	var tracker Tracker
	tracker.allocated(10)
	if got := tracker.Reset(); got != 10 {
		t.Fatalf("Reset() = %d, want 10", got)
	}
	tracker.allocated(7)
	tracker.freed(7)
	current, peak := tracker.Snapshot()
	if current != 10 || peak != 17 {
		t.Fatalf("Snapshot() = %d, %d, want 10, 17", current, peak)
	}
	tracker.Reset()
	_, peak = tracker.Snapshot()
	if peak != 10 {
		t.Fatalf("peak after Reset = %d, want 10", peak)
	}
}

func TestTrackerConcurrent(t *testing.T) {
	var tracker Tracker
	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tracker.allocated(4)
			tracker.freed(4)
		}()
	}
	wg.Wait()
	current, peak := tracker.Snapshot()
	if current != 0 || peak == 0 {
		t.Fatalf("Snapshot() = %d, %d, want zero current and nonzero peak", current, peak)
	}
}

func TestAllocTrackedFailureDoesNotCount(t *testing.T) {
	var tracker Tracker
	if _, err := AllocTracked[byte](nil, &tracker, 1); err == nil {
		t.Fatal("AllocTracked(nil) succeeded")
	}
	current, peak := tracker.Snapshot()
	if current != 0 || peak != 0 {
		t.Fatalf("Snapshot() = %d, %d after failure, want 0, 0", current, peak)
	}
}

func TestAllocTrackedRejectsByteSizeOverflow(t *testing.T) {
	length := int(^uint(0) >> 1)
	if _, err := AllocTracked[uint64](nil, new(Tracker), length); err == nil {
		t.Fatal("AllocTracked accepted overflowing byte size")
	}
}
