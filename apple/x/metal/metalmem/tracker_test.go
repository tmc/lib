package metalmem

import (
	"sync"
	"testing"

	"github.com/tmc/lib/apple/x/metal"
)

func TestTracker(t *testing.T) {
	var tracker Tracker
	a := tracker.acquire(10)
	b := a
	tracker.acquire(0).release()
	if current, peak := tracker.Snapshot(); current != 10 || peak != 10 {
		t.Fatalf("Snapshot = %d, %d, want 10, 10", current, peak)
	}
	tracker.ResetPeak()
	c := tracker.acquire(7)
	b.release()
	a.release()
	c.release()
	if current, peak := tracker.Snapshot(); current != 0 || peak != 17 {
		t.Fatalf("Snapshot after release = %d, %d, want 0, 17", current, peak)
	}
}

func TestTrackedArenaPhysicalAllocation(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	defer dev.Close()
	var tracker Tracker
	a := NewTrackedArena(dev, &tracker)
	s := AllocIn[float32](a, "physical", 32)
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	alias := s
	if current, peak := tracker.Snapshot(); current != 128 || peak != 128 {
		t.Fatalf("Snapshot = %d, %d, want 128, 128", current, peak)
	}
	alias.Free()
	s.Free()
	a.Free()
	if current, peak := tracker.Snapshot(); current != 0 || peak != 128 {
		t.Fatalf("Snapshot after release = %d, %d, want 0, 128", current, peak)
	}
}

func TestTrackerConcurrent(t *testing.T) {
	var tracker Tracker
	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tracker.acquire(8).release()
		}()
	}
	wg.Wait()
	current, peak := tracker.Snapshot()
	if current != 0 || peak == 0 {
		t.Fatalf("Snapshot = %d, %d, want zero current and nonzero peak", current, peak)
	}
}
