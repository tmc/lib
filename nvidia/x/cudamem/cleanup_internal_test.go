package cudamem

import (
	"errors"
	"github.com/tmc/lib/nvidia/driver"
	"sync"
	"testing"
)

func TestCleanupInUseAndConcurrent(t *testing.T) {
	tracker := &Tracker{}
	tracker.allocated(16)
	a := &allocation{ptr: 1, bytes: 16, tracker: tracker, users: 1}
	run := func(fn func() error) error { return fn() }
	calls := 0 // Serialized by allocation.mu.
	free := func(driver.DevicePtr) error { calls++; return nil }
	if err := a.free(run, free); !errors.Is(err, ErrInUse) {
		t.Fatalf("free in-use allocation: %v", err)
	}
	if calls != 0 || a.ptr != 1 {
		t.Fatal("in-use allocation was freed")
	}
	a.users = 0
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := a.free(run, free); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if calls != 1 {
		t.Fatalf("native free calls = %d, want 1", calls)
	}
	if current, _ := tracker.Snapshot(); current != 0 {
		t.Fatalf("live bytes = %d, want 0", current)
	}
}

func TestCleanupContextFailure(t *testing.T) {
	failure := errors.New("injected failure")
	for _, stage := range []string{"entry", "destroy", "restore"} {
		t.Run(stage, func(t *testing.T) {
			tracker := &Tracker{}
			tracker.allocated(16)
			owner := &allocation{ptr: 1, bytes: 16, tracker: tracker}
			calls := 0
			run := func(fn func() error) error {
				if stage == "entry" {
					return failure
				}
				if err := fn(); err != nil {
					return err
				}
				return failure
			}
			destroy := func(handle driver.DevicePtr) error {
				if handle != 1 {
					t.Fatalf("handle = %v", handle)
				}
				calls++
				if stage == "destroy" {
					return failure
				}
				return nil
			}
			err := owner.free(run, destroy)
			if !errors.Is(err, failure) {
				t.Fatalf("error = %v", err)
			}
			wantLive := stage != "restore"
			if (owner.ptr != 0) != wantLive {
				t.Fatalf("incorrect retained handle: %+v", owner)
			}
			current, _ := tracker.Snapshot()
			wantBytes := uint64(0)
			if wantLive {
				wantBytes = 16
			}
			if current != wantBytes {
				t.Fatalf("live bytes = %d, want %d", current, wantBytes)
			}
			wantCalls := 1
			if stage == "entry" {
				wantCalls = 0
			}
			if calls != wantCalls {
				t.Fatalf("calls = %d, want %d", calls, wantCalls)
			}
			if wantLive {
				wantCalls++
			}
			run = func(fn func() error) error { return fn() }
			destroy = func(driver.DevicePtr) error { calls++; return nil }
			for range 2 {
				err = owner.free(run, destroy)
				if err != nil {
					t.Fatal(err)
				}
			}
			if calls != wantCalls || owner.ptr != 0 {
				t.Fatalf("after retries: calls=%d want=%d", calls, wantCalls)
			}
			if current, _ := tracker.Snapshot(); current != 0 {
				t.Fatalf("remaining bytes = %d", current)
			}
		})
	}
}
