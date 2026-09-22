package cudamem

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestAsyncFreeContextOwnership(t *testing.T) {
	failure := errors.New("injected failure")
	for _, stage := range []string{"entry", "enqueue", "restore", "success"} {
		t.Run(stage, func(t *testing.T) {
			tracker := &Tracker{}
			tracker.allocated(32)
			a := &allocation{ptr: 0x1000, bytes: 32, tracker: tracker}
			calls := 0
			inside := false
			var callbacks []func()
			register := func(fn func()) {
				if inside {
					t.Fatal("cleanup registered inside stream lifetime lock")
				}
				callbacks = append(callbacks, fn)
			}
			err := a.freeAsync(func(fn func(driver.Stream) error) error {
				if stage == "entry" {
					return failure
				}
				inside = true
				err := fn(7)
				inside = false
				if err != nil {
					return err
				}
				if stage == "restore" {
					return failure
				}
				return nil
			}, func(ptr driver.DevicePtr, stream driver.Stream) error {
				if ptr != 0x1000 || stream != 7 {
					t.Fatal("wrong native arguments")
				}
				calls++
				if stage == "enqueue" {
					return failure
				}
				return nil
			}, register)
			if errors.Is(err, failure) != (stage != "success") {
				t.Fatalf("error=%v", err)
			}
			queued := stage == "restore" || stage == "success"
			if (a.ptr == 0) != queued || (len(callbacks) == 1) != queued {
				t.Fatal("enqueue ownership lost")
			}
			if current, _ := tracker.Snapshot(); current != 32 {
				t.Fatal("freed before completion")
			}
			wantCalls := 1
			if stage == "enqueue" {
				wantCalls = 2
			}
			for range 2 {
				if err := a.freeAsync(func(fn func(driver.Stream) error) error { return fn(7) }, func(driver.DevicePtr, driver.Stream) error { calls++; return nil }, register); err != nil {
					t.Fatal(err)
				}
			}
			if calls != wantCalls || len(callbacks) != 1 {
				t.Fatalf("calls=%d callbacks=%d", calls, len(callbacks))
			}
			callbacks[0]()
			if current, peak := tracker.Snapshot(); current != 0 || peak != 32 {
				t.Fatalf("accounting=%d/%d", current, peak)
			}
		})
	}
}
