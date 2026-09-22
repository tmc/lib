package cudamem

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestAllocationErrorOwnership(t *testing.T) {
	failure := errors.New("injected failure")
	for _, stage := range []string{"entry", "allocate", "partial", "restore", "success"} {
		t.Run(stage, func(t *testing.T) {
			dev, tracker := &cuda.Device{}, &Tracker{}
			calls := 0
			slice, err := allocateSlice[uint32](dev, tracker, 8, func(fn func() error) error {
				if stage == "entry" {
					return failure
				}
				if err := fn(); err != nil {
					return err
				}
				if stage == "restore" {
					return failure
				}
				return nil
			}, func(bytes uint) (driver.DevicePtr, error) {
				calls++
				if bytes != 32 {
					t.Fatalf("bytes=%d", bytes)
				}
				if stage == "allocate" {
					return 0, failure
				}
				if stage == "partial" {
					return 0x1000, failure
				}
				return 0x1000, nil
			})
			if errors.Is(err, failure) != (stage != "success") {
				t.Fatalf("error=%v", err)
			}
			if (calls == 0) != (stage == "entry") {
				t.Fatalf("calls=%d", calls)
			}
			owned := stage == "partial" || stage == "restore" || stage == "success"
			current, peak := tracker.Snapshot()
			if !owned {
				if slice.Ptr() != 0 || slice.Len() != 0 || current != 0 || peak != 0 {
					t.Fatal("unallocated memory reported owned")
				}
				return
			}
			if slice.Ptr() != 0x1000 || slice.Len() != 8 || slice.Cap() != 8 || slice.alloc.device != dev || current != 32 || peak != 32 {
				t.Fatal("lost allocation identity/accounting")
			}
			copy := slice.Slice(2, 4)
			freed := 0
			for _, a := range []*allocation{copy.alloc, slice.alloc} {
				if err := a.free(func(fn func() error) error { return fn() }, func(ptr driver.DevicePtr) error {
					if ptr != 0x1000 {
						t.Fatalf("free ptr=%v", ptr)
					}
					freed++
					return nil
				}); err != nil {
					t.Fatal(err)
				}
			}
			current, peak = tracker.Snapshot()
			if freed != 1 || current != 0 || peak != 32 || slice.Ptr() != 0 || copy.Ptr() != 0 {
				t.Fatal("cleanup ownership/accounting not shared")
			}
		})
	}
}

func TestAllocationExtentChecks(t *testing.T) {
	for _, length := range []int{-1, int(^uint(0) >> 1)} {
		_, err := allocateSlice[uint64](nil, nil, length, func(func() error) error {
			t.Fatal("invalid extent entered CUDA context")
			return nil
		}, nil)
		if err == nil {
			t.Fatalf("accepted length %d", length)
		}
	}
	if _, err := Alloc[byte](nil, 1); err == nil {
		t.Fatal("nil allocation device accepted")
	}
	if _, err := AllocManaged[byte](nil, 1); err == nil {
		t.Fatal("nil managed device accepted")
	}
}
