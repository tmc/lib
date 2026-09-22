package cudamem

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestAsyncAllocationOwnership(t *testing.T) {
	failure := errors.New("injected")
	for _, stage := range []string{"entry", "allocate", "partial", "restore", "success"} {
		t.Run(stage, func(t *testing.T) {
			dev := &cuda.Device{}
			calls := 0
			slice, err := allocateAsyncSlice[uint32](dev, 65, func(fn func(driver.Stream) error) error {
				if stage == "entry" {
					return failure
				}
				if err := fn(7); err != nil {
					return err
				}
				if stage == "restore" {
					return failure
				}
				return nil
			}, func(stream driver.Stream, bytes uint) (driver.DevicePtr, error) {
				calls++
				if stream != 7 || bytes != 512 {
					t.Fatalf("stream=%v bytes=%d", stream, bytes)
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
				t.Fatal("unexpected allocator call")
			}
			owned := stage != "entry" && stage != "allocate"
			if (slice.Ptr() != 0) != owned {
				t.Fatal("lost allocation owner")
			}
			if owned && (slice.Len() != 65 || slice.Cap() != 128 || slice.alloc.bytes != 512 || slice.alloc.device != dev) {
				t.Fatal("incorrect bucket ownership metadata")
			}
		})
	}
}

func TestAsyncAllocationExtentChecks(t *testing.T) {
	run := func(func(driver.Stream) error) error { t.Fatal("invalid extent entered stream"); return nil }
	if _, err := allocateAsyncSlice[uint64](nil, math.MaxInt, run, nil); err == nil {
		t.Fatal("multiplication overflow accepted")
	}
	if _, err := allocateAsyncSlice[byte](nil, math.MaxInt, run, nil); err == nil {
		t.Fatal("bucket capacity overflow accepted")
	}
	if _, err := allocateAsyncSlice[byte](nil, -1, run, nil); err == nil {
		t.Fatal("negative extent accepted")
	}
}
