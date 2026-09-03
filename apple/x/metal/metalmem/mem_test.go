package metalmem_test

import (
	"testing"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestAllocAndCopies(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}

	slice, err := metalmem.Alloc[float32](dev, 64)
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}

	if slice.Len() != 64 || slice.Cap() != 64 {
		t.Errorf("unexpected len/cap: %d/%d", slice.Len(), slice.Cap())
	}

	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i) * 1.5
	}

	if err := slice.CopyFromHost(src); err != nil {
		t.Fatalf("CopyFromHost failed: %v", err)
	}

	dst := make([]float32, 64)
	if err := slice.CopyToHost(dst); err != nil {
		t.Fatalf("CopyToHost failed: %v", err)
	}

	for i := range src {
		if src[i] != dst[i] {
			t.Fatalf("mismatch at %d: want %f, got %f", i, src[i], dst[i])
		}
	}
}

func TestReleasedSliceFailsClosed(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	s, err := metalmem.Alloc[uint32](dev, 4)
	if err != nil {
		t.Fatal(err)
	}
	sub, err := s.Slice(1, 3)
	if err != nil {
		t.Fatal(err)
	}
	s.Free()
	sub.Free()
	if s.Buffer() != nil || s.HostSlice() != nil || sub.HostSlice() != nil {
		t.Fatal("released allocation remains accessible")
	}
	if _, err := sub.Slice(0, 1); err == nil {
		t.Fatal("slice of released allocation succeeded")
	}
	if err := sub.CopyFromHost([]uint32{1}); err == nil {
		t.Fatal("copy to released allocation succeeded")
	}
}

func TestCopyToHostChecksEmptyDestination(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	s, err := metalmem.Alloc[uint32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()
	if err := s.CopyToHost(nil); err == nil {
		t.Fatal("non-empty source copied to empty destination")
	}
}
