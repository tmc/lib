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
