package mempool

import (
	"testing"

	"github.com/tmc/lib/apple/x/metal"
)

func TestAllocUnified(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device not available: %v", err)
	}
	defer dev.Close()

	slice, err := Alloc[float32](dev, 16)
	if err != nil {
		t.Fatalf("alloc failed: %v", err)
	}
	defer slice.Free()

	if slice.Len() != 16 {
		t.Fatalf("expected length 16, got %d", slice.Len())
	}
	host := slice.HostSlice()
	if len(host) != 16 {
		t.Fatalf("expected host slice length 16, got %d", len(host))
	}
	host[0] = 3.14
	if host[0] != 3.14 {
		t.Fatalf("expected 3.14, got %f", host[0])
	}
}
