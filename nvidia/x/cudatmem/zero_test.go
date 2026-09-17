package cudatmem

import "testing"

func TestZeroAllocatorRejected(t *testing.T) {
	var a Allocator
	if _, err := a.AcquirePermit(32); err == nil {
		t.Fatal("accepted uninitialized allocator")
	}
	if _, err := a.Allocate(&Permit{}, 32, 32, 1); err == nil {
		t.Fatal("accepted zero allocator allocation")
	}
}

func TestPermitHeldUntilReleased(t *testing.T) {
	a, _ := New(SM100A)
	p, _ := a.AcquirePermit(32)
	x, err := a.Allocate(p, 32, 512, 128)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Relinquish(p); err == nil {
		t.Fatal("abandoned live allocation")
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatal(err)
	}
	x, err = a.Allocate(p, 32, 512, 128)
	if err != nil {
		t.Fatalf("reuse full freed pool: %v", err)
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatal(err)
	}
	if err := a.Relinquish(p); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}
