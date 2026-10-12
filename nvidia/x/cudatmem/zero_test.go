package cudatmem

import (
	"errors"
	"testing"
)

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
	other, _ := a.AcquirePermit(32)
	x, err := a.Allocate(p, 32, 512, 128)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatal(err)
	}
	x, err = a.Allocate(p, 32, 512, 128)
	if err != nil {
		t.Fatalf("reuse full freed pool: %v", err)
	}
	if err := a.Relinquish(p); err != nil {
		t.Fatalf("relinquish with live allocation: %v", err)
	}
	if _, err := a.Allocate(p, 32, 32, 1); !errors.Is(err, ErrPermission) {
		t.Fatalf("allocate after relinquish: %v", err)
	}
	if err := a.Release(other, x, 32); !errors.Is(err, ErrPermission) {
		t.Fatalf("foreign release: %v", err)
	}
	if err := a.Close(); !errors.Is(err, ErrBusy) {
		t.Fatalf("close with live allocation: %v", err)
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatalf("release after relinquish: %v", err)
	}
	if err := a.Relinquish(other); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}
