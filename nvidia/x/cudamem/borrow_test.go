package cudamem

import (
	"errors"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"testing"
)

func TestBorrowActualOwnerAndAliases(t *testing.T) {
	a, b := &cuda.Device{}, &cuda.Device{}
	s := Slice[float32]{alloc: &allocation{ptr: 4096, device: a}, ptr: 4096, len: 4, cap: 4, elemSize: 4}
	if s.Device() != a {
		t.Fatal("lost actual device")
	}
	if _, err := s.Borrow(b); err == nil {
		t.Fatal("cross-device borrow accepted")
	}
	release, err := s.Borrow(a)
	if err != nil {
		t.Fatal(err)
	}
	alias := s.Slice(1, 3)
	if err := alias.alloc.free(func(fn func() error) error { return fn() }, func(_ driver.DevicePtr) error { return nil }); !errors.Is(err, ErrInUse) {
		t.Fatal("borrowed alias freed")
	}
	release()
	release()
	if s.alloc.users != 0 {
		t.Fatal("release not once")
	}
	if err := alias.alloc.free(func(fn func() error) error { return fn() }, func(_ driver.DevicePtr) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if s.Device() != nil {
		t.Fatal("freed identity advertised")
	}
	if _, err := alias.Borrow(a); err == nil || errors.Is(err, ErrInUse) {
		t.Fatal("freed alias borrowed")
	}
}
