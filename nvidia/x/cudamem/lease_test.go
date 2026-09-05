package cudamem

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestTransferLeaseProtectsAliases(t *testing.T) {
	// Synthetic pointers never reach CUDA: Free must reject outstanding leases.
	a := &allocation{ptr: 1}
	s := Slice[byte]{ptr: 1, len: 2, cap: 2, elemSize: 1, alloc: a}
	release1, err := s.retain(nil)
	if err != nil {
		t.Fatal(err)
	}
	alias := s.Slice(1, 2)
	release2, err := alias.retain(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := alias.Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("Free = %v", err)
	}
	release1()
	if err := s.Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("Free with remaining lease = %v", err)
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(release2)
	}
	wg.Wait()
	if a.users != 0 {
		t.Fatalf("leases = %d", a.users)
	}
	// Mark freed rather than invoking CUDA on the synthetic address.
	a.ptr = 0
	if err := alias.Free(); err != nil {
		t.Fatal(err)
	}
}

func TestTransferLeaseValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		slice  Slice[byte]
		device *cuda.Device
	}{
		{"unowned", Slice[byte]{ptr: 1}, nil},
		{"freed", Slice[byte]{alloc: &allocation{}}, nil},
		{"different device", Slice[byte]{alloc: &allocation{ptr: 1}}, new(cuda.Device)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if release, err := tt.slice.retain(tt.device); err == nil || release != nil {
				t.Fatalf("release=%v error=%v", release != nil, err)
			}
		})
	}
}
