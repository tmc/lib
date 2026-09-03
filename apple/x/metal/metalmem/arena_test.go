package metalmem_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestArenaEmptyAllocations exercises the paths that need no GPU: an empty
// allocation is free, and the arena stays usable and reports no error.
func TestArenaEmptyAllocations(t *testing.T) {
	a := metalmem.NewArena(nil)
	empty := metalmem.AllocIn[float32](a, "empty", 0)
	uploaded := metalmem.UploadIn[float32](a, "nothing", nil)

	if err := a.Err(); err != nil {
		t.Fatalf("Err after empty allocations = %v, want nil", err)
	}
	for _, s := range []metalmem.Slice[float32]{empty, uploaded} {
		if s.Len() != 0 {
			t.Errorf("Len = %d, want 0", s.Len())
		}
		if s.Buffer() != nil {
			t.Error("Buffer is non-nil; an empty slice must bind a null buffer")
		}
	}
	a.Free()
	a.Free() // freeing twice must be safe
}

// TestArenaStopsAfterFirstError checks that a failed allocation short-circuits
// the rest, so callers can defer error handling to one Err call.
func TestArenaStopsAfterFirstError(t *testing.T) {
	a := metalmem.NewArena(nil) // a nil device fails every non-empty allocation
	first := metalmem.AllocIn[float32](a, "first", 16)
	second := metalmem.AllocIn[int32](a, "second", 8)

	err := a.Err()
	if err == nil {
		t.Fatal("Err = nil, want an error naming the first failed allocation")
	}
	if !strings.Contains(err.Error(), "first") {
		t.Errorf("error %q does not name the allocation that failed", err)
	}
	if strings.Contains(err.Error(), "second") {
		t.Errorf("error %q reports a later allocation; the first must win", err)
	}
	if first.Len() != 0 || second.Len() != 0 {
		t.Errorf("failed allocations returned non-zero slices: %d, %d", first.Len(), second.Len())
	}
	a.Free()
}

func TestAllocRejectsNegativeLength(t *testing.T) {
	if _, err := metalmem.Alloc[float32](nil, -1); err == nil {
		t.Fatal("Alloc with a negative length succeeded, want error")
	}
}

func TestZeroSliceIsFreeAndNull(t *testing.T) {
	var s metalmem.Slice[float32]
	if s.Buffer() != nil {
		t.Error("the zero Slice has a buffer")
	}
	if s.HostSlice() != nil {
		t.Error("the zero Slice has a host slice")
	}
	if got := s.TotalBytes(); got != 0 {
		t.Errorf("TotalBytes = %d, want 0", got)
	}
	if err := s.CopyFromHost(nil); err != nil {
		t.Errorf("CopyFromHost(nil) = %v, want nil", err)
	}
	s.Free() // must not panic
}
