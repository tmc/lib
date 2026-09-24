package cudamem_test

import (
	"context"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// TestArenaResetCycle runs step-shaped cycles on a real chunked arena and
// requires every step after the first to get the first step's pointers
// without reserving more device memory.
func TestArenaResetCycle(t *testing.T) {
	dev := cudatest.Require(t)
	a := cudamem.NewChunkedArena(dev, 1<<20)
	defer a.Free()
	stepOnce := func() []driver.DevicePtr {
		if err := a.Reset(); err != nil {
			t.Fatal(err)
		}
		var ptrs []driver.DevicePtr
		for _, n := range []int{1000, 300000, 7} {
			ptrs = append(ptrs, cudamem.AllocIn[float32](a, "activation", n).Ptr())
		}
		for range 4 {
			m := a.Mark()
			ptrs = append(ptrs, cudamem.AllocIn[float32](a, "scratch", 200000).Ptr())
			if err := a.Release(m); err != nil {
				t.Fatal(err)
			}
		}
		if err := a.Err(); err != nil {
			t.Fatal(err)
		}
		return ptrs
	}
	first := stepOnce()
	reserved := a.Stats().Reserved
	for i := range 3 {
		if got := stepOnce(); !slices.Equal(got, first) {
			t.Fatalf("step %d pointers %x, want %x", i+2, got, first)
		}
	}
	if st := a.Stats(); st.Reserved != reserved {
		t.Fatalf("reserved grew from %d to %d", reserved, st.Reserved)
	}
}

func TestMemsetAsync(t *testing.T) {
	dev := cudatest.Require(t)
	stream := dev.DefaultStream()
	for _, n := range []int{1, 3, 4, 1001, 4096} {
		s, err := cudamem.Alloc[byte](dev, n+2)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Free()
		if err := s.CopyFromHost(slices.Repeat([]byte{0xee}, n+2)); err != nil {
			t.Fatal(err)
		}
		// Offset by one byte so the unaligned path runs too.
		for _, off := range []int{0, 1} {
			if err := s.Slice(off, off+n).MemsetAsync(byte(0x40+off), stream); err != nil {
				t.Fatal(err)
			}
			if err := stream.Synchronize(context.Background()); err != nil {
				t.Fatal(err)
			}
			got := make([]byte, n+2)
			if err := s.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			for i, b := range got {
				want := byte(0xee)
				switch {
				case off == 0 && i < n:
					want = 0x40
				case off == 1 && i >= 1 && i <= n:
					want = 0x41
				case off == 1 && i == 0:
					want = 0x40
				}
				if b != want {
					t.Fatalf("n=%d off=%d: byte %d = %#x, want %#x", n, off, i, b, want)
				}
			}
		}
	}
	var empty cudamem.Slice[float32]
	if err := empty.MemsetAsync(0, stream); err != nil {
		t.Errorf("MemsetAsync on an empty slice: %v", err)
	}
}
