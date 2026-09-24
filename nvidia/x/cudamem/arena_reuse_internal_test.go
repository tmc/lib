package cudamem

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

// fakeBlocks is a block allocator that hands out distinct fake pointers and
// counts calls, standing in for the driver.
type fakeBlocks struct {
	calls int
	next  driver.DevicePtr
}

func (f *fakeBlocks) alloc(bytes int) (Slice[byte], error) {
	f.calls++
	f.next += 1 << 32
	return Slice[byte]{ptr: f.next, len: bytes, cap: bytes, elemSize: 1}, nil
}

func newFakeArena(chunk int) (*Arena, *fakeBlocks) {
	f := &fakeBlocks{}
	return &Arena{chunk: chunk, newBlock: f.alloc}, f
}

// step makes a training step's requests: activations that live for the
// step, then per-layer scratch released after each layer. It returns every
// pointer it was given.
func step(t *testing.T, a *Arena, sizes []int, layers int) []driver.DevicePtr {
	t.Helper()
	if err := a.Reset(); err != nil {
		t.Fatal(err)
	}
	var ptrs []driver.DevicePtr
	for _, n := range sizes {
		ptrs = append(ptrs, AllocIn[float32](a, "activation", n).Ptr())
	}
	for l := range layers {
		m := a.Mark()
		for _, n := range sizes {
			ptrs = append(ptrs, AllocIn[float32](a, "scratch", n+l).Ptr())
		}
		if err := a.Release(m); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	return ptrs
}

func TestArenaStepReuse(t *testing.T) {
	a, f := newFakeArena(1 << 16)
	sizes := []int{1000, 3000, 7, 20000} // the last spills into more blocks
	first := step(t, a, sizes, 3)
	blocks := f.calls
	if blocks == 0 {
		t.Fatal("first step allocated no blocks")
	}
	for i := range 5 {
		if got := step(t, a, sizes, 3); !reflect.DeepEqual(got, first) {
			t.Fatalf("step %d pointers differ from the first step", i+2)
		}
	}
	if f.calls != blocks {
		t.Fatalf("later steps allocated %d more blocks", f.calls-blocks)
	}

	// A bigger step allocates only what does not fit, and the original
	// step afterwards still allocates nothing and keeps its pointers
	// wherever the bigger step did not reshape the blocks.
	step(t, a, append(slices.Clone(sizes), 50000), 3)
	grown := f.calls
	if grown == blocks {
		t.Fatal("bigger step allocated nothing")
	}
	step(t, a, sizes, 3)
	again := step(t, a, sizes, 3)
	if f.calls != grown {
		t.Fatalf("repeat of the original step allocated %d blocks", f.calls-grown)
	}
	if got := step(t, a, sizes, 3); !reflect.DeepEqual(got, again) {
		t.Fatal("identical steps after growth got different pointers")
	}
}

func TestArenaMarkRelease(t *testing.T) {
	a, _ := newFakeArena(4096)
	x := AllocIn[byte](a, "x", 100)
	m := a.Mark()
	y := AllocIn[byte](a, "y", 100)
	if err := a.Release(m); err != nil {
		t.Fatal(err)
	}
	if z := AllocIn[byte](a, "z", 50); z.Ptr() != y.Ptr() {
		t.Fatalf("allocation after Release at %#x, want reused %#x", z.Ptr(), y.Ptr())
	}
	if err := a.Reset(); err != nil {
		t.Fatal(err)
	}
	if w := AllocIn[byte](a, "w", 1); w.Ptr() != x.Ptr() {
		t.Fatalf("allocation after Reset at %#x, want %#x", w.Ptr(), x.Ptr())
	}
	late := a.Mark()
	AllocIn[byte](a, "v", 1000)
	if err := a.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := a.Release(late); err == nil {
		t.Fatal("Release accepted a mark past the current allocations")
	}
}

func TestArenaStats(t *testing.T) {
	a, _ := newFakeArena(4096)
	AllocIn[byte](a, "a", 1)     // 256 after alignment
	AllocIn[float32](a, "b", 65) // 260 bytes, 512 aligned
	m := a.Mark()
	AllocIn[byte](a, "c", 1000) // 1024
	want := ArenaStats{InUse: 1792, HighWater: 1792, Reserved: 4096}
	if got := a.Stats(); got != want {
		t.Fatalf("Stats = %+v, want %+v", got, want)
	}
	if err := a.Release(m); err != nil {
		t.Fatal(err)
	}
	AllocIn[byte](a, "d", 1)
	want = ArenaStats{InUse: 1024, HighWater: 1792, Reserved: 4096}
	if got := a.Stats(); got != want {
		t.Fatalf("after Release, Stats = %+v, want %+v", got, want)
	}
	AllocIn[byte](a, "big", 10000) // a block of its own
	if got := a.Stats(); got.Reserved != 4096+10240 || got.HighWater != 1024+10240 {
		t.Fatalf("after a large request, Stats = %+v", got)
	}
}

// TestArenaReleaseUnchunked checks that an unchunked arena frees exactly
// the allocations made after the mark, most recent first, and keeps one
// whose free fails.
func TestArenaReleaseUnchunked(t *testing.T) {
	a := NewArena(nil)
	var freed []int
	failure := errors.New("busy")
	for i := range 4 {
		allocIn(a, "x", func() (Slice[byte], error) {
			return Slice[byte]{ptr: driver.DevicePtr(0x1000 * (i + 1)), len: 1, cap: 1, elemSize: 1}, nil
		})
		a.frees[i] = func() error {
			freed = append(freed, i)
			if i == 2 {
				return failure
			}
			return nil
		}
		a.spans = append(a.spans, arenaSpan{driver.DevicePtr(0x1000 * (i + 1)), 1})
		a.use(1)
		if i == 0 {
			m := a.Mark()
			defer func() {
				if err := a.Release(m); !errors.Is(err, failure) {
					t.Errorf("Release = %v, want %v", err, failure)
				}
				if !reflect.DeepEqual(freed, []int{3, 2, 1}) {
					t.Errorf("freed %v, want [3 2 1]", freed)
				}
				if len(a.frees) != 2 || len(a.spans) != 1 || a.Stats().InUse != 1 {
					t.Errorf("after Release: %d frees, %d spans, %+v", len(a.frees), len(a.spans), a.Stats())
				}
			}()
		}
	}
}
