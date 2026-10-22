package dsv4expertcache

import (
	"fmt"
	"math"
	"testing"
)

func TestMetalHotness(t *testing.T) {
	c := cache(t)
	h := new(MetalHotness)
	commit := func(keys []Key) *Lease {
		t.Helper()
		l, err := h.Acquire(c, keys)
		if err != nil {
			t.Fatal(err)
		}
		for _, u := range l.Uploads() {
			if err := l.Started(u, &completion{}); err != nil {
				t.Fatal(err)
			}
		}
		if err := l.Commit(); err != nil {
			t.Fatal(err)
		}
		return l
	}
	l := commit([]Key{key(0), key(1), key(0), key(0)})
	if h.routes[0][0] != 3 || h.routes[0][1] != 1 {
		t.Fatal("route counts")
	}
	if err := l.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	// Despite more recent use, colder expert 1 is the victim.
	l = commit([]Key{key(2)})
	if len(l.Uploads()) != 1 || l.Uploads()[0].Slot != 1 {
		t.Fatal("cold victim", l.Uploads())
	}
	if err := l.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	for range 17 {
		if err := h.NoteToken(0); err != nil {
			t.Fatal(err)
		}
		if err := h.NoteToken(1); err != nil {
			t.Fatal(err)
		}
	}
	if h.tokens != 17 || h.routes[0][0] != 1 {
		t.Fatal("token aging", h.tokens, h.routes[0][0])
	}
	h.routes[0][0] = math.MaxUint32
	l = commit([]Key{key(0)})
	if h.routes[0][0] != math.MaxUint32 {
		t.Fatal("hotness overflow")
	}
	if _, err := h.Acquire(c, []Key{key(1)}); err == nil {
		t.Fatal("active consumer lost")
	}
	if err := l.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
}

func TestMetalPinCapacity(t *testing.T) {
	for _, tt := range []struct {
		entries, locked uint32
		bytes, expert   uint64
		want            uint32
	}{
		{100, 0, 0, 1, 100}, {100, 70, 0, 1, 70},
		{100, 70, 3 << 30, 1 << 29, 4},
		{100, 70, 1 << 30, 1 << 29, 2},
		{100, 70, 3 << 30, 0, 70}, {0, 0, 0, 1, 0},
	} {
		if got := MetalPinCapacity(tt.entries, tt.locked, tt.bytes, tt.expert); got != tt.want {
			t.Fatal(got, tt.want)
		}
	}
}

func ExampleMetalPinCapacity() {
	fmt.Println(MetalPinCapacity(100, 70, 3<<30, 1<<29))
	// Output: 4
}
