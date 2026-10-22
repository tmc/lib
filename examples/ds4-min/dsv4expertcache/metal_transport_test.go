package dsv4expertcache

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestMetalPhysicalBudget(t *testing.T) {
	for _, budget := range []uint64{12, 32767} {
		b := &fakeMetalUpload{pageBytes: 16384}
		if u, err := newMetalUpload(b, 2, [3]int{2, 2, 2}, budget); err == nil || u != nil || b.buf != nil || b.planes != nil {
			t.Fatalf("budget %d allocated storage before admission", budget)
		}
	}
	for _, tt := range []struct {
		name                              string
		slots                             uint64
		strides                           [3]uint64
		page, budget, slot, shared, stage uint64
		fail                              bool
	}{
		{"logical-only", 2, [3]uint64{2, 2, 2}, 16384, 12, 0, 0, 0, true},
		{"one-byte-short", 2, [3]uint64{2, 2, 2}, 16384, 32767, 0, 0, 0, true},
		{"exact", 2, [3]uint64{2, 2, 2}, 16384, 32768, 16384, 32768, 16384, false},
		{"whole-page", 1, [3]uint64{1000, 1000, 2096}, 4096, 4096, 4096, 4096, 4096, false},
		{"page-crossing", 1, [3]uint64{1000, 1000, 2097}, 4096, 8192, 8192, 8192, 4096, false},
		{"triplet-overflow", 1, [3]uint64{math.MaxUint64, 1, 1}, 4096, math.MaxUint64, 0, 0, 0, true},
		{"rounding-overflow", 1, [3]uint64{math.MaxUint64 - 2, 1, 1}, 4096, math.MaxUint64, 0, 0, 0, true},
		{"slab-overflow", math.MaxUint64, [3]uint64{1, 1, 1}, 4096, math.MaxUint64, 0, 0, 0, true},
		{"reserve-overflow", math.MaxUint64 / 4096, [3]uint64{1, 1, 1}, 4096, math.MaxUint64, 0, 0, 0, true},
		{"zero-page", 1, [3]uint64{1, 1, 1}, 0, 100, 0, 0, 0, true},
		{"nonpower-page", 1, [3]uint64{1, 1, 1}, 3, 100, 0, 0, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			slot, shared, stage, err := metalAllocationBytes(tt.slots, tt.strides, tt.page, tt.budget)
			if (err != nil) != tt.fail || slot != tt.slot || shared != tt.shared || stage != tt.stage {
				t.Fatalf("got %d/%d/%d err=%v", slot, shared, stage, err)
			}
		})
	}
}

type fakeMetalUpload struct {
	pageBytes            uint64
	buf                  []byte
	planes               [][3][]byte
	writes               int
	failWrite, failClose bool
	closed               bool
	pinFailure           *MetalPinFailure
}

func TestMetalSelectRoutes(t *testing.T) {
	b := new(fakeMetalUpload)
	u, err := newMetalUpload(b, 1, [3]int{2, 2, 2}, 6)
	if err != nil {
		t.Fatal(err)
	}
	c, err := New(key(0).Identity, 1, 6, 6)
	if err != nil {
		t.Fatal(err)
	}
	s, _, err := u.SelectRoutes(source(t), c, 0, []int{1})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	s, ids, err := u.SelectRoutes(source(t), c, 0, []int{0, 0, 1})
	if err != nil || !reflect.DeepEqual(ids, []int{1}) || b.writes != 3 {
		t.Fatalf("resident protection: ids=%v writes=%d err=%v", ids, b.writes, err)
	}
	if u.hotness.routes[0][0] != 2 || u.hotness.routes[0][1] != 2 {
		t.Fatal("overflow routes not counted exactly once", u.hotness.routes[0][:2])
	}
	if err := s.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	before := u.hotness.routes
	if _, _, err := u.SelectRoutes(source(t), c, 0, []int{0, 2}); err == nil || before != u.hotness.routes {
		t.Fatal("invalid route changed hotness")
	}
	if err := u.Close(); err != nil {
		t.Fatal(err)
	}
	before = u.hotness.routes
	if _, _, err := u.SelectRoutes(source(t), c, 0, []int{0}); err == nil || before != u.hotness.routes {
		t.Fatal("closed route changed hotness")
	}
}

func (b *fakeMetalUpload) pageSize() uint64 { return max(b.pageBytes, 1) }

func (b *fakeMetalUpload) allocate(n int, strides [3]int) error {
	b.buf = make([]byte, max(strides[0], strides[1], strides[2]))
	b.planes = make([][3][]byte, n)
	for i := range b.planes {
		for c, s := range strides {
			b.planes[i][c] = make([]byte, s)
		}
	}
	return nil
}
func (b *fakeMetalUpload) stage() []byte { return b.buf }
func (b *fakeMetalUpload) write(slot, c int, data []byte) error {
	if b.pinFailure != nil {
		return b.pinFailure
	}
	if b.failWrite {
		return errors.New("write")
	}
	copy(b.planes[slot][c], data)
	b.writes++
	return nil
}

func TestMetalPinBackoffOwnership(t *testing.T) {
	b := new(fakeMetalUpload)
	u, err := newMetalUpload(b, 3, [3]int{2, 2, 2}, 18)
	if err != nil {
		t.Fatal(err)
	}
	c, err := New(identity(), 3, 6, 18)
	if err != nil {
		t.Fatal(err)
	}
	tr := Triplet{Tensor{Offset: 0, Bytes: 6, Stride: 2, Experts: 3}, Tensor{Offset: 6, Bytes: 6, Stride: 2, Experts: 3}, Tensor{Offset: 12, Bytes: 6, Stride: 2, Experts: 3}}
	src, err := NewSource(strings.NewReader("abcdefghijklmnopqr"), 18, identity(), map[int]Triplet{0: tr})
	if err != nil {
		t.Fatal(err)
	}
	s, err := u.Select(src, c, 0, []int{0, 1}, false)
	if err != nil {
		t.Fatal(err)
	}
	failure := &MetalPinFailure{LockedSlots: 1, LockedBytes: 6, Err: errors.New("mlock")}
	if _, err := u.PinBackoff(c, failure); err == nil {
		t.Fatal("active backoff moved consumer")
	}
	if err := s.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	b.pinFailure = failure
	s, err = u.Select(src, c, 0, []int{2}, false)
	if err == nil || s == nil || s.Ready() || u.failed {
		t.Fatal("pin failure ownership", err)
	}
	if err := s.Discard(); err != nil {
		t.Fatal(err)
	}
	if _, err := u.PinBackoff(cache(t), failure); err == nil {
		t.Fatal("foreign cache cap accepted")
	}
	cap, err := u.PinBackoff(c, failure)
	if err != nil || cap != 1 || len(c.slots) != 1 || u.slots != 1 {
		t.Fatal("backoff", cap, err)
	}
	b.pinFailure = nil
	writes := b.writes
	s, err = u.Select(src, c, 0, []int{0}, false)
	if err != nil || b.writes != writes {
		t.Fatal("locked hit lost", err)
	}
	if err := s.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	if err := u.Close(); err != nil {
		t.Fatal(err)
	}
}
func (b *fakeMetalUpload) close() error {
	if b.failClose {
		return errors.New("close")
	}
	b.closed = true
	return nil
}

func TestMetalUploadConsumerOwnership(t *testing.T) {
	b := new(fakeMetalUpload)
	u, err := newMetalUpload(b, 2, [3]int{2, 2, 2}, 12)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := u.Submit(Upload{Slot: 0, Generation: 1, Key: key(0)}, 0, []byte("xx")); err == nil || b.writes != 0 {
		t.Fatal("unowned upload accepted")
	}
	c := cache(t)
	s, err := u.Select(source(t), c, 0, []int{1, 0, 1}, false)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := s.Slots()
	if err != nil || !reflect.DeepEqual(ids, []int{0, 1, 0}) {
		t.Fatal(ids, err)
	}
	if b.writes != 6 {
		t.Fatal("duplicate upload", b.writes)
	}
	for route, slot := range ids {
		var got string
		for _, p := range b.planes[slot] {
			got += string(p)
		}
		want := []string{"cdghkl", "abefij", "cdghkl"}[route]
		if got != want {
			t.Fatalf("route %d bytes %q want %q", route, got, want)
		}
	}
	if _, err := u.Select(source(t), c, 0, []int{0}, false); err == nil {
		t.Fatal("active lease reused")
	}
	if err := u.Close(); err == nil || b.closed {
		t.Fatal("active storage freed")
	}
	if err := s.Release(&completion{err: errors.New("pending")}); err == nil {
		t.Fatal("wait failure hidden")
	}
	if _, err := s.Slots(); err != nil {
		t.Fatal("failed wait lost lease", err)
	}
	if err := s.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Slots(); err == nil {
		t.Fatal("released mapping accepted")
	}
	s, err = u.Select(source(t), c, 0, []int{0, 1, 0}, false)
	if err != nil || b.writes != 6 {
		t.Fatal("hit bytes uploaded", b.writes, err)
	}
	if err := s.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
	b.failClose = true
	if err := u.Close(); err == nil || u.closed {
		t.Fatal("failed teardown lost owner")
	}
	if _, err := u.Select(source(t), c, 0, []int{0}, false); err == nil {
		t.Fatal("closing owner accepted work")
	}
	b.failClose = false
	if err := u.Close(); err != nil || !b.closed {
		t.Fatal("close retry", err)
	}
}

func TestMetalUploadFaults(t *testing.T) {
	for _, name := range []string{"graph", "write", "source", "budget"} {
		t.Run(name, func(t *testing.T) {
			b := new(fakeMetalUpload)
			budget := uint64(12)
			if name == "budget" {
				budget = 11
			}
			u, err := newMetalUpload(b, 2, [3]int{2, 2, 2}, budget)
			if name == "budget" {
				if err == nil {
					t.Fatal("budget accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if name == "write" {
				b.failWrite = true
			}
			src := source(t)
			if name == "source" {
				src.identity.Model[0]++
			}
			_, err = u.Select(src, cache(t), 0, []int{0}, name == "graph")
			if err == nil {
				t.Fatal("fault accepted")
			}
			if name == "graph" && b.writes != 0 {
				t.Fatal("graph rejection too late")
			}
			if err := u.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
