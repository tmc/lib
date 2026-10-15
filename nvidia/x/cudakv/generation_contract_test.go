package cudakv

import (
	"errors"
	"reflect"
	"testing"
	"unsafe"
)

func TestRingGenerationConstructorAdmission(t *testing.T) {
	limit := uint64(^uint(0)>>1) / uint64(unsafe.Sizeof(Descriptor{}))
	for _, tt := range []struct {
		name                 string
		capacity, generation uint64
	}{
		{"zero capacity", 0, 1},
		{"zero generation", 1, 0},
		{"unrepresentable bytes", limit + 1, 1},
		{"unrepresentable length", ^uint64(0), 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if ring, err := NewDeviceCommandRing(tt.capacity, tt.generation); err == nil || ring != nil {
				t.Fatalf("constructor returned %v, %v; want nil ring and error", ring, err)
			}
		})
	}
	// This generation needs no increment, truncation, clock or global counter.
	r, err := NewDeviceCommandRing(1, ^uint64(0))
	if err != nil {
		t.Fatal(err)
	}
	want := Descriptor{SeqID: ^uint64(0), Generation: ^uint64(0), PromptToken: -17, Flags: ^uint32(0)}
	if err := r.Publish(want); err != nil {
		t.Fatal(err)
	}
	if got, ok := r.Consume(); !ok || got != want {
		t.Fatalf("max generation roundtrip = %+v, %v", got, ok)
	}
}

type ringState struct {
	head, tail, queued int
	values             []Descriptor
}

func snapshotRing(r *DeviceCommandRing) ringState {
	r.mu.Lock()
	defer r.mu.Unlock()
	return ringState{r.head, r.tail, r.queued, append([]Descriptor(nil), r.ring...)}
}

func TestRingGenerationRefusalPreservesState(t *testing.T) {
	for _, generation := range []uint64{0, 6, 8, ^uint64(0)} {
		t.Run(stringGeneration(generation), func(t *testing.T) {
			r, err := NewDeviceCommandRing(3, 7)
			if err != nil {
				t.Fatal(err)
			}
			empty := snapshotRing(r)
			if err := r.Publish(Descriptor{SeqID: 98, Generation: generation}); err == nil {
				t.Fatal("first producer replaced the constructor generation")
			}
			if got := snapshotRing(r); !reflect.DeepEqual(got, empty) {
				t.Fatal("initial refusal changed ring")
			}
			first := Descriptor{SeqID: 10, Generation: 7, PromptToken: -1, Flags: 3}
			if err := r.Publish(first); err != nil {
				t.Fatal(err)
			}
			before := snapshotRing(r)
			if err := r.Publish(Descriptor{SeqID: 99, Generation: generation}); err == nil {
				t.Fatalf("accepted generation %d in generation 7 ring", generation)
			}
			if got := snapshotRing(r); !reflect.DeepEqual(got, before) {
				t.Fatalf("refusal changed ring: %+v -> %+v", before, got)
			}
			for id := uint64(11); id <= 12; id++ {
				if err := r.Publish(Descriptor{SeqID: id, Generation: 7}); err != nil {
					t.Fatal(err)
				}
			}
			full := snapshotRing(r)
			if err := r.Publish(Descriptor{SeqID: 13, Generation: generation}); err == nil || errors.Is(err, errRingFull) {
				t.Fatalf("invalid descriptor was not admitted before capacity check: %v", err)
			}
			if got := snapshotRing(r); !reflect.DeepEqual(got, full) {
				t.Fatal("invalid full-ring publication changed state")
			}
			for id := uint64(10); id <= 12; id++ {
				got, ok := r.Consume()
				if !ok || got.SeqID != id || got.Generation != 7 {
					t.Fatalf("FIFO after refusal: %+v, %v; want seq %d", got, ok, id)
				}
				if id == 10 && got != first {
					t.Fatal("descriptor fields did not roundtrip")
				}
			}
			if r.Len() != 0 {
				t.Fatal("refusal consumed capacity")
			}
		})
	}
}

func TestRingGenerationPartialWrap(t *testing.T) {
	r, err := NewDeviceCommandRing(3, 7)
	if err != nil {
		t.Fatal(err)
	}
	next := uint64(0)
	for i := 0; i < 3; i++ {
		if err := r.Publish(Descriptor{SeqID: next, Generation: 7}); err != nil {
			t.Fatal(err)
		}
		next++
	}
	want := uint64(0)
	for cycle := 0; cycle < 300; cycle++ {
		for i := 0; i < 2; i++ {
			got, ok := r.Consume()
			if !ok || got.SeqID != want {
				t.Fatalf("partial wrap got %+v, %v; want %d", got, ok, want)
			}
			want++
		}
		for i := 0; i < 2; i++ {
			if err := r.Publish(Descriptor{SeqID: next, Generation: 7}); err != nil {
				t.Fatal(err)
			}
			next++
		}
		if r.Len() != 3 || r.head < 0 || r.head >= len(r.ring) || r.tail < 0 || r.tail >= len(r.ring) {
			t.Fatal("slot cursor or occupancy escaped its bound")
		}
	}
	for want < next {
		got, ok := r.Consume()
		if !ok || got.SeqID != want {
			t.Fatalf("final drain got %+v, %v; want %d", got, ok, want)
		}
		want++
	}
}

func stringGeneration(generation uint64) string {
	if generation == 0 {
		return "zero"
	}
	if generation < 7 {
		return "stale"
	}
	if generation == 8 {
		return "future"
	}
	return "max"
}

func TestRingGenerationWrapAndCopy(t *testing.T) {
	for _, capacity := range []uint64{1, 2, 3, 5} {
		r, err := NewDeviceCommandRing(capacity, 19)
		if err != nil {
			t.Fatal(err)
		}
		for cycle := uint64(0); cycle < 300; cycle++ {
			for offset := uint64(0); offset < capacity; offset++ {
				value := Descriptor{SeqID: cycle*capacity + offset, Generation: 19, PromptToken: int32(cycle), Flags: 7}
				if err := r.Publish(value); err != nil {
					t.Fatal(err)
				}
				value.SeqID = ^uint64(0) // publication owns a value copy
			}
			before := snapshotRing(r)
			if err := r.Publish(Descriptor{Generation: 19}); !errors.Is(err, errRingFull) {
				t.Fatalf("full ring: %v", err)
			}
			if got := snapshotRing(r); !reflect.DeepEqual(got, before) {
				t.Fatal("full refusal changed ring")
			}
			if r.Len() != int(capacity) {
				t.Fatal("full ring lost a slot")
			}
			for offset := uint64(0); offset < capacity; offset++ {
				got, ok := r.Consume()
				want := Descriptor{SeqID: cycle*capacity + offset, Generation: 19, PromptToken: int32(cycle), Flags: 7}
				if !ok || got != want {
					t.Fatalf("capacity %d cycle %d: %+v, %v; want %+v", capacity, cycle, got, ok, want)
				}
			}
			before = snapshotRing(r)
			if got, ok := r.Consume(); ok || got != (Descriptor{}) {
				t.Fatal("empty ring returned a descriptor")
			}
			if got := snapshotRing(r); !reflect.DeepEqual(got, before) {
				t.Fatal("empty consume changed cursors")
			}
			for _, value := range before.values {
				if value != (Descriptor{}) {
					t.Fatal("consumed slot retained a descriptor")
				}
			}
		}
	}
}

func TestRingGenerationLifetime(t *testing.T) {
	old, err := NewDeviceCommandRing(1, 7)
	if err != nil {
		t.Fatal(err)
	}
	if err := old.Publish(Descriptor{SeqID: 1, Generation: 7}); err != nil {
		t.Fatal(err)
	}
	old.Close()
	old.Close()
	next, err := NewDeviceCommandRing(1, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := next.Publish(Descriptor{Generation: 7}); err == nil || next.Len() != 0 {
		t.Fatal("new lifetime accepted old generation")
	}
	if err := old.Publish(Descriptor{Generation: 7}); !errors.Is(err, errRingClosed) {
		t.Fatalf("closed old ring: %v", err)
	}
	if got, ok := old.Consume(); !ok || got.SeqID != 1 || got.Generation != 7 {
		t.Fatal("new lifetime lost old queued work")
	}
	if err := next.Publish(Descriptor{SeqID: 2, Generation: 8}); err != nil {
		t.Fatal(err)
	}
	if got, ok := next.Consume(); !ok || got.SeqID != 2 || got.Generation != 8 {
		t.Fatal("new generation did not roundtrip")
	}
}

func TestRingGenerationUninitialized(t *testing.T) {
	var zero DeviceCommandRing
	for _, r := range []*DeviceCommandRing{nil, &zero} {
		if err := r.Publish(Descriptor{Generation: 1}); err == nil {
			t.Fatal("uninitialized publication succeeded")
		}
		if r.Len() != 0 {
			t.Fatal("uninitialized length is not zero")
		}
		if _, ok := r.Consume(); ok {
			t.Fatal("uninitialized consume succeeded")
		}
		r.Close()
	}
}
