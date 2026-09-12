package cuptiactivity

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestCollectionCompletion(t *testing.T) {
	size := int(unsafe.Sizeof(cupti.ActivityAPI{}))
	pool, err := newBufferPool(size, 1)
	if err != nil {
		t.Fatal(err)
	}
	ptr, bytes := pool.request()
	// Reach test-owned storage through the allocation, not uintptr conversion.
	b := pool.buffers[ptr]
	a := (*cupti.ActivityAPI)(unsafe.Pointer(&b.words[0]))
	*a = cupti.ActivityAPI{Kind: cupti.ActivityKindDriver, Start: 10, End: 20, CorrelationId: 7}
	iterations := 0
	c := collection{pool: pool, limit: 0,
		layouts: map[cupti.ActivityKind]uint{cupti.ActivityKindDriver: uint(size)},
		next: func(_ *uint8, _ uint, out **cupti.Activity) error {
			iterations++
			if iterations > 1 {
				return cupti.ErrMaxLimitReached
			}
			*out = (*cupti.Activity)(unsafe.Pointer(a))
			return nil
		},
		dropped: func(cupti.Context, uint32) (uint, error) { return 3, nil },
	}
	c.complete(0, 0, ptr, bytes, bytes)
	if c.discarded != 1 || c.nativeDropped != 3 || c.failures != 0 || len(c.records) != 0 {
		t.Fatalf("discarded=%d dropped=%d failures=%d records=%d", c.discarded, c.nativeDropped, c.failures, len(c.records))
	}
	if pending, _ := pool.status(); pending != 0 {
		t.Fatal("buffer retained after callback")
	}
	ptr, bytes = pool.request()
	c.complete(0, 0, ptr, bytes, 0)
	if c.nativeDropped != 6 {
		t.Fatalf("cumulative dropped=%d", c.nativeDropped)
	}
}

func TestCollectionErrors(t *testing.T) {
	pool, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("native dropped query failed")
	c := collection{pool: pool, dropped: func(cupti.Context, uint32) (uint, error) { return 0, want }}
	ptr, size := pool.request()
	c.complete(0, 0, ptr, size, 0)
	if c.failures != 1 || !errors.Is(c.firstError, want) {
		t.Fatalf("failures=%d error=%v", c.failures, c.firstError)
	}
	c.complete(0, 0, 0, 0, 0)
	if c.failures != 3 || !errors.Is(c.firstError, want) {
		t.Fatal("callback errors lost")
	}
}

func TestExternalCorrelationIsNotIncompleteTiming(t *testing.T) {
	pool, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	ptr, size := pool.request()
	a := (*cupti.ActivityExternalCorrelation)(unsafe.Pointer(&pool.buffers[ptr].words[0]))
	*a = cupti.ActivityExternalCorrelation{Kind: cupti.ActivityKindExternalCorrelation, ExternalKind: cupti.ExternalCorrelationKindCustom0, ExternalId: 99, CorrelationId: 42}
	calls := 0
	c := collection{pool: pool, limit: 1, layouts: map[cupti.ActivityKind]uint{a.Kind: uint(unsafe.Sizeof(*a))}, next: func(_ *uint8, _ uint, out **cupti.Activity) error {
		calls++
		if calls > 1 {
			return cupti.ErrMaxLimitReached
		}
		*out = (*cupti.Activity)(unsafe.Pointer(a))
		return nil
	}, dropped: func(cupti.Context, uint32) (uint, error) { return 0, nil }}
	c.complete(0, 0, ptr, size, uint(unsafe.Sizeof(*a)))
	if c.incomplete != 0 || c.failures != 0 || len(c.records) != 1 || c.records[0].ExternalID != 99 {
		t.Fatalf("incomplete=%d failures=%d records=%v", c.incomplete, c.failures, c.records)
	}
}

func TestMemoryInstantCompleteness(t *testing.T) {
	for _, timestamp := range []uint64{0, 10} {
		size := int(unsafe.Sizeof(cupti.ActivityMemory4{}))
		pool, err := newBufferPool(size, 1)
		if err != nil {
			t.Fatal(err)
		}
		ptr, n := pool.request()
		a := (*cupti.ActivityMemory4)(unsafe.Pointer(&pool.buffers[ptr].words[0]))
		*a = cupti.ActivityMemory4{Kind: cupti.ActivityKindMemory2, Timestamp: timestamp}
		calls := 0
		c := collection{pool: pool, limit: 1, layouts: map[cupti.ActivityKind]uint{a.Kind: uint(size)}, next: func(_ *uint8, _ uint, out **cupti.Activity) error {
			calls++
			if calls > 1 {
				return cupti.ErrMaxLimitReached
			}
			*out = (*cupti.Activity)(unsafe.Pointer(a))
			return nil
		}, dropped: func(cupti.Context, uint32) (uint, error) { return 0, nil }}
		c.complete(0, 0, ptr, n, n)
		want := uint64(0)
		if timestamp == 0 {
			want = 1
		}
		if c.incomplete != want || c.failures != 0 || len(c.records) != 1 {
			t.Fatalf("timestamp=%d incomplete=%d failures=%d", timestamp, c.incomplete, c.failures)
		}
	}
}
