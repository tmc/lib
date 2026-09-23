package cudakv

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestFP8ConstructionOwnership(t *testing.T) {
	failure := errors.New("allocate failed")
	for failAt := range 8 {
		calls := 0
		allocate := func(n int, scale bool) error {
			index := calls
			calls++
			if scale != (index%4 >= 2) {
				t.Fatal("wrong allocation type/order")
			}
			want := 64
			if scale {
				want = 2
			}
			if n != want {
				t.Fatalf("elements=%d want=%d", n, want)
			}
			if index == failAt {
				return failure
			}
			return nil
		}
		c, err := newFP8Cache(nil, 2, 2, 4, 8, 64,
			func(n int) (cudamem.Slice[uint8], error) { return cudamem.Slice[uint8]{}, allocate(n, false) },
			func(n int) (cudamem.Slice[float32], error) { return cudamem.Slice[float32]{}, allocate(n, true) })
		if !errors.Is(err, failure) || c == nil || calls != failAt+1 || len(c.layers) != failAt/4+1 {
			t.Fatalf("failed construction: calls=%d err=%v", calls, err)
		}
		if _, err := c.Layer(0); err == nil {
			t.Fatal("partial FP8 cache exposed")
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFP8CloseRetries(t *testing.T) {
	failure := errors.New("free failed")
	for failAt := range 8 {
		c := &FP8Cache{layers: make([]FP8LayerCache, 2)}
		calls := 0
		free := func(scale bool) error {
			if scale != (c.freed%4 >= 2) {
				t.Fatal("wrong release type")
			}
			calls++
			if calls-1 == failAt {
				return failure
			}
			return nil
		}
		freeBytes := func(cudamem.Slice[uint8]) error { return free(false) }
		freeScale := func(cudamem.Slice[float32]) error { return free(true) }
		if err := c.closeWith(freeBytes, freeScale); !errors.Is(err, failure) {
			t.Fatal(err)
		}
		if c.freed != failAt || c.layers == nil {
			t.Fatal("failed cleanup owner lost")
		}
		if _, err := c.Layer(0); err == nil {
			t.Fatal("closing FP8 cache exposed")
		}
		for range 2 {
			if err := c.closeWith(freeBytes, freeScale); err != nil {
				t.Fatal(err)
			}
		}
		if calls != 9 || c.layers != nil {
			t.Fatalf("release calls=%d", calls)
		}
	}
}

func TestFP8ExtentChecks(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, dims := range [][4]int{{0, 1, 1, 1}, {1, maxInt, 2, 2}, {1, maxInt/8 + 1, 1, 1}, {maxInt / 2, 1, 1, 1}} {
		if _, err := fp8Elements(dims[0], dims[1], dims[2], dims[3]); err == nil {
			t.Fatalf("accepted dimensions %v", dims)
		}
	}
	if elements, err := fp8Elements(2, 2, 4, 8); err != nil || elements != 64 {
		t.Fatalf("elements=%d err=%v", elements, err)
	}
}
