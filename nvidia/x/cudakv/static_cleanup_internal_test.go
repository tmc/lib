package cudakv

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestStaticCacheConstructionFailure(t *testing.T) {
	failure := errors.New("allocation failed")
	for failAt := range 4 {
		calls := 0
		c, err := newStaticCache(nil, 2, 2, 4, 8, func(elements int) (cudamem.Slice[float32], error) {
			if elements != 64 {
				t.Fatalf("elements=%d", elements)
			}
			index := calls
			calls++
			if index == failAt {
				return cudamem.Slice[float32]{}, failure
			}
			return cudamem.Slice[float32]{}, nil
		})
		if !errors.Is(err, failure) || c == nil || calls != failAt+1 {
			t.Fatalf("owner=%v calls=%d err=%v", c, calls, err)
		}
		if _, err := c.Layer(0); err == nil {
			t.Fatal("partial cache exposed")
		}
		if err := c.Advance(1); err == nil {
			t.Fatal("partial cache advanced")
		}
		if err := c.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestStaticCacheCloseRetry(t *testing.T) {
	failure := errors.New("free failed")
	for failAt := range 4 {
		c := &Cache{layers: make([]LayerCache, 2), maxSeqLen: 8}
		calls := 0
		if err := c.closeWith(func(cudamem.Slice[float32]) error {
			index := calls
			calls++
			if index == failAt {
				return failure
			}
			return nil
		}); !errors.Is(err, failure) {
			t.Fatalf("close=%v", err)
		}
		if c.freed != failAt || c.layers == nil {
			t.Fatal("lost failed release or repeated successful releases")
		}
		if _, err := c.Layer(0); err == nil {
			t.Fatal("closing cache exposed buffers")
		}
		for range 2 {
			if err := c.closeWith(func(cudamem.Slice[float32]) error { calls++; return nil }); err != nil {
				t.Fatal(err)
			}
		}
		if calls != 5 || c.layers != nil {
			t.Fatalf("release calls=%d", calls)
		}
	}
}

func TestStaticCacheOverflow(t *testing.T) {
	if c, err := NewTracked(&cuda.Device{}, nil, 2, 2, 4, int(^uint(0)>>1)); c != nil || err == nil {
		t.Fatal("overflow accepted before allocation")
	}
}
