package cudakv_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestKVInvalidDims(t *testing.T) {
	_, err := cudakv.New(nil, -1, 8, 128, 2048)
	if err == nil {
		t.Fatal("expected error on invalid dimensions")
	}
}

func TestKVCacheAllocation(t *testing.T) {
	dev := cudatest.Require(t)
	cache, err := cudakv.New(dev, 2, 4, 64, 512)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer cache.Close()

	layer0, err := cache.Layer(0)
	if err != nil {
		t.Fatalf("Layer(0) failed: %v", err)
	}

	if layer0.K.Len() != 4*64*512 {
		t.Errorf("expected K len %d, got %d", 4*64*512, layer0.K.Len())
	}

	if cache.Pos() != 0 {
		t.Errorf("expected initial pos 0, got %d", cache.Pos())
	}
	cache.Advance(1)
	if cache.Pos() != 1 {
		t.Errorf("expected pos 1, got %d", cache.Pos())
	}
	cache.Reset()
	if cache.Pos() != 0 {
		t.Errorf("expected pos 0, got %d", cache.Pos())
	}
}
