package cudakv_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestKVInvalidDims(t *testing.T) {
	tests := []struct {
		name string
		fn   func() error
	}{
		{"cache", func() error {
			_, err := cudakv.New(nil, -1, 8, 128, 2048)
			return err
		}},
		{"fp8 cache", func() error {
			_, err := cudakv.NewFP8(nil, -1, 8, 128, 2048)
			return err
		}},
		{"paged cache", func() error {
			_, err := cudakv.NewPagedCache(nil, -1, 8, 128, 16, 64)
			return err
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.fn(); err == nil {
				t.Fatal("constructor accepted invalid dimensions")
			}
		})
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

func TestPagedCacheAllocation(t *testing.T) {
	dev := cudatest.Require(t)
	pc, err := cudakv.NewPagedCache(dev, 2, 4, 64, 16, 64)
	if err != nil {
		t.Fatalf("NewPagedCache failed: %v", err)
	}
	defer pc.Close()

	if pc.NumFreeBlocks() != 64 {
		t.Errorf("expected 64 free blocks, got %d", pc.NumFreeBlocks())
	}

	// Sequence 1: 30 tokens -> needs ceil(30/16) = 2 blocks
	if err := pc.AllocSequence(1001, 30); err != nil {
		t.Fatalf("AllocSequence failed: %v", err)
	}
	if pc.NumFreeBlocks() != 62 {
		t.Errorf("expected 62 free blocks, got %d", pc.NumFreeBlocks())
	}

	// Append tokens to cross block boundary (from 30 to 32 -> 33 tokens -> needs 3rd block)
	_ = pc.AppendToken(1001) // token 31
	_ = pc.AppendToken(1001) // token 32
	_ = pc.AppendToken(1001) // token 33 -> allocates 3rd block

	if pc.NumFreeBlocks() != 61 {
		t.Errorf("expected 61 free blocks after expansion, got %d", pc.NumFreeBlocks())
	}

	pc.FreeSequence(1001)
	if pc.NumFreeBlocks() != 64 {
		t.Errorf("expected 64 free blocks after recycle, got %d", pc.NumFreeBlocks())
	}
}
