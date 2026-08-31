package cudablock_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudablock"
)

func TestHeader(t *testing.T) {
	header := cudablock.Header
	if !strings.Contains(header, "warp_reduce_sum") {
		t.Error("expected warp_reduce_sum in header")
	}
	if !strings.Contains(header, "tile_load_2d") {
		t.Error("expected tile_load_2d in header")
	}
	if !strings.Contains(header, "dequant_int4") {
		t.Error("expected dequant_int4 in header")
	}
}

func TestTileConfig(t *testing.T) {
	tc := cudablock.TileConfig{
		BlockThreads: 256,
		TileM:        16,
		TileN:        16,
		TileK:        16,
		ElemBytes:    4, // float32
	}

	// (16*16 + 16*16) * 4 = 512 * 4 = 2048 bytes
	shmem := tc.SharedMemBytes()
	if shmem != 2048 {
		t.Errorf("expected 2048 shared memory bytes, got %d", shmem)
	}
}
