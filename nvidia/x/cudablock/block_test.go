package cudablock_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudablock"
)

func TestHeader(t *testing.T) {
	if len(cudablock.Header) == 0 {
		t.Fatal("expected non-empty cudablock.Header")
	}
	if !strings.Contains(cudablock.Header, "warp_reduce_sum") {
		t.Error("expected cudablock.Header to define warp_reduce_sum")
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
