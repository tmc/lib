package cudamem_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestBucketSize(t *testing.T) {
	tests := []struct {
		input uint
		want  uint
	}{
		// Zero size
		{0, 256},

		// Tier 1 (<= 1 KiB): 256-byte granularity
		{1, 256},
		{128, 256},
		{256, 256},
		{257, 512},
		{512, 512},
		{513, 768},
		{1024, 1024},

		// Tier 2 (1 KiB - 1 MiB): Power-of-two
		{1025, 2048},
		{2000, 2048},
		{2048, 2048},
		{2049, 4096},
		{65000, 65536},
		{1000000, 1048576},
		{1048576, 1048576},

		// Tier 3 (> 1 MiB): 2 MiB huge-page granularity
		{1048577, 2 * 1024 * 1024},
		{2 * 1024 * 1024, 2 * 1024 * 1024},
		{2*1024*1024 + 1, 4 * 1024 * 1024},
		{5 * 1024 * 1024, 6 * 1024 * 1024},
	}

	for _, tt := range tests {
		got := cudamem.BucketSize(tt.input)
		if got != tt.want {
			t.Errorf("BucketSize(%d) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestVariedSequenceLengthBucketing(t *testing.T) {
	// Simulate dynamic token generation (seq lengths 1 to 512 tokens with float32 embeddings)
	const hiddenDim = 128 // 128 * 4 bytes = 512 bytes per token
	tokenBytes := func(seqLen int) uint {
		return uint(seqLen * hiddenDim * 4)
	}

	// SeqLen 1 (512 bytes) -> bucket 512 bytes
	b1 := cudamem.BucketSize(tokenBytes(1))
	if b1 != 512 {
		t.Errorf("expected bucket 512 for seqLen 1, got %d", b1)
	}

	// SeqLen 2 (1024 bytes) -> bucket 1024 bytes
	b2 := cudamem.BucketSize(tokenBytes(2))
	if b2 != 1024 {
		t.Errorf("expected bucket 1024 for seqLen 2, got %d", b2)
	}

	// SeqLen 3 (1536 bytes) -> bucket 2048 bytes
	b3 := cudamem.BucketSize(tokenBytes(3))
	if b3 != 2048 {
		t.Errorf("expected bucket 2048 for seqLen 3, got %d", b3)
	}

	// SeqLen 4 (2048 bytes) -> bucket 2048 bytes (reuses same slab as SeqLen 3!)
	b4 := cudamem.BucketSize(tokenBytes(4))
	if b4 != 2048 {
		t.Errorf("expected bucket 2048 for seqLen 4, got %d", b4)
	}

	if b3 != b4 {
		t.Errorf("seqLen 3 and seqLen 4 should share same bucket for zero-fragmentation reuse")
	}
}
