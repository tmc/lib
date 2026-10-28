package cudaattn_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaattn"
)

func ExamplePadAttentionHeads() {
	padded, padCount, err := cudaattn.PadAttentionHeads(48)
	if err != nil {
		panic(err)
	}
	fmt.Printf("padded=%d padCount=%d\n", padded, padCount)
	// Output:
	// padded=64 padCount=16
}

func ExampleCandidateBlockMaxCPU() {
	cfg := cudaattn.CandidateBlockConfig{
		BlockSize:        4,
		BlockTopK:        3,
		TokenTopK:        8,
		RecentBlocks:     1,
		ChunkBlocks:      2,
		ComputeChunkSize: 8192,
	}
	scores := []float32{0.1, 2.5, 9.0, 1.0, 0.2}
	blocks, tokens, err := cudaattn.CandidateBlockMaxCPU(scores, 18, cfg)
	if err != nil {
		panic(err)
	}
	fmt.Printf("blocks=%v paddedLen=%d\n", blocks, len(tokens))
	// Output:
	// blocks=[0 4 2] paddedLen=16
}
