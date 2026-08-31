package cudakvrot

import (
	"fmt"
)

// SlidingWindowCache manages a fixed-size rotating buffer for sliding window attention.
type SlidingWindowCache struct {
	windowSize int
	headDim    int
	numKVHeads int
	numLayers  int
	curPos     int
}

// NewSlidingWindowCache creates a cache manager for windowSize context tokens.
func NewSlidingWindowCache(numLayers, numKVHeads, headDim, windowSize int) (*SlidingWindowCache, error) {
	if windowSize <= 0 || headDim <= 0 || numKVHeads <= 0 || numLayers <= 0 {
		return nil, fmt.Errorf("cudakvrot: invalid dimensions")
	}
	return &SlidingWindowCache{
		windowSize: windowSize,
		headDim:    headDim,
		numKVHeads: numKVHeads,
		numLayers:  numLayers,
		curPos:     0,
	}, nil
}

// SlotForPosition returns the physical circular buffer index for logical position pos.
func (swc *SlidingWindowCache) SlotForPosition(pos int) int {
	return pos % swc.windowSize
}

// ContextBounds returns the start and end logical token indices visible to attention at position pos.
func (swc *SlidingWindowCache) ContextBounds(pos int) (start, end int) {
	end = pos + 1
	start = pos + 1 - swc.windowSize
	if start < 0 {
		start = 0
	}
	return start, end
}

// Advance advances the current logical token position by n.
func (swc *SlidingWindowCache) Advance(n int) {
	swc.curPos += n
}

// Chunk represents a segment of a long prompt for chunked prefill.
type Chunk struct {
	Index      int
	StartToken int
	EndToken   int
	Tokens     []int32
}

// ChunkPrompt splits a prompt of tokens into chunks of max chunkSize.
func ChunkPrompt(prompt []int32, chunkSize int) []Chunk {
	if chunkSize <= 0 || len(prompt) == 0 {
		return nil
	}

	numChunks := (len(prompt) + chunkSize - 1) / chunkSize
	chunks := make([]Chunk, numChunks)

	for i := 0; i < numChunks; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(prompt) {
			end = len(prompt)
		}

		chunks[i] = Chunk{
			Index:      i,
			StartToken: start,
			EndToken:   end,
			Tokens:     prompt[start:end],
		}
	}

	return chunks
}
