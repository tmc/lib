package cudakv

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// LayerCache holds the allocated K and V tensor memory buffers for a single transformer layer.
type LayerCache struct {
	LayerIndex int
	K          cudamem.Slice[float32]
	V          cudamem.Slice[float32]
}

// Cache manages the multi-layer KV memory buffers for autoregressive decoding.
type Cache struct {
	dev        *cuda.Device
	numLayers  int
	numKVHeads int
	headDim    int
	maxSeqLen  int
	layers     []LayerCache
	pos        int
}

// New creates a new static KV-cache allocating per-layer K and V buffers on dev.
func New(dev *cuda.Device, numLayers, numKVHeads, headDim, maxSeqLen int) (*Cache, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf("cudakv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d)",
			numLayers, numKVHeads, headDim, maxSeqLen)
	}

	elementsPerLayer := maxSeqLen * numKVHeads * headDim
	layers := make([]LayerCache, numLayers)

	for i := 0; i < numLayers; i++ {
		k, err := cudamem.Alloc[float32](dev, elementsPerLayer)
		if err != nil {
			// Free previously allocated layers on error
			for j := 0; j < i; j++ {
				_ = layers[j].K.Free()
				_ = layers[j].V.Free()
			}
			return nil, fmt.Errorf("cudakv: alloc layer %d K buffer: %w", i, err)
		}

		v, err := cudamem.Alloc[float32](dev, elementsPerLayer)
		if err != nil {
			_ = k.Free()
			for j := 0; j < i; j++ {
				_ = layers[j].K.Free()
				_ = layers[j].V.Free()
			}
			return nil, fmt.Errorf("cudakv: alloc layer %d V buffer: %w", i, err)
		}

		layers[i] = LayerCache{
			LayerIndex: i,
			K:          k,
			V:          v,
		}
	}

	return &Cache{
		dev:        dev,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
		layers:     layers,
		pos:        0,
	}, nil
}

// Layer returns the KV buffers for the specified layer index.
func (c *Cache) Layer(idx int) (LayerCache, error) {
	if idx < 0 || idx >= len(c.layers) {
		return LayerCache{}, fmt.Errorf("cudakv: layer index %d out of bounds (total %d)", idx, len(c.layers))
	}
	return c.layers[idx], nil
}

// Pos returns the current token sequence position.
func (c *Cache) Pos() int {
	return c.pos
}

// Advance advances the token sequence position by n.
func (c *Cache) Advance(n int) {
	c.pos += n
}

// Reset rewinds the token sequence position to zero.
func (c *Cache) Reset() {
	c.pos = 0
}

// Close releases all GPU memory buffers allocated for the cache.
func (c *Cache) Close() error {
	for i := range c.layers {
		_ = c.layers[i].K.Free()
		_ = c.layers[i].V.Free()
	}
	c.layers = nil
	return nil
}
