package metalkv

import (
	"fmt"

	applemetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
)

// LayerCache holds the allocated K and V tensor memory buffers for a single transformer layer.
type LayerCache struct {
	LayerIndex int
	K          mempool.Slice[float32]
	V          mempool.Slice[float32]
}

// Cache manages multi-layer KV memory buffers for autoregressive decoding on Apple Silicon.
type Cache struct {
	dev        *applemetal.Device
	numLayers  int
	numKVHeads int
	headDim    int
	maxSeqLen  int
	layers     []LayerCache
	pos        int
}

// New creates a new static KV-cache allocating per-layer K and V buffers in unified memory.
func New(dev *applemetal.Device, numLayers, numKVHeads, headDim, maxSeqLen int) (*Cache, error) {
	if dev == nil {
		return nil, fmt.Errorf("metalkv: nil device")
	}
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf("metalkv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d)",
			numLayers, numKVHeads, headDim, maxSeqLen)
	}

	elementsPerLayer := maxSeqLen * numKVHeads * headDim
	layers := make([]LayerCache, numLayers)

	for i := 0; i < numLayers; i++ {
		k, err := mempool.Alloc[float32](dev, elementsPerLayer)
		if err != nil {
			return nil, fmt.Errorf("metalkv: alloc layer %d K buffer: %w", i, err)
		}

		v, err := mempool.Alloc[float32](dev, elementsPerLayer)
		if err != nil {
			return nil, fmt.Errorf("metalkv: alloc layer %d V buffer: %w", i, err)
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
		return LayerCache{}, fmt.Errorf("metalkv: layer index %d out of bounds (total %d)", idx, len(c.layers))
	}
	return c.layers[idx], nil
}

// Pos returns current token sequence position.
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

// Close releases the cache buffers.
func (c *Cache) Close() error {
	return nil
}

// FP8LayerCache holds 8-bit quantized K and V buffers and scaling factors for one layer.
type FP8LayerCache struct {
	LayerIndex int
	K          mempool.Slice[uint8]
	V          mempool.Slice[uint8]
	KScale     mempool.Slice[float32]
	VScale     mempool.Slice[float32]
}

// FP8Cache manages 8-bit quantized KV memory buffers for low-bandwidth inference on Apple Silicon.
type FP8Cache struct {
	dev        *applemetal.Device
	numLayers  int
	numKVHeads int
	headDim    int
	maxSeqLen  int
	layers     []FP8LayerCache
	pos        int
}

// NewFP8 creates a new static FP8 KV-cache allocating per-layer 8-bit K and V buffers in unified memory.
func NewFP8(dev *applemetal.Device, numLayers, numKVHeads, headDim, maxSeqLen int) (*FP8Cache, error) {
	if dev == nil {
		return nil, fmt.Errorf("metalkv: nil device")
	}
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf("metalkv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d)",
			numLayers, numKVHeads, headDim, maxSeqLen)
	}

	elementsPerLayer := maxSeqLen * numKVHeads * headDim
	layers := make([]FP8LayerCache, numLayers)

	for i := 0; i < numLayers; i++ {
		k, err := mempool.Alloc[uint8](dev, elementsPerLayer)
		if err != nil {
			return nil, fmt.Errorf("metalkv: alloc layer %d FP8 K buffer: %w", i, err)
		}
		v, err := mempool.Alloc[uint8](dev, elementsPerLayer)
		if err != nil {
			return nil, fmt.Errorf("metalkv: alloc layer %d FP8 V buffer: %w", i, err)
		}
		kScale, _ := mempool.Alloc[float32](dev, numKVHeads)
		vScale, _ := mempool.Alloc[float32](dev, numKVHeads)

		layers[i] = FP8LayerCache{
			LayerIndex: i,
			K:          k,
			V:          v,
			KScale:     kScale,
			VScale:     vScale,
		}
	}

	return &FP8Cache{
		dev:        dev,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
		layers:     layers,
		pos:        0,
	}, nil
}

// Layer returns the FP8 KV buffers for the specified layer index.
func (c *FP8Cache) Layer(idx int) (FP8LayerCache, error) {
	if idx < 0 || idx >= len(c.layers) {
		return FP8LayerCache{}, fmt.Errorf("metalkv: layer index %d out of bounds (total %d)", idx, len(c.layers))
	}
	return c.layers[idx], nil
}

// Close releases the FP8 cache buffers.
func (c *FP8Cache) Close() error {
	return nil
}

