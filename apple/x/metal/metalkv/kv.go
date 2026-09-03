package metalkv

import (
	"fmt"
	"sync"

	applemetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// LayerCache holds the allocated K and V tensor memory buffers for a single transformer layer.
type LayerCache struct {
	LayerIndex int
	K          mempool.Slice[float32]
	V          mempool.Slice[float32]
}

// Cache manages multi-layer KV memory buffers for autoregressive decoding on Apple Silicon.
type Cache struct {
	mu         sync.RWMutex
	closed     bool
	dev        *applemetal.Device
	numLayers  int
	numKVHeads int
	headDim    int
	maxSeqLen  int
	layers     []LayerCache
	mem        *metalmem.Arena
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
	if numKVHeads > int(^uint(0)>>1)/headDim || maxSeqLen > int(^uint(0)>>1)/(numKVHeads*headDim) {
		return nil, fmt.Errorf("metalkv: dimensions overflow")
	}

	elementsPerLayer := maxSeqLen * numKVHeads * headDim
	layers := make([]LayerCache, numLayers)

	mem := metalmem.NewArena(dev)
	for i := range numLayers {
		layers[i] = LayerCache{
			LayerIndex: i,
			K:          metalmem.AllocIn[float32](mem, fmt.Sprintf("layer[%d].K", i), elementsPerLayer),
			V:          metalmem.AllocIn[float32](mem, fmt.Sprintf("layer[%d].V", i), elementsPerLayer),
		}
	}
	if err := mem.Err(); err != nil {
		mem.Free()
		return nil, err
	}

	return &Cache{
		dev:        dev,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
		layers:     layers,
		mem:        mem,
	}, nil
}

// Layer returns the KV buffers for the specified layer index.
func (c *Cache) Layer(idx int) (LayerCache, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return LayerCache{}, fmt.Errorf("metalkv: cache closed")
	}
	if idx < 0 || idx >= len(c.layers) {
		return LayerCache{}, fmt.Errorf("metalkv: layer index %d out of bounds (total %d)", idx, len(c.layers))
	}
	return c.layers[idx], nil
}

// Pos returns current token sequence position.
func (c *Cache) Pos() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.pos
}

// Advance advances the token sequence position by n.
func (c *Cache) Advance(n int) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return fmt.Errorf("metalkv: cache closed")
	}
	if n < 0 || n > c.maxSeqLen-c.pos {
		return fmt.Errorf("metalkv: advance %d from %d exceeds sequence capacity %d", n, c.pos, c.maxSeqLen)
	}
	c.pos += n
	return nil
}

// Reset rewinds the token sequence position to zero.
func (c *Cache) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	c.pos = 0
}

// Close releases the cache buffers. Closing twice is safe.
func (c *Cache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.mem.Free()
	c.closed = true
	c.layers = nil
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
	mu         sync.RWMutex
	closed     bool
	dev        *applemetal.Device
	numLayers  int
	numKVHeads int
	headDim    int
	maxSeqLen  int
	layers     []FP8LayerCache
	mem        *metalmem.Arena
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
	if numKVHeads > int(^uint(0)>>1)/headDim || maxSeqLen > int(^uint(0)>>1)/(numKVHeads*headDim) {
		return nil, fmt.Errorf("metalkv: dimensions overflow")
	}

	elementsPerLayer := maxSeqLen * numKVHeads * headDim
	layers := make([]FP8LayerCache, numLayers)

	mem := metalmem.NewArena(dev)
	for i := range numLayers {
		layers[i] = FP8LayerCache{
			LayerIndex: i,
			K:          metalmem.AllocIn[uint8](mem, fmt.Sprintf("layer[%d].K", i), elementsPerLayer),
			V:          metalmem.AllocIn[uint8](mem, fmt.Sprintf("layer[%d].V", i), elementsPerLayer),
			KScale:     metalmem.AllocIn[float32](mem, fmt.Sprintf("layer[%d].KScale", i), numKVHeads),
			VScale:     metalmem.AllocIn[float32](mem, fmt.Sprintf("layer[%d].VScale", i), numKVHeads),
		}
	}
	if err := mem.Err(); err != nil {
		mem.Free()
		return nil, err
	}

	return &FP8Cache{
		dev:        dev,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
		layers:     layers,
		mem:        mem,
	}, nil
}

// Layer returns the FP8 KV buffers for the specified layer index.
func (c *FP8Cache) Layer(idx int) (FP8LayerCache, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return FP8LayerCache{}, fmt.Errorf("metalkv: cache closed")
	}
	if idx < 0 || idx >= len(c.layers) {
		return FP8LayerCache{}, fmt.Errorf("metalkv: layer index %d out of bounds (total %d)", idx, len(c.layers))
	}
	return c.layers[idx], nil
}

// Close releases the FP8 cache buffers. Closing twice is safe.
func (c *FP8Cache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.mem.Free()
	c.closed = true
	c.layers = nil
	return nil
}
