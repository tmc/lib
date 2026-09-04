package cudakv

import (
	"fmt"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// LayerCache holds the allocated K and V tensor memory buffers for a single transformer layer.
type LayerCache struct {
	LayerIndex int
	K          cudamem.Slice[float32]
	V          cudamem.Slice[float32]
}

// Cache manages static multi-layer KV memory buffers for single-sequence decoding.
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
	return NewTracked(dev, nil, numLayers, numKVHeads, headDim, maxSeqLen)
}

// NewTracked creates a cache whose device allocations are accounted by tracker.
func NewTracked(dev *cuda.Device, tracker *cudamem.Tracker, numLayers, numKVHeads, headDim, maxSeqLen int) (*Cache, error) {
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
		k, err := cudamem.AllocTracked[float32](dev, tracker, elementsPerLayer)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = layers[j].K.Free()
				_ = layers[j].V.Free()
			}
			return nil, fmt.Errorf("cudakv: alloc layer %d K buffer: %w", i, err)
		}

		v, err := cudamem.AllocTracked[float32](dev, tracker, elementsPerLayer)
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

// FP8LayerCache holds 8-bit quantized K and V buffers and scaling factors for one layer.
type FP8LayerCache struct {
	LayerIndex int
	K          cudamem.Slice[uint8]
	V          cudamem.Slice[uint8]
	KScale     cudamem.Slice[float32]
	VScale     cudamem.Slice[float32]
}

// FP8Cache manages 8-bit quantized KV memory buffers for low-bandwidth inference.
type FP8Cache struct {
	dev        *cuda.Device
	numLayers  int
	numKVHeads int
	headDim    int
	maxSeqLen  int
	layers     []FP8LayerCache
	pos        int
}

// NewFP8 creates a new static FP8 KV-cache allocating per-layer 8-bit K and V buffers on dev.
func NewFP8(dev *cuda.Device, numLayers, numKVHeads, headDim, maxSeqLen int) (*FP8Cache, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf("cudakv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d)",
			numLayers, numKVHeads, headDim, maxSeqLen)
	}

	elementsPerLayer := maxSeqLen * numKVHeads * headDim
	layers := make([]FP8LayerCache, numLayers)

	for i := 0; i < numLayers; i++ {
		k, err := cudamem.Alloc[uint8](dev, elementsPerLayer)
		if err != nil {
			return nil, fmt.Errorf("cudakv: alloc layer %d FP8 K buffer: %w", i, err)
		}
		v, err := cudamem.Alloc[uint8](dev, elementsPerLayer)
		if err != nil {
			_ = k.Free()
			return nil, fmt.Errorf("cudakv: alloc layer %d FP8 V buffer: %w", i, err)
		}
		kScale, _ := cudamem.Alloc[float32](dev, numKVHeads)
		vScale, _ := cudamem.Alloc[float32](dev, numKVHeads)

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
		return FP8LayerCache{}, fmt.Errorf("cudakv: layer index %d out of bounds (total %d)", idx, len(c.layers))
	}
	return c.layers[idx], nil
}

// Close releases all GPU memory buffers allocated for the FP8 cache.
func (c *FP8Cache) Close() error {
	for i := range c.layers {
		_ = c.layers[i].K.Free()
		_ = c.layers[i].V.Free()
		_ = c.layers[i].KScale.Free()
		_ = c.layers[i].VScale.Free()
	}
	c.layers = nil
	return nil
}

// PagedCache manages dynamic, non-contiguous physical KV memory blocks for multi-sequence continuous batching.
type PagedCache struct {
	mu sync.Mutex

	dev         *cuda.Device
	numLayers   int
	numKVHeads  int
	headDim     int
	blockSize   int
	totalBlocks int
	freeBlocks  []int
	blockTables map[int64][]int
	seqLens     map[int64]int
	kBlocks     []cudamem.Slice[float32]
	vBlocks     []cudamem.Slice[float32]
}

// Page holds the K and V storage for one layer and one physical slot.
// The slices are views into PagedCache storage and must not be freed.
type Page struct {
	K cudamem.Slice[float32]
	V cudamem.Slice[float32]
}

// NewPagedCache creates a paged KV cache allocator with totalBlocks physical slots of blockSize tokens.
func NewPagedCache(dev *cuda.Device, numLayers, numKVHeads, headDim, blockSize, totalBlocks int) (*PagedCache, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}
	if blockSize <= 0 || totalBlocks <= 0 {
		return nil, fmt.Errorf("cudakv: invalid block parameters (size=%d, total=%d)", blockSize, totalBlocks)
	}

	elementsPerLayer := totalBlocks * blockSize * numKVHeads * headDim
	kBlocks := make([]cudamem.Slice[float32], numLayers)
	vBlocks := make([]cudamem.Slice[float32], numLayers)

	for i := 0; i < numLayers; i++ {
		k, err := cudamem.Alloc[float32](dev, elementsPerLayer)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = kBlocks[j].Free()
				_ = vBlocks[j].Free()
			}
			return nil, fmt.Errorf("cudakv: alloc paged K layer %d: %w", i, err)
		}

		v, err := cudamem.Alloc[float32](dev, elementsPerLayer)
		if err != nil {
			_ = k.Free()
			for j := 0; j < i; j++ {
				_ = kBlocks[j].Free()
				_ = vBlocks[j].Free()
			}
			return nil, fmt.Errorf("cudakv: alloc paged V layer %d: %w", i, err)
		}

		kBlocks[i] = k
		vBlocks[i] = v
	}

	free := make([]int, totalBlocks)
	for i := 0; i < totalBlocks; i++ {
		free[i] = i
	}

	return &PagedCache{
		dev:         dev,
		numLayers:   numLayers,
		numKVHeads:  numKVHeads,
		headDim:     headDim,
		blockSize:   blockSize,
		totalBlocks: totalBlocks,
		freeBlocks:  free,
		blockTables: make(map[int64][]int),
		seqLens:     make(map[int64]int),
		kBlocks:     kBlocks,
		vBlocks:     vBlocks,
	}, nil
}

// AllocSequence allocates initial block storage for a new incoming sequence.
func (pc *PagedCache) AllocSequence(seqID int64, promptTokens int) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	if _, exists := pc.blockTables[seqID]; exists {
		return fmt.Errorf("cudakv: sequence %d already allocated", seqID)
	}

	neededBlocks := (promptTokens + pc.blockSize - 1) / pc.blockSize
	if len(pc.freeBlocks) < neededBlocks {
		return fmt.Errorf("cudakv: out of physical KV blocks (needed %d, available %d)", neededBlocks, len(pc.freeBlocks))
	}

	blocks := make([]int, neededBlocks)
	for i := 0; i < neededBlocks; i++ {
		blocks[i] = pc.freeBlocks[len(pc.freeBlocks)-1]
		pc.freeBlocks = pc.freeBlocks[:len(pc.freeBlocks)-1]
	}

	pc.blockTables[seqID] = blocks
	pc.seqLens[seqID] = promptTokens
	return nil
}

// AppendToken reserves physical memory for an additional decoded token in sequence.
func (pc *PagedCache) AppendToken(seqID int64) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	blocks, exists := pc.blockTables[seqID]
	if !exists {
		return fmt.Errorf("cudakv: sequence %d not found", seqID)
	}

	currentLen := pc.seqLens[seqID]
	if currentLen%pc.blockSize == 0 {
		// Needs a new block
		if len(pc.freeBlocks) == 0 {
			return fmt.Errorf("cudakv: out of KV blocks during generation")
		}
		newBlock := pc.freeBlocks[len(pc.freeBlocks)-1]
		pc.freeBlocks = pc.freeBlocks[:len(pc.freeBlocks)-1]
		pc.blockTables[seqID] = append(blocks, newBlock)
	}

	pc.seqLens[seqID]++
	return nil
}

// FreeSequence recycles physical memory blocks for a completed sequence.
func (pc *PagedCache) FreeSequence(seqID int64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	if blocks, exists := pc.blockTables[seqID]; exists {
		pc.freeBlocks = append(pc.freeBlocks, blocks...)
		delete(pc.blockTables, seqID)
		delete(pc.seqLens, seqID)
	}
}

// NumFreeBlocks returns the count of currently unallocated physical blocks.
func (pc *PagedCache) NumFreeBlocks() int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return len(pc.freeBlocks)
}

// BlockTable returns a copy of the block table for sequence seqID.
func (pc *PagedCache) BlockTable(seqID int64) []int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	res := make([]int, len(pc.blockTables[seqID]))
	copy(res, pc.blockTables[seqID])
	return res
}

// Page returns the storage for layer and physical slot. The returned views are
// valid until Close and must not be freed.
func (pc *PagedCache) Page(layer, slot int) (Page, error) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if layer < 0 || layer >= pc.numLayers {
		return Page{}, fmt.Errorf("cudakv: layer index %d out of bounds (total %d)", layer, pc.numLayers)
	}
	if slot < 0 || slot >= pc.totalBlocks {
		return Page{}, fmt.Errorf("cudakv: block index %d out of bounds (total %d)", slot, pc.totalBlocks)
	}
	elements := pc.blockSize * pc.numKVHeads * pc.headDim
	start := slot * elements
	return Page{K: pc.kBlocks[layer].Slice(start, start+elements), V: pc.vBlocks[layer].Slice(start, start+elements)}, nil
}

// Close deallocates all physical GPU memory buffers.
func (pc *PagedCache) Close() error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	for i := range pc.kBlocks {
		_ = pc.kBlocks[i].Free()
		_ = pc.vBlocks[i].Free()
	}
	pc.kBlocks = nil
	pc.vBlocks = nil
	pc.blockTables = nil
	pc.freeBlocks = nil
	return nil
}
