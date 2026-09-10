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
func (c *Cache) Advance(n int) error {
	if c.layers == nil {
		return fmt.Errorf("cudakv: cache closed")
	}
	if n < 0 || n > c.maxSeqLen-c.pos {
		return fmt.Errorf("cudakv: advance %d from %d exceeds sequence capacity %d", n, c.pos, c.maxSeqLen)
	}
	c.pos += n
	return nil
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
}

// NewFP8 creates a new static FP8 KV-cache allocating per-layer 8-bit K and V buffers on dev.
func NewFP8(dev *cuda.Device, numLayers, numKVHeads, headDim, maxSeqLen int) (*FP8Cache, error) {
	return NewFP8Tracked(dev, nil, numLayers, numKVHeads, headDim, maxSeqLen)
}

// NewFP8Tracked creates an FP8 cache whose device allocations, the K and V
// buffers and the per-head scales, are accounted by tracker.
//
// The scales are left uninitialized: the caller decides the quantization
// and writes one factor per KV head into KScale and VScale of every layer
// before storing into the cache.
func NewFP8Tracked(dev *cuda.Device, tracker *cudamem.Tracker, numLayers, numKVHeads, headDim, maxSeqLen int) (*FP8Cache, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf("cudakv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d)",
			numLayers, numKVHeads, headDim, maxSeqLen)
	}

	elementsPerLayer := maxSeqLen * numKVHeads * headDim
	c := &FP8Cache{
		dev:        dev,
		numLayers:  numLayers,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		maxSeqLen:  maxSeqLen,
	}
	for i := 0; i < numLayers; i++ {
		layer := FP8LayerCache{LayerIndex: i}
		var err error
		for _, buf := range []struct {
			name string
			dst  *cudamem.Slice[uint8]
		}{{"K", &layer.K}, {"V", &layer.V}} {
			if *buf.dst, err = cudamem.AllocTracked[uint8](dev, tracker, elementsPerLayer); err != nil {
				err = fmt.Errorf("cudakv: alloc layer %d FP8 %s buffer: %w", i, buf.name, err)
				break
			}
		}
		if err == nil {
			for _, scale := range []struct {
				name string
				dst  *cudamem.Slice[float32]
			}{{"K", &layer.KScale}, {"V", &layer.VScale}} {
				if *scale.dst, err = cudamem.AllocTracked[float32](dev, tracker, numKVHeads); err != nil {
					err = fmt.Errorf("cudakv: alloc layer %d FP8 %s scale: %w", i, scale.name, err)
					break
				}
			}
		}
		c.layers = append(c.layers, layer)
		if err != nil {
			_ = c.Close()
			return nil, err
		}
	}
	return c, nil
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

// PagedCache is a pool of fixed-size KV blocks shared by many sequences, each
// with its own block table, for continuous batching.
//
// A block is freed when its last holder lets go. A sequence holds the blocks
// in its table; a prefix cache keeps blocks past the end of the sequence that
// wrote them with Retain and gives them up with Release. AllocSequenceWithPrefix
// starts a sequence on blocks another holder already has. Nothing here reads or
// writes the blocks: that is the attention kernel's job, through Page.
type PagedCache struct {
	mu sync.Mutex

	dev         *cuda.Device
	numLayers   int
	numKVHeads  int
	headDim     int
	blockSize   int
	totalBlocks int
	blocks      *blockAllocator
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
	return NewPagedCacheTracked(dev, nil, numLayers, numKVHeads, headDim, blockSize, totalBlocks)
}

// NewPagedCacheTracked creates a paged cache whose pool is accounted by tracker.
func NewPagedCacheTracked(dev *cuda.Device, tracker *cudamem.Tracker, numLayers, numKVHeads, headDim, blockSize, totalBlocks int) (*PagedCache, error) {
	if err := validatePagedDimensions(numLayers, numKVHeads, headDim, blockSize, totalBlocks); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}

	elementsPerLayer := totalBlocks * blockSize * numKVHeads * headDim
	kBlocks := make([]cudamem.Slice[float32], numLayers)
	vBlocks := make([]cudamem.Slice[float32], numLayers)

	for i := 0; i < numLayers; i++ {
		k, err := cudamem.AllocTracked[float32](dev, tracker, elementsPerLayer)
		if err != nil {
			for j := 0; j < i; j++ {
				_ = kBlocks[j].Free()
				_ = vBlocks[j].Free()
			}
			return nil, fmt.Errorf("cudakv: alloc paged K layer %d: %w", i, err)
		}

		v, err := cudamem.AllocTracked[float32](dev, tracker, elementsPerLayer)
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

	return &PagedCache{
		dev:         dev,
		numLayers:   numLayers,
		numKVHeads:  numKVHeads,
		headDim:     headDim,
		blockSize:   blockSize,
		totalBlocks: totalBlocks,
		blocks:      newBlockAllocator(blockSize, totalBlocks),
		kBlocks:     kBlocks,
		vBlocks:     vBlocks,
	}, nil
}

func validatePagedDimensions(numLayers, numKVHeads, headDim, blockSize, totalBlocks int) error {
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || blockSize <= 0 || totalBlocks <= 0 {
		return fmt.Errorf("cudakv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, block_size=%d, total_blocks=%d)",
			numLayers, numKVHeads, headDim, blockSize, totalBlocks)
	}
	const (
		buffersPerLayer = 2 // K and V
		bytesPerElement = 4 // float32
	)
	maxInt := int(^uint(0) >> 1)
	totalBytes := totalBlocks
	for _, factor := range []int{blockSize, numKVHeads, headDim, numLayers, buffersPerLayer, bytesPerElement} {
		if totalBytes > maxInt/factor {
			return fmt.Errorf("cudakv: paged cache size overflows int")
		}
		totalBytes *= factor
	}
	return nil
}

// AllocSequence gives a new sequence enough blocks for promptTokens tokens.
func (pc *PagedCache) AllocSequence(seqID int64, promptTokens int) error {
	return pc.AllocSequenceWithPrefix(seqID, nil, promptTokens)
}

// AllocSequenceWithPrefix gives a new sequence a block table that begins with
// prefix, blocks another holder (a finished sequence kept by a prefix cache, or
// a live one) already has, and continues with fresh blocks up to promptTokens.
//
// The prefix blocks count as full: promptTokens must be at least
// len(prefix)*blockSize, and the sequence writes only from that position on.
// Every prefix block must currently be held; sharing a free block is an error,
// since nothing guarantees what it contains.
func (pc *PagedCache) AllocSequenceWithPrefix(seqID int64, prefix []int, promptTokens int) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.allocSequence(seqID, prefix, promptTokens)
}

// AppendToken reserves physical memory for an additional decoded token in sequence.
func (pc *PagedCache) AppendToken(seqID int64) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.appendToken(seqID)
}

// FreeSequence ends a sequence's hold on its blocks. Blocks nothing else holds
// return to the pool; blocks a prefix cache retained, or another sequence
// shares, stay allocated. Freeing an unknown sequence does nothing.
func (pc *PagedCache) FreeSequence(seqID int64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.blocks.freeSequence(seqID)
}

// Retain keeps blocks allocated after the sequence holding them is freed,
// until a matching Release. It is how a prefix cache takes ownership of the
// blocks a finished sequence wrote. Every block must currently be held.
func (pc *PagedCache) Retain(blocks []int) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.retain(blocks)
}

// Release undoes one Retain, or one share taken by AllocSequenceWithPrefix,
// returning to the pool any block left with no holder.
func (pc *PagedCache) Release(blocks []int) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.release(blocks)
}

// Holders reports how many sequences and retainers hold block. Zero means free.
func (pc *PagedCache) Holders(block int) (int, error) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if block < 0 || block >= len(pc.blocks.holders) {
		return 0, fmt.Errorf("cudakv: block index %d out of bounds (total %d)", block, len(pc.blocks.holders))
	}
	return pc.blocks.holders[block], nil
}

// NumFreeBlocks returns the count of currently unallocated physical blocks.
func (pc *PagedCache) NumFreeBlocks() int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return len(pc.blocks.free)
}

// BlockTable returns a copy of the block table for sequence seqID.
func (pc *PagedCache) BlockTable(seqID int64) []int {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.table(seqID)
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
	pc.blocks = newBlockAllocator(pc.blockSize, 0)
	return nil
}
