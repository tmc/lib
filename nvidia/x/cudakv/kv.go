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
	closed     bool
	freed      int // Number of K/V buffer releases completed in layer order.
}

// New creates a new static KV-cache allocating per-layer K and V buffers on dev.
// A non-nil error-result cache must be closed before closing dev.
func New(dev *cuda.Device, numLayers, numKVHeads, headDim, maxSeqLen int) (*Cache, error) {
	return NewTracked(dev, nil, numLayers, numKVHeads, headDim, maxSeqLen)
}

// NewTracked creates a cache whose device allocations are accounted by tracker.
// A non-nil error-result cache is closed to use but retains buffers for Close.
func NewTracked(dev *cuda.Device, tracker *cudamem.Tracker, numLayers, numKVHeads, headDim, maxSeqLen int) (*Cache, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf("cudakv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d)",
			numLayers, numKVHeads, headDim, maxSeqLen)
	}

	if err := validatePagedDimensions(numLayers, numKVHeads, headDim, maxSeqLen, 1); err != nil {
		return nil, err
	}
	return newStaticCache(dev, numLayers, numKVHeads, headDim, maxSeqLen, func(n int) (cudamem.Slice[float32], error) { return cudamem.AllocTracked[float32](dev, tracker, n) })
}

func newStaticCache(dev *cuda.Device, numLayers, numKVHeads, headDim, maxSeqLen int, allocate func(int) (cudamem.Slice[float32], error)) (*Cache, error) {
	c := &Cache{dev: dev, numLayers: numLayers, numKVHeads: numKVHeads, headDim: headDim, maxSeqLen: maxSeqLen, layers: make([]LayerCache, numLayers)}
	elements := maxSeqLen * numKVHeads * headDim
	for i := range c.layers {
		c.layers[i].LayerIndex = i
		for _, buffer := range []struct {
			name string
			dst  *cudamem.Slice[float32]
		}{{"K", &c.layers[i].K}, {"V", &c.layers[i].V}} {
			var err error
			*buffer.dst, err = allocate(elements)
			if err != nil {
				c.closed = true
				return c, fmt.Errorf("cudakv: alloc layer %d %s buffer: %w", i, buffer.name, err)
			}
		}
	}
	return c, nil
}

// Layer returns the KV buffers for the specified layer index.
func (c *Cache) Layer(idx int) (LayerCache, error) {
	if c == nil || c.closed {
		return LayerCache{}, fmt.Errorf("cudakv: cache closed")
	}
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
	if c.closed || c.layers == nil {
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
// A failed release retains the remaining buffers for retry. Further use is refused.
func (c *Cache) Close() error {
	return c.closeWith(func(s cudamem.Slice[float32]) error { return s.Free() })
}

func (c *Cache) closeWith(free func(cudamem.Slice[float32]) error) error {
	if c == nil {
		return nil
	}
	c.closed = true
	for c.freed < len(c.layers)*2 {
		layer := &c.layers[c.freed/2]
		buffer := &layer.K
		if c.freed%2 != 0 {
			buffer = &layer.V
		}
		if err := free(*buffer); err != nil {
			return fmt.Errorf("cudakv: free buffer %d: %w", c.freed, err)
		}
		*buffer = cudamem.Slice[float32]{}
		c.freed++
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
	closed     bool
	freed      int
}

// NewFP8 creates a new static FP8 KV-cache allocating per-layer 8-bit K and V buffers on dev.
// A non-nil error-result cache must be closed before closing dev.
func NewFP8(dev *cuda.Device, numLayers, numKVHeads, headDim, maxSeqLen int) (*FP8Cache, error) {
	return NewFP8Tracked(dev, nil, numLayers, numKVHeads, headDim, maxSeqLen)
}

// NewFP8Tracked creates an FP8 cache whose device allocations, the K and V
// buffers and the per-head scales, are accounted by tracker.
//
// The scales are left uninitialized: the caller decides the quantization
// and writes one factor per KV head into KScale and VScale of every layer
// before storing into the cache.
// A non-nil error-result cache retains cleanup ownership and must be closed.
func NewFP8Tracked(dev *cuda.Device, tracker *cudamem.Tracker, numLayers, numKVHeads, headDim, maxSeqLen int) (*FP8Cache, error) {
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}
	if numLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf("cudakv: invalid dimensions (layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d)",
			numLayers, numKVHeads, headDim, maxSeqLen)
	}

	elementsPerLayer, err := fp8Elements(numLayers, numKVHeads, headDim, maxSeqLen)
	if err != nil {
		return nil, err
	}
	return newFP8Cache(dev, numLayers, numKVHeads, headDim, maxSeqLen, elementsPerLayer,
		func(n int) (cudamem.Slice[uint8], error) { return cudamem.AllocTracked[uint8](dev, tracker, n) },
		func(n int) (cudamem.Slice[float32], error) { return cudamem.AllocTracked[float32](dev, tracker, n) })
}

func fp8Elements(layers, heads, dim, length int) (int, error) {
	maxInt := int(^uint(0) >> 1)
	if layers <= 0 || heads <= 0 || dim <= 0 || length <= 0 {
		return 0, fmt.Errorf("cudakv: invalid FP8 dimensions")
	}
	elements := length
	for _, factor := range []int{heads, dim} {
		if elements > maxInt/factor {
			return 0, fmt.Errorf("cudakv: FP8 element count overflows int")
		}
		elements *= factor
	}
	if heads > maxInt/8 || elements > (maxInt-heads*8)/2 {
		return 0, fmt.Errorf("cudakv: FP8 layer bytes overflow int")
	}
	if layers > maxInt/(elements*2+heads*8) {
		return 0, fmt.Errorf("cudakv: FP8 cache bytes overflow int")
	}
	return elements, nil
}

func newFP8Cache(dev *cuda.Device, numLayers, numKVHeads, headDim, maxSeqLen, elementsPerLayer int,
	allocate func(int) (cudamem.Slice[uint8], error), scaleAllocate func(int) (cudamem.Slice[float32], error)) (*FP8Cache, error) {
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
			if *buf.dst, err = allocate(elementsPerLayer); err != nil {
				err = fmt.Errorf("cudakv: alloc layer %d FP8 %s buffer: %w", i, buf.name, err)
				break
			}
		}
		if err == nil {
			for _, scale := range []struct {
				name string
				dst  *cudamem.Slice[float32]
			}{{"K", &layer.KScale}, {"V", &layer.VScale}} {
				if *scale.dst, err = scaleAllocate(numKVHeads); err != nil {
					err = fmt.Errorf("cudakv: alloc layer %d FP8 %s scale: %w", i, scale.name, err)
					break
				}
			}
		}
		c.layers = append(c.layers, layer)
		if err != nil {
			c.closed = true
			return c, err
		}
	}
	return c, nil
}

// Layer returns the FP8 KV buffers for the specified layer index.
func (c *FP8Cache) Layer(idx int) (FP8LayerCache, error) {
	if c == nil || c.closed {
		return FP8LayerCache{}, fmt.Errorf("cudakv: FP8 cache closed")
	}
	if idx < 0 || idx >= len(c.layers) {
		return FP8LayerCache{}, fmt.Errorf("cudakv: layer index %d out of bounds (total %d)", idx, len(c.layers))
	}
	return c.layers[idx], nil
}

// Close releases all GPU memory buffers allocated for the FP8 cache.
// Failed cleanup retains the remaining buffers for retry and forbids further use.
func (c *FP8Cache) Close() error {
	return c.closeWith(func(s cudamem.Slice[uint8]) error { return s.Free() }, func(s cudamem.Slice[float32]) error { return s.Free() })
}

func (c *FP8Cache) closeWith(free func(cudamem.Slice[uint8]) error, freeScale func(cudamem.Slice[float32]) error) error {
	if c == nil {
		return nil
	}
	c.closed = true
	for c.freed < len(c.layers)*4 {
		layer := &c.layers[c.freed/4]
		if c.freed%4 < 2 {
			buffer := &layer.K
			if c.freed%4 == 1 {
				buffer = &layer.V
			}
			if err := free(*buffer); err != nil {
				return fmt.Errorf("cudakv: free FP8 buffer %d: %w", c.freed, err)
			}
			*buffer = cudamem.Slice[uint8]{}
		} else {
			buffer := &layer.KScale
			if c.freed%4 == 3 {
				buffer = &layer.VScale
			}
			if err := freeScale(*buffer); err != nil {
				return fmt.Errorf("cudakv: free FP8 scale %d: %w", c.freed, err)
			}
			*buffer = cudamem.Slice[float32]{}
		}
		c.freed++
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
	mu     sync.Mutex
	closed bool
	freed  int

	dev         *cuda.Device
	tracker     *cudamem.Tracker
	numLayers   int
	numKVHeads  int
	headDim     int
	blockSize   int
	totalBlocks int
	blocks      *BlockAllocator
	kBlocks     []cudamem.Slice[float32]
	vBlocks     []cudamem.Slice[float32]

	hostK [][]float32
	hostV [][]float32
}

// Page holds the K and V storage for one layer and one physical slot.
// The slices are views into PagedCache storage and must not be freed.
type Page struct {
	K cudamem.Slice[float32]
	V cudamem.Slice[float32]
}

// NewPagedCache creates a paged KV cache allocator with totalBlocks physical slots of blockSize tokens.
// A non-nil error-result cache must be closed before closing dev.
func NewPagedCache(dev *cuda.Device, numLayers, numKVHeads, headDim, blockSize, totalBlocks int) (*PagedCache, error) {
	return NewPagedCacheTracked(dev, nil, numLayers, numKVHeads, headDim, blockSize, totalBlocks)
}

// NewPagedCacheTracked creates a paged cache whose pool is accounted by tracker.
// A non-nil error-result cache retains partial allocations for Close.
func NewPagedCacheTracked(dev *cuda.Device, tracker *cudamem.Tracker, numLayers, numKVHeads, headDim, blockSize, totalBlocks int) (*PagedCache, error) {
	if err := validatePagedDimensions(numLayers, numKVHeads, headDim, blockSize, totalBlocks); err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("cudakv: nil device")
	}

	return newPagedCache(dev, tracker, numLayers, numKVHeads, headDim, blockSize, totalBlocks, func(n int) (cudamem.Slice[float32], error) { return cudamem.AllocTracked[float32](dev, tracker, n) })
}

func newPagedCache(dev *cuda.Device, tracker *cudamem.Tracker, numLayers, numKVHeads, headDim, blockSize, totalBlocks int, allocate func(int) (cudamem.Slice[float32], error)) (*PagedCache, error) {
	pc := &PagedCache{dev: dev, tracker: tracker, numLayers: numLayers, numKVHeads: numKVHeads, headDim: headDim, blockSize: blockSize, totalBlocks: totalBlocks,
		blocks: NewBlockAllocator(blockSize, totalBlocks), kBlocks: make([]cudamem.Slice[float32], numLayers), vBlocks: make([]cudamem.Slice[float32], numLayers)}
	elements := totalBlocks * blockSize * numKVHeads * headDim
	for i := 0; i < numLayers; i++ {
		for _, buffer := range []struct {
			name string
			dst  *cudamem.Slice[float32]
		}{{"K", &pc.kBlocks[i]}, {"V", &pc.vBlocks[i]}} {
			var err error
			*buffer.dst, err = allocate(elements)
			if err != nil {
				pc.closed = true
				pc.blocks = newBlockAllocator(blockSize, 0)
				return pc, fmt.Errorf("cudakv: alloc paged %s layer %d: %w", buffer.name, i, err)
			}
		}
	}
	return pc, nil
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
	if pc.closed {
		return fmt.Errorf("cudakv: paged cache closed")
	}
	return pc.blocks.allocSequence(seqID, prefix, promptTokens)
}

// AppendToken reserves physical memory for an additional decoded token in sequence.
func (pc *PagedCache) AppendToken(seqID int64) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed {
		return fmt.Errorf("cudakv: paged cache closed")
	}
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

// Table returns a copy of the block table for sequence seqID.
func (pc *PagedCache) Table(seqID int64) []int {
	return pc.BlockTable(seqID)
}

// BlockSize returns the token capacity per block.
func (pc *PagedCache) BlockSize() int {
	return pc.blockSize
}

// Page returns the storage for layer and physical slot. The returned views are
// valid until Close and must not be freed.
func (pc *PagedCache) Page(layer, slot int) (Page, error) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed {
		return Page{}, fmt.Errorf("cudakv: paged cache closed")
	}
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
// The caller must first complete all GPU use of pages. Failed cleanup retains
// the remaining buffers for retry; page access and allocation remain closed.
func (pc *PagedCache) Close() error {
	return pc.closeWith(func(s cudamem.Slice[float32]) error { return s.Free() })
}

func (pc *PagedCache) closeWith(free func(cudamem.Slice[float32]) error) error {
	if pc == nil {
		return nil
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	pc.closed = true
	pc.blocks = newBlockAllocator(pc.blockSize, 0)
	pc.hostK, pc.hostV = nil, nil
	for pc.freed < len(pc.kBlocks)*2 {
		buffer := &pc.kBlocks[pc.freed/2]
		if pc.freed%2 != 0 {
			buffer = &pc.vBlocks[pc.freed/2]
		}
		if err := free(*buffer); err != nil {
			return fmt.Errorf("cudakv: free paged buffer %d: %w", pc.freed, err)
		}
		*buffer = cudamem.Slice[float32]{}
		pc.freed++
	}
	pc.kBlocks = nil
	pc.vBlocks = nil
	pc.blocks = newBlockAllocator(pc.blockSize, 0)
	return nil
}

// NewTestPagedCache creates a paged KV cache backed by host memory for testing
// allocation, token retention, and compaction without a CUDA GPU.
func NewTestPagedCache(numLayers, numKVHeads, headDim, blockSize, totalBlocks int) (*PagedCache, error) {
	if err := validatePagedDimensions(numLayers, numKVHeads, headDim, blockSize, totalBlocks); err != nil {
		return nil, err
	}
	elementsPerLayer := totalBlocks * blockSize * numKVHeads * headDim
	hostK := make([][]float32, numLayers)
	hostV := make([][]float32, numLayers)
	for i := 0; i < numLayers; i++ {
		hostK[i] = make([]float32, elementsPerLayer)
		hostV[i] = make([]float32, elementsPerLayer)
	}
	return &PagedCache{
		numLayers:   numLayers,
		numKVHeads:  numKVHeads,
		headDim:     headDim,
		blockSize:   blockSize,
		totalBlocks: totalBlocks,
		blocks:      NewBlockAllocator(blockSize, totalBlocks),
		hostK:       hostK,
		hostV:       hostV,
	}, nil
}

// Tracker returns the memory tracker associated with this cache, or nil.
func (pc *PagedCache) Tracker() *cudamem.Tracker {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.tracker
}

// FragmentationStats returns current pool fragmentation statistics.
func (pc *PagedCache) FragmentationStats() FragmentationStats {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.FragmentationStats()
}

// FragmentationIndex returns the current fragmentation ratio (0.0 to 1.0).
func (pc *PagedCache) FragmentationIndex() float64 {
	return pc.FragmentationStats().FragmentationIndex
}

// MemoryFragmentation reports fragmentation stats in bytes, connecting the block
// pool metrics with cudamem.Tracker accounting.
func (pc *PagedCache) MemoryFragmentation() (fragIndex float64, totalFreeBytes, largestFreeBytes uint64) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	stats := pc.blocks.FragmentationStats()
	bytesPerBlock := uint64(pc.blockSize * pc.numKVHeads * pc.headDim * 4 * pc.numLayers * 2) // K and V
	totalFreeBytes = uint64(stats.FreeBlocks) * bytesPerBlock
	largestFreeBytes = uint64(stats.LargestContiguous) * bytesPerBlock
	return stats.FragmentationIndex, totalFreeBytes, largestFreeBytes
}

// VirtualPageTable returns the virtual-to-physical block table for seqID.
func (pc *PagedCache) VirtualPageTable(seqID int64) VirtualPageTable {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return VirtualPageTable{
		SeqID:  seqID,
		Blocks: pc.blocks.Table(seqID),
	}
}

// VirtualPageTables returns virtual page tables for all sequences.
func (pc *PagedCache) VirtualPageTables() []VirtualPageTable {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.blocks.VirtualPageTables()
}

// Defrag performs proactive compaction on pc according to policy.
func (pc *PagedCache) Defrag(policy ...DefragPolicy) (CompactionPlan, error) {
	p := DefaultDefragPolicy()
	if len(policy) > 0 {
		p = policy[0]
	}
	c := NewCompactor(p)
	return c.Compact(pc)
}

// WriteToken stores K and V activations for tokenIdx of seqID in layer.
// This is used for host-backed testing or CPU verification.
func (pc *PagedCache) WriteToken(seqID int64, tokenIdx int, layer int, k, v []float32) error {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if layer < 0 || layer >= pc.numLayers {
		return fmt.Errorf("%s: layer index %d out of bounds (total %d)", pkgName, layer, pc.numLayers)
	}
	table, exists := pc.blocks.tables[seqID]
	if !exists {
		return fmt.Errorf("%s: sequence %d not found", pkgName, seqID)
	}
	blockIdx := tokenIdx / pc.blockSize
	if blockIdx >= len(table) {
		return fmt.Errorf("%s: token index %d exceeds allocated blocks (count %d)", pkgName, tokenIdx, len(table))
	}
	physBlock := table[blockIdx]
	offsetInBlock := tokenIdx % pc.blockSize
	elemPerToken := pc.numKVHeads * pc.headDim
	if len(k) != elemPerToken || len(v) != elemPerToken {
		return fmt.Errorf("%s: token dimension mismatch: got k=%d v=%d, want %d", pkgName, len(k), len(v), elemPerToken)
	}
	if len(pc.hostK) > layer {
		blockElem := pc.blockSize * elemPerToken
		start := physBlock*blockElem + offsetInBlock*elemPerToken
		copy(pc.hostK[layer][start:start+elemPerToken], k)
		copy(pc.hostV[layer][start:start+elemPerToken], v)
	}
	return nil
}

// ReadToken retrieves K and V activations for tokenIdx of seqID in layer.
func (pc *PagedCache) ReadToken(seqID int64, tokenIdx int, layer int) (k, v []float32, err error) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if layer < 0 || layer >= pc.numLayers {
		return nil, nil, fmt.Errorf("%s: layer index %d out of bounds (total %d)", pkgName, layer, pc.numLayers)
	}
	table, exists := pc.blocks.tables[seqID]
	if !exists {
		return nil, nil, fmt.Errorf("%s: sequence %d not found", pkgName, seqID)
	}
	blockIdx := tokenIdx / pc.blockSize
	if blockIdx >= len(table) {
		return nil, nil, fmt.Errorf("%s: token index %d exceeds allocated blocks (count %d)", pkgName, tokenIdx, len(table))
	}
	physBlock := table[blockIdx]
	offsetInBlock := tokenIdx % pc.blockSize
	elemPerToken := pc.numKVHeads * pc.headDim
	if len(pc.hostK) > layer {
		blockElem := pc.blockSize * elemPerToken
		start := physBlock*blockElem + offsetInBlock*elemPerToken
		k = append([]float32(nil), pc.hostK[layer][start:start+elemPerToken]...)
		v = append([]float32(nil), pc.hostV[layer][start:start+elemPerToken]...)
		return k, v, nil
	}
	return nil, nil, fmt.Errorf("%s: host token storage not available", pkgName)
}

func (pc *PagedCache) copyBlockAsync(srcSlot, dstSlot int, stream *cuda.Stream) error {
	if pc.closed {
		return fmt.Errorf("cudakv: paged cache closed")
	}
	if srcSlot < 0 || srcSlot >= pc.totalBlocks || dstSlot < 0 || dstSlot >= pc.totalBlocks {
		return fmt.Errorf("%s: block index out of bounds (src=%d, dst=%d, total=%d)", pkgName, srcSlot, dstSlot, pc.totalBlocks)
	}
	elements := pc.blockSize * pc.numKVHeads * pc.headDim

	// Copy host memory if present
	if len(pc.hostK) > 0 {
		for l := 0; l < pc.numLayers; l++ {
			srcStart := srcSlot * elements
			dstStart := dstSlot * elements
			copy(pc.hostK[l][dstStart:dstStart+elements], pc.hostK[l][srcStart:srcStart+elements])
			copy(pc.hostV[l][dstStart:dstStart+elements], pc.hostV[l][srcStart:srcStart+elements])
		}
	}

	// Copy GPU device memory if present
	if len(pc.kBlocks) > 0 && pc.dev != nil {
		for l := 0; l < pc.numLayers; l++ {
			srcStart := srcSlot * elements
			dstStart := dstSlot * elements
			srcK := pc.kBlocks[l].Slice(srcStart, srcStart+elements)
			dstK := pc.kBlocks[l].Slice(dstStart, dstStart+elements)
			srcV := pc.vBlocks[l].Slice(srcStart, srcStart+elements)
			dstV := pc.vBlocks[l].Slice(dstStart, dstStart+elements)

			if stream != nil {
				if err := dstK.CopyDeviceToDeviceAsync(srcK, stream); err != nil {
					return fmt.Errorf("%s: copy K layer %d: %w", pkgName, l, err)
				}
				if err := dstV.CopyDeviceToDeviceAsync(srcV, stream); err != nil {
					return fmt.Errorf("%s: copy V layer %d: %w", pkgName, l, err)
				}
			} else {
				if err := dstK.CopyDeviceToDevice(srcK); err != nil {
					return fmt.Errorf("%s: copy K layer %d: %w", pkgName, l, err)
				}
				if err := dstV.CopyDeviceToDevice(srcV); err != nil {
					return fmt.Errorf("%s: copy V layer %d: %w", pkgName, l, err)
				}
			}
		}
	}
	return nil
}
