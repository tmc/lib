//go:build darwin

// Package metalprefix stores immutable, leased KV prefix blocks in Metal
// shared memory.
package metalprefix

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"time"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/mempool"
	"github.com/tmc/lib/lm/kvcache"
)

// Fence delays publication until preceding Metal writes have completed.
type Fence interface {
	Wait(context.Context) error
}

type completedFence struct{}

func (completedFence) Wait(context.Context) error { return nil }

// CompletedFence returns a fence for data whose producing command buffer has
// already completed. It is appropriate after WaitUntilCompleted, not merely
// after Commit.
func CompletedFence() Fence { return completedFence{} }

// ChannelFence returns a fence completed by closing done.
func ChannelFence(done <-chan struct{}) Fence { return channelFence{done} }

type channelFence struct{ done <-chan struct{} }

func (f channelFence) Wait(ctx context.Context) error {
	if f.done == nil {
		return errors.New("metalprefix: nil completion channel")
	}
	select {
	case <-f.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Config describes one cache and its physical Metal allocation limit.
type Config struct {
	Device      *metal.Device
	MaxBytes    int64
	MaxEntries  int
	BlockTokens int
	Layers      int
	KVHeads     int
	HeadDim     int
}

// Stats reports physical allocation, including undiscoverable blocks pinned
// by live leases.
type Stats struct {
	Entries, Blocks int
	Bytes           int64
	Hits, Misses    int64
	SavedTokens     int64
}

// Identity identifies one immutable layer block allocation.
type Identity struct {
	ID         kvcache.ID
	Generation kvcache.Generation
	Start      int
	End        int
	Layer      int
}

// BlockView describes one immutable Metal allocation pinned by a Lease. K and
// V begin at Start; callers must not interpret them as sequence-zero buffers.
// The buffers remain valid only until the Lease is released.
type BlockView struct {
	Identity
	K       apple.MTLBuffer
	V       apple.MTLBuffer
	KOffset uint
	VOffset uint
}

type block struct {
	owner kvcache.Lease
	id    kvcache.ID
	gen   kvcache.Generation
	start int
	end   int
	layer int
	k     mempool.Slice[float32]
	v     mempool.Slice[float32]
	refs  int
}

type blockRef struct {
	b     *block
	lease kvcache.Lease
}

type entry struct {
	tokens   []int32
	current  int32
	blocks   []blockRef
	lastUsed time.Time
}

// Cache owns immutable Metal shared-memory blocks. Metal unified memory is
// recorded as a residency domain, not modeled as a CUDA-style transfer tier.
type Cache struct {
	mu sync.Mutex

	cfg                 Config
	store               *kvcache.Store
	entries             map[string]*entry
	blocks              map[[2]uint64]*block
	bytes               int64
	hits, misses, saved int64
	closed              bool
	waitCommandBuffer   func(context.Context, apple.MTLCommandBuffer) error
}

// New returns an empty bounded cache.
func New(cfg Config) (*Cache, error) {
	if cfg.Device == nil || cfg.MaxBytes <= 0 || cfg.MaxEntries <= 0 ||
		cfg.BlockTokens <= 0 || cfg.Layers <= 0 || cfg.KVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil, errors.New("metalprefix: invalid configuration")
	}
	store, err := kvcache.NewStore(0)
	if err != nil {
		return nil, fmt.Errorf("metalprefix: create block store: %w", err)
	}
	return &Cache{
		cfg: cfg, store: store, entries: make(map[string]*entry), blocks: make(map[[2]uint64]*block),
		waitCommandBuffer: metal.WaitCommandBuffer,
	}, nil
}

func tokenKey(tokens []int32) string {
	b := make([]byte, len(tokens)*4)
	for i, token := range tokens {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(token))
	}
	return string(b)
}

func (c *Cache) valuesPerToken() int { return c.cfg.KVHeads * c.cfg.HeadDim }

func (c *Cache) validateData(tokens []int32, k, v [][]float32) error {
	if len(tokens) == 0 || len(k) != c.cfg.Layers || len(v) != c.cfg.Layers {
		return errors.New("metalprefix: invalid prefix shape")
	}
	want := len(tokens) * c.valuesPerToken()
	for layer := 0; layer < c.cfg.Layers; layer++ {
		if len(k[layer]) < want || len(v[layer]) < want {
			return fmt.Errorf("metalprefix: layer %d has incomplete KV data", layer)
		}
	}
	return nil
}

// Put publishes a prefix after fence completes. K and V use
// [layer][token][kv-head][head-dimension] order. Published blocks are immutable.
func (c *Cache) Put(ctx context.Context, tokens []int32, current int32, k, v [][]float32, fence Fence) error {
	if ctx == nil || fence == nil {
		return errors.New("metalprefix: nil context or fence")
	}
	if err := c.validateData(tokens, k, v); err != nil {
		return err
	}
	if err := fence.Wait(ctx); err != nil {
		return fmt.Errorf("metalprefix: wait for Metal completion: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return kvcache.ErrClosed
	}

	key := tokenKey(tokens)
	var base *entry
	baseTokens := 0
	for otherKey, candidate := range c.entries {
		if otherKey == key || len(candidate.tokens) >= len(tokens) {
			continue
		}
		if commonPrefix(candidate.tokens, tokens) == len(candidate.tokens) {
			n := len(candidate.tokens) / c.cfg.BlockTokens * c.cfg.BlockTokens
			if n > baseTokens {
				base, baseTokens = candidate, n
			}
		}
	}

	refs := make([]blockRef, 0, c.cfg.Layers*((len(tokens)+c.cfg.BlockTokens-1)/c.cfg.BlockTokens))
	rollback := func() {
		for i := len(refs) - 1; i >= 0; i-- {
			c.dropRefLocked(refs[i])
		}
	}
	if base != nil {
		for _, ref := range base.blocks {
			if ref.b.end > baseTokens {
				continue
			}
			shared, err := c.store.Share(ref.lease)
			if err != nil {
				rollback()
				return fmt.Errorf("metalprefix: share prefix block: %w", err)
			}
			ref.b.refs++
			refs = append(refs, blockRef{b: ref.b, lease: shared})
		}
	}

	newBytes := int64(0)
	for start := baseTokens; start < len(tokens); start += c.cfg.BlockTokens {
		end := min(start+c.cfg.BlockTokens, len(tokens))
		newBytes += int64(2*c.cfg.Layers*(end-start)*c.valuesPerToken()) * 4
	}
	newEntry := c.entries[key] == nil
	for ((newEntry && len(c.entries) >= c.cfg.MaxEntries) || c.bytes+newBytes > c.cfg.MaxBytes) && len(c.entries) != 0 {
		if !c.evictOldestLocked(key) {
			break
		}
	}
	if c.bytes+newBytes > c.cfg.MaxBytes || (newEntry && len(c.entries) >= c.cfg.MaxEntries) {
		rollback()
		return kvcache.ErrExhausted
	}

	for start := baseTokens; start < len(tokens); start += c.cfg.BlockTokens {
		end := min(start+c.cfg.BlockTokens, len(tokens))
		lo, hi := start*c.valuesPerToken(), end*c.valuesPerToken()
		for layer := 0; layer < c.cfg.Layers; layer++ {
			ref, err := c.allocBlockLocked(tokens, start, end, layer, k[layer][lo:hi], v[layer][lo:hi])
			if err != nil {
				rollback()
				return err
			}
			refs = append(refs, ref)
		}
	}

	if old := c.entries[key]; old != nil {
		delete(c.entries, key)
		c.dropEntryLocked(old)
	}
	c.entries[key] = &entry{tokens: append([]int32(nil), tokens...), current: current, blocks: refs, lastUsed: time.Now()}
	return nil
}

// PutBuffers publishes a prefix by copying directly between Metal buffers.
// sourceOffset is measured in float32 values and identifies token zero in
// every source layer. The source buffers remain the caller's responsibility
// until PutBuffers returns.
func (c *Cache) PutBuffers(ctx context.Context, tokens []int32, current int32, k, v []mempool.Slice[float32], sourceOffset int) error {
	if ctx == nil || len(tokens) == 0 || len(k) != c.cfg.Layers || len(v) != c.cfg.Layers || sourceOffset < 0 {
		return errors.New("metalprefix: invalid buffer prefix")
	}
	want := sourceOffset + len(tokens)*c.valuesPerToken()
	for layer := range k {
		if k[layer].Len() < want || v[layer].Len() < want {
			return fmt.Errorf("metalprefix: layer %d source buffer too small", layer)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return kvcache.ErrClosed
	}
	key := tokenKey(tokens)
	var base *entry
	baseTokens := 0
	for otherKey, candidate := range c.entries {
		if otherKey == key || len(candidate.tokens) >= len(tokens) {
			continue
		}
		if commonPrefix(candidate.tokens, tokens) == len(candidate.tokens) {
			n := len(candidate.tokens) / c.cfg.BlockTokens * c.cfg.BlockTokens
			if n > baseTokens {
				base, baseTokens = candidate, n
			}
		}
	}

	refs := make([]blockRef, 0, c.cfg.Layers*((len(tokens)+c.cfg.BlockTokens-1)/c.cfg.BlockTokens))
	rollback := func() {
		for i := len(refs) - 1; i >= 0; i-- {
			c.dropRefLocked(refs[i])
		}
	}
	if base != nil {
		for _, ref := range base.blocks {
			if ref.b.end > baseTokens {
				continue
			}
			shared, err := c.store.Share(ref.lease)
			if err != nil {
				rollback()
				return fmt.Errorf("metalprefix: share prefix block: %w", err)
			}
			ref.b.refs++
			refs = append(refs, blockRef{b: ref.b, lease: shared})
		}
	}

	newBytes := int64(0)
	for start := baseTokens; start < len(tokens); start += c.cfg.BlockTokens {
		end := min(start+c.cfg.BlockTokens, len(tokens))
		newBytes += int64(2*c.cfg.Layers*(end-start)*c.valuesPerToken()) * 4
	}
	newEntry := c.entries[key] == nil
	for ((newEntry && len(c.entries) >= c.cfg.MaxEntries) || c.bytes+newBytes > c.cfg.MaxBytes) && len(c.entries) != 0 {
		if !c.evictOldestLocked(key) {
			break
		}
	}
	if c.bytes+newBytes > c.cfg.MaxBytes || (newEntry && len(c.entries) >= c.cfg.MaxEntries) {
		rollback()
		return kvcache.ErrExhausted
	}

	cb := c.cfg.Device.Queue().CommandBuffer()
	blit := cb.BlitCommandEncoder()
	for start := baseTokens; start < len(tokens); start += c.cfg.BlockTokens {
		end := min(start+c.cfg.BlockTokens, len(tokens))
		values := (end - start) * c.valuesPerToken()
		sourceByteOffset := uint(sourceOffset+start*c.valuesPerToken()) * 4
		for layer := 0; layer < c.cfg.Layers; layer++ {
			ref, err := c.allocEmptyBlockLocked(tokens, start, end, layer)
			if err != nil {
				blit.EndEncoding()
				rollback()
				return err
			}
			refs = append(refs, ref)
			blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(k[layer].Buffer(), k[layer].Offset()+sourceByteOffset, ref.b.k.Buffer(), ref.b.k.Offset(), uint(values*4))
			blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(v[layer].Buffer(), v[layer].Offset()+sourceByteOffset, ref.b.v.Buffer(), ref.b.v.Offset(), uint(values*4))
		}
	}
	blit.EndEncoding()
	if err := ctx.Err(); err != nil {
		rollback()
		return err
	}
	cb.Commit()
	if err := c.waitCommandBuffer(ctx, cb); err != nil {
		rollback()
		return fmt.Errorf("metalprefix: copy prefix buffers: %w", err)
	}
	if old := c.entries[key]; old != nil {
		delete(c.entries, key)
		c.dropEntryLocked(old)
	}
	c.entries[key] = &entry{tokens: append([]int32(nil), tokens...), current: current, blocks: refs, lastUsed: time.Now()}
	return nil
}

// ExtendBuffers publishes tokens by sharing complete blocks from base and
// copying the remaining base fragment and suffix directly between Metal
// buffers. The source buffers need contain only suffix positions at and after
// base.Tokens; earlier positions are never read.
func (c *Cache) ExtendBuffers(ctx context.Context, base *Lease, tokens []int32, current int32, k, v []mempool.Slice[float32], sourceOffset int) error {
	if ctx == nil || base == nil || base.cache != c || len(k) != c.cfg.Layers || len(v) != c.cfg.Layers || sourceOffset < 0 {
		return errors.New("metalprefix: invalid buffer extension")
	}
	baseTokens := base.tokens
	if baseTokens <= 0 || len(tokens) <= baseTokens {
		return errors.New("metalprefix: extension does not continue base")
	}
	want := sourceOffset + len(tokens)*c.valuesPerToken()
	for layer := range k {
		if k[layer].Len() < want || v[layer].Len() < want {
			return fmt.Errorf("metalprefix: layer %d extension buffer too small", layer)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return kvcache.ErrClosed
	}
	if len(base.refs) == 0 {
		return kvcache.ErrStale
	}
	if len(base.keyTokens) != baseTokens || commonPrefix(base.keyTokens, tokens) != baseTokens {
		return errors.New("metalprefix: extension does not continue base")
	}
	key := tokenKey(tokens)
	if c.entries[key] != nil {
		return nil
	}
	sharedTokens := baseTokens / c.cfg.BlockTokens * c.cfg.BlockTokens
	refs := make([]blockRef, 0, c.cfg.Layers*((len(tokens)+c.cfg.BlockTokens-1)/c.cfg.BlockTokens))
	rollback := func() {
		for i := len(refs) - 1; i >= 0; i-- {
			c.dropRefLocked(refs[i])
		}
	}
	for _, ref := range base.refs {
		if ref.b.end > sharedTokens {
			continue
		}
		shared, err := c.store.Share(ref.lease)
		if err != nil {
			rollback()
			return kvcache.ErrStale
		}
		ref.b.refs++
		refs = append(refs, blockRef{b: ref.b, lease: shared})
	}

	newBytes := int64(0)
	for start := sharedTokens; start < len(tokens); start += c.cfg.BlockTokens {
		end := min(start+c.cfg.BlockTokens, len(tokens))
		newBytes += int64(2*c.cfg.Layers*(end-start)*c.valuesPerToken()) * 4
	}
	for (len(c.entries) >= c.cfg.MaxEntries || c.bytes+newBytes > c.cfg.MaxBytes) && len(c.entries) != 0 {
		if !c.evictOldestLocked(key) {
			break
		}
	}
	if c.bytes+newBytes > c.cfg.MaxBytes || len(c.entries) >= c.cfg.MaxEntries {
		rollback()
		return kvcache.ErrExhausted
	}

	valuesPerToken := c.valuesPerToken()
	cb := c.cfg.Device.Queue().CommandBuffer()
	blit := cb.BlitCommandEncoder()
	for start := sharedTokens; start < len(tokens); start += c.cfg.BlockTokens {
		end := min(start+c.cfg.BlockTokens, len(tokens))
		oldEnd := min(end, baseTokens)
		for layer := 0; layer < c.cfg.Layers; layer++ {
			ref, err := c.allocEmptyBlockLocked(tokens, start, end, layer)
			if err != nil {
				blit.EndEncoding()
				rollback()
				return err
			}
			refs = append(refs, ref)
			if oldEnd > start {
				old := base.blockLocked(layer, start, oldEnd)
				if old == nil {
					blit.EndEncoding()
					rollback()
					return kvcache.ErrStale
				}
				bytes := uint((oldEnd - start) * valuesPerToken * 4)
				blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(old.k.Buffer(), old.k.Offset()+uint((start-old.start)*valuesPerToken*4), ref.b.k.Buffer(), ref.b.k.Offset(), bytes)
				blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(old.v.Buffer(), old.v.Offset()+uint((start-old.start)*valuesPerToken*4), ref.b.v.Buffer(), ref.b.v.Offset(), bytes)
			}
			suffixStart := max(start, baseTokens)
			if suffixStart < end {
				bytes := uint((end - suffixStart) * valuesPerToken * 4)
				src := uint(sourceOffset+suffixStart*valuesPerToken) * 4
				dst := ref.b.k.Offset() + uint((suffixStart-start)*valuesPerToken*4)
				blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(k[layer].Buffer(), k[layer].Offset()+src, ref.b.k.Buffer(), dst, bytes)
				dst = ref.b.v.Offset() + uint((suffixStart-start)*valuesPerToken*4)
				blit.CopyFromBufferSourceOffsetToBufferDestinationOffsetSize(v[layer].Buffer(), v[layer].Offset()+src, ref.b.v.Buffer(), dst, bytes)
			}
		}
	}
	blit.EndEncoding()
	if err := ctx.Err(); err != nil {
		rollback()
		return err
	}
	cb.Commit()
	if err := c.waitCommandBuffer(ctx, cb); err != nil {
		rollback()
		return fmt.Errorf("metalprefix: copy buffer extension: %w", err)
	}
	c.entries[key] = &entry{tokens: append([]int32(nil), tokens...), current: current, blocks: refs, lastUsed: time.Now()}
	return nil
}

func commonPrefix(a, b []int32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func (c *Cache) allocBlockLocked(tokens []int32, start, end, layer int, k, v []float32) (blockRef, error) {
	ref, err := c.allocEmptyBlockLocked(tokens, start, end, layer)
	if err != nil {
		return blockRef{}, err
	}
	copy(ref.b.k.HostSlice(), k)
	copy(ref.b.v.HostSlice(), v)
	return ref, nil
}

func (c *Cache) allocEmptyBlockLocked(tokens []int32, start, end, layer int) (blockRef, error) {
	values := (end - start) * c.valuesPerToken()
	bytes := int64(2*values) * 4
	if c.bytes+bytes > c.cfg.MaxBytes {
		return blockRef{}, kvcache.ErrExhausted
	}
	ks, err := mempool.Alloc[float32](c.cfg.Device, values)
	if err != nil {
		return blockRef{}, fmt.Errorf("metalprefix: allocate K block: %w", err)
	}
	vs, err := mempool.Alloc[float32](c.cfg.Device, values)
	if err != nil {
		ks.Free()
		return blockRef{}, fmt.Errorf("metalprefix: allocate V block: %w", err)
	}
	owner, err := c.store.Allocate(kvcache.Spec{
		Owner:  tokenKey(tokens[:end]),
		Tokens: kvcache.Range{Start: start, End: end},
		Format: kvcache.Format{Layer: layer, DType: "float32", Layout: "token-kvhead-head", KVHeads: c.cfg.KVHeads, HeadDim: c.cfg.HeadDim},
	}, kvcache.Residency{Backend: "metal", Tier: "unified", Domain: c.cfg.Device.Name()})
	if err != nil {
		ks.Free()
		vs.Free()
		return blockRef{}, fmt.Errorf("metalprefix: record block: %w", err)
	}
	id, gen := owner.ID()
	b := &block{owner: owner, id: id, gen: gen, start: start, end: end, layer: layer, k: ks, v: vs, refs: 1}
	c.blocks[[2]uint64{uint64(id), uint64(gen)}] = b
	c.bytes += bytes
	return blockRef{b: b, lease: owner}, nil
}

// Lease pins every physical block needed for one prefix. Its zero value is invalid.
type Lease struct {
	cache     *Cache
	tokens    int
	current   int32
	keyTokens []int32
	refs      []blockRef
	once      sync.Once
}

// Acquire returns the longest cached prefix of tokens. It either acquires all
// constituent blocks or rolls back every newly acquired block lease.
func (c *Cache) Acquire(tokens []int32) (*Lease, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, false, kvcache.ErrClosed
	}
	var found *entry
	for _, candidate := range c.entries {
		if len(candidate.tokens) <= len(tokens) && commonPrefix(candidate.tokens, tokens) == len(candidate.tokens) &&
			(found == nil || len(candidate.tokens) > len(found.tokens)) {
			found = candidate
		}
	}
	if found == nil {
		c.misses++
		return nil, false, nil
	}
	lease, err := c.acquireEntryLocked(found)
	if err != nil {
		return nil, false, err
	}
	c.hits++
	c.saved += int64(len(found.tokens))
	return lease, true, nil
}

// AcquireExact acquires tokens without recording a cache lookup. It is used by
// a producer to retain the entry it just published.
func (c *Cache) AcquireExact(tokens []int32) (*Lease, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, kvcache.ErrClosed
	}
	found := c.entries[tokenKey(tokens)]
	if found == nil {
		return nil, kvcache.ErrStale
	}
	return c.acquireEntryLocked(found)
}

func (c *Cache) acquireEntryLocked(found *entry) (*Lease, error) {
	refs := make([]blockRef, 0, len(found.blocks))
	for _, ref := range found.blocks {
		shared, err := c.store.Share(ref.lease)
		if err != nil {
			for i := len(refs) - 1; i >= 0; i-- {
				c.dropRefLocked(refs[i])
			}
			return nil, fmt.Errorf("metalprefix: acquire prefix: %w", err)
		}
		ref.b.refs++
		refs = append(refs, blockRef{b: ref.b, lease: shared})
	}
	found.lastUsed = time.Now()
	return &Lease{cache: c, tokens: len(found.tokens), current: found.current, keyTokens: append([]int32(nil), found.tokens...), refs: refs}, nil
}

func (l *Lease) blockLocked(layer, start, end int) *block {
	for _, ref := range l.refs {
		if ref.b.layer == layer && ref.b.start <= start && ref.b.end >= end {
			return ref.b
		}
	}
	return nil
}

// Tokens returns the matched token count.
func (l *Lease) Tokens() int {
	if l == nil {
		return 0
	}
	return l.tokens
}

// Current returns the token produced after the cached prefix.
func (l *Lease) Current() int32 {
	if l == nil {
		return 0
	}
	return l.current
}

// Blocks returns the identities and token ranges pinned by the lease.
func (l *Lease) Blocks() []Identity {
	if l == nil {
		return nil
	}
	ids := make([]Identity, len(l.refs))
	for i, ref := range l.refs {
		ids[i] = Identity{ID: ref.b.id, Generation: ref.b.gen, Start: ref.b.start, End: ref.b.end, Layer: ref.b.layer}
	}
	return ids
}

// ViewLease pins one layer's physical block-table entries independently of the
// prefix lease from which it was acquired.
type ViewLease struct {
	cache *Cache
	refs  []blockRef
	views []BlockView
	once  sync.Once
}

// PinLayer returns the block-table entries for layer in token order. It
// acquires independent references so the returned buffers remain live even if
// the parent Lease is released concurrently. The caller must release the
// returned ViewLease after the last Metal command using its buffers completes.
func (l *Lease) PinLayer(layer int) (*ViewLease, error) {
	if l == nil || l.cache == nil || layer < 0 || layer >= l.cache.cfg.Layers {
		return nil, errors.New("metalprefix: invalid layer view")
	}
	l.cache.mu.Lock()
	defer l.cache.mu.Unlock()
	if len(l.refs) == 0 {
		return nil, kvcache.ErrStale
	}
	var refs []blockRef
	views := make([]BlockView, 0, (l.tokens+l.cache.cfg.BlockTokens-1)/l.cache.cfg.BlockTokens)
	for _, ref := range l.refs {
		if ref.b.layer != layer {
			continue
		}
		shared, err := l.cache.store.Share(ref.lease)
		if err != nil {
			for i := len(refs) - 1; i >= 0; i-- {
				l.cache.dropRefLocked(refs[i])
			}
			return nil, kvcache.ErrStale
		}
		ref.b.refs++
		refs = append(refs, blockRef{b: ref.b, lease: shared})
		views = append(views, BlockView{
			Identity: Identity{ID: ref.b.id, Generation: ref.b.gen, Start: ref.b.start, End: ref.b.end, Layer: ref.b.layer},
			K:        ref.b.k.Buffer(), V: ref.b.v.Buffer(), KOffset: ref.b.k.Offset(), VOffset: ref.b.v.Offset(),
		})
	}
	return &ViewLease{cache: l.cache, refs: refs, views: views}, nil
}

// Views returns a copy of the pinned block-table entries.
func (l *ViewLease) Views() []BlockView {
	if l == nil {
		return nil
	}
	return append([]BlockView(nil), l.views...)
}

// Release drops the independent layer pins. It is safe to call more than once.
func (l *ViewLease) Release() {
	if l == nil || l.cache == nil {
		return
	}
	l.once.Do(func() {
		l.cache.mu.Lock()
		defer l.cache.mu.Unlock()
		for i := len(l.refs) - 1; i >= 0; i-- {
			l.cache.dropRefLocked(l.refs[i])
		}
		l.refs = nil
		l.views = nil
	})
}

// Restore copies n cached tokens into contiguous destination KV buffers.
// The caller retains the lease until any GPU command reading dst has completed.
func (l *Lease) Restore(k, v [][]float32, n int) error {
	if l == nil || l.cache == nil || n <= 0 || n > l.tokens || len(k) != l.cache.cfg.Layers || len(v) != l.cache.cfg.Layers {
		return errors.New("metalprefix: invalid restore")
	}
	want := n * l.cache.valuesPerToken()
	for layer := range k {
		if len(k[layer]) < want || len(v[layer]) < want {
			return fmt.Errorf("metalprefix: layer %d restore destination too small", layer)
		}
	}
	for _, ref := range l.refs {
		b := ref.b
		if b.start >= n {
			continue
		}
		end := min(b.end, n)
		values := (end - b.start) * l.cache.valuesPerToken()
		off := b.start * l.cache.valuesPerToken()
		copy(k[b.layer][off:off+values], b.k.HostSlice()[:values])
		copy(v[b.layer][off:off+values], b.v.HostSlice()[:values])
	}
	return nil
}

// Release drops all block leases. It is safe to call more than once.
func (l *Lease) Release() {
	if l == nil || l.cache == nil {
		return
	}
	l.once.Do(func() {
		l.cache.mu.Lock()
		defer l.cache.mu.Unlock()
		for i := len(l.refs) - 1; i >= 0; i-- {
			l.cache.dropRefLocked(l.refs[i])
		}
		l.refs = nil
		l.keyTokens = nil
	})
}

func (c *Cache) dropRefLocked(ref blockRef) {
	if ref.b == nil || ref.b.refs <= 0 {
		return
	}
	ref.b.refs--
	if ref.b.refs != 0 {
		_ = ref.lease.Release()
		return
	}
	_ = c.store.Evict(context.Background(), ref.lease, func(context.Context, kvcache.Descriptor) error {
		ref.b.k.Free()
		ref.b.v.Free()
		return nil
	})
	delete(c.blocks, [2]uint64{uint64(ref.b.id), uint64(ref.b.gen)})
	c.bytes -= int64(ref.b.k.Len()+ref.b.v.Len()) * 4
}

func (c *Cache) dropEntryLocked(e *entry) {
	for i := len(e.blocks) - 1; i >= 0; i-- {
		c.dropRefLocked(e.blocks[i])
	}
}

func (c *Cache) evictOldestLocked(except string) bool {
	var key string
	var oldest *entry
	for candidateKey, candidate := range c.entries {
		if candidateKey == except {
			continue
		}
		if oldest == nil || candidate.lastUsed.Before(oldest.lastUsed) {
			key, oldest = candidateKey, candidate
		}
	}
	if oldest == nil {
		return false
	}
	delete(c.entries, key) // Remove discoverability before dropping ownership.
	c.dropEntryLocked(oldest)
	return true
}

// Stats returns a consistent accounting snapshot.
func (c *Cache) Stats() Stats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return Stats{Entries: len(c.entries), Blocks: len(c.blocks), Bytes: c.bytes, Hits: c.hits, Misses: c.misses, SavedTokens: c.saved}
}

// Close removes all entries from lookup. It returns kvcache.ErrBusy while a
// client lease still pins physical storage; a later Close completes shutdown.
func (c *Cache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed {
		c.closed = true
		for key, e := range c.entries {
			delete(c.entries, key)
			c.dropEntryLocked(e)
		}
	}
	if len(c.blocks) != 0 {
		return kvcache.ErrBusy
	}
	return c.store.Close()
}
