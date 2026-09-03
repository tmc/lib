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
	Entries int
	Blocks  int
	Bytes   int64
}

// Identity identifies one immutable layer block allocation.
type Identity struct {
	ID         kvcache.ID
	Generation kvcache.Generation
	Start      int
	End        int
	Layer      int
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

	cfg     Config
	store   *kvcache.Store
	entries map[string]*entry
	blocks  map[[2]uint64]*block
	bytes   int64
	closed  bool
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
	return &Cache{cfg: cfg, store: store, entries: make(map[string]*entry), blocks: make(map[[2]uint64]*block)}, nil
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
	bytes := int64(len(k)+len(v)) * 4
	if c.bytes+bytes > c.cfg.MaxBytes {
		return blockRef{}, kvcache.ErrExhausted
	}
	ks, err := mempool.AllocFromSlice(c.cfg.Device, k)
	if err != nil {
		return blockRef{}, fmt.Errorf("metalprefix: allocate K block: %w", err)
	}
	vs, err := mempool.AllocFromSlice(c.cfg.Device, v)
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
	cache   *Cache
	tokens  int
	current int32
	refs    []blockRef
	once    sync.Once
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
		return nil, false, nil
	}
	refs := make([]blockRef, 0, len(found.blocks))
	for _, ref := range found.blocks {
		shared, err := c.store.Share(ref.lease)
		if err != nil {
			for i := len(refs) - 1; i >= 0; i-- {
				c.dropRefLocked(refs[i])
			}
			return nil, false, fmt.Errorf("metalprefix: acquire prefix: %w", err)
		}
		ref.b.refs++
		refs = append(refs, blockRef{b: ref.b, lease: shared})
	}
	found.lastUsed = time.Now()
	return &Lease{cache: c, tokens: len(found.tokens), current: found.current, refs: refs}, true, nil
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
	return Stats{Entries: len(c.entries), Blocks: len(c.blocks), Bytes: c.bytes}
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
