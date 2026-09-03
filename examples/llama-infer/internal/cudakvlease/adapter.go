// Package cudakvlease binds CUDA physical KV pages to generation-safe logical leases.
package cudakvlease

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/tmc/lib/lm/kvcache"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

var (
	ErrClosed  = errors.New("cudakvlease: adapter closed")
	ErrInvalid = errors.New("cudakvlease: invalid argument")
	ErrStale   = errors.New("cudakvlease: stale allocation")
	ErrNoView  = errors.New("cudakvlease: physical page views unavailable")
)

// Pages is the physical page allocator used by Adapter.
type Pages interface {
	AllocSequence(seqID int64, promptTokens int) error
	FreeSequence(seqID int64)
	BlockTable(seqID int64) []int
	NumFreeBlocks() int
	Close() error
}

// Completion waits for device work which may read a page.
type Completion interface {
	Synchronize(context.Context) error
}

// Config describes one CUDA paged allocation domain.
type Config struct {
	NumLayers  int
	KVHeads    int
	HeadDim    int
	BlockSize  int
	TotalPages int
	DType      string
	Domain     string
}

func (c Config) valid() bool {
	return c.NumLayers > 0 && c.KVHeads > 0 && c.HeadDim > 0 && c.BlockSize > 0 &&
		c.TotalPages > 0 && c.DType != "" && c.Domain != ""
}

type identity struct {
	id         kvcache.ID
	generation kvcache.Generation
}

type allocation struct {
	seqID int64
	slot  int
}

// Adapter owns a CUDA page allocator and its corresponding logical block store.
// A physical page is not returned to the allocator until all Uses have completed
// their device fences and Retire's final fence has completed.
type Adapter struct {
	mu sync.Mutex

	pages   Pages
	store   *kvcache.Store
	config  Config
	nextSeq int64
	blocks  map[identity]allocation
	closed  bool
}

// New returns an adapter over pages. The adapter takes ownership of pages.
func New(pages Pages, config Config) (*Adapter, error) {
	if pages == nil || !config.valid() {
		return nil, ErrInvalid
	}
	store, err := kvcache.NewStore(config.TotalPages)
	if err != nil {
		return nil, fmt.Errorf("cudakvlease: create block store: %w", err)
	}
	return &Adapter{pages: pages, store: store, config: config, blocks: make(map[identity]allocation)}, nil
}

// NewCUDA allocates CUDA pages and returns their logical adapter.
func NewCUDA(dev *cuda.Device, config Config) (*Adapter, error) {
	if dev == nil || !config.valid() {
		return nil, ErrInvalid
	}
	pages, err := cudakv.NewPagedCache(dev, config.NumLayers, config.KVHeads, config.HeadDim, config.BlockSize, config.TotalPages)
	if err != nil {
		return nil, fmt.Errorf("cudakvlease: allocate CUDA pages: %w", err)
	}
	a, err := New(pages, config)
	if err != nil {
		_ = pages.Close()
		return nil, err
	}
	return a, nil
}

// Allocation is the owning lease for one physical page. Its zero value is invalid.
type Allocation struct {
	adapter *Adapter
	lease   kvcache.Lease
}

// ID returns the logical identity and generation associated with the page.
func (a Allocation) ID() (kvcache.ID, kvcache.Generation) { return a.lease.ID() }

// Allocate reserves one physical CUDA page for tokens.
func (a *Adapter) Allocate(owner string, tokens kvcache.Range) (Allocation, error) {
	if owner == "" || tokens.Start < 0 || tokens.End <= tokens.Start || tokens.End-tokens.Start > a.config.BlockSize {
		return Allocation{}, ErrInvalid
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return Allocation{}, ErrClosed
	}
	a.nextSeq++
	seqID := a.nextSeq
	if err := a.pages.AllocSequence(seqID, tokens.End-tokens.Start); err != nil {
		return Allocation{}, fmt.Errorf("cudakvlease: allocate physical page: %w", err)
	}
	table := a.pages.BlockTable(seqID)
	if len(table) != 1 {
		a.pages.FreeSequence(seqID)
		return Allocation{}, fmt.Errorf("cudakvlease: allocator returned %d pages, want 1", len(table))
	}
	lease, err := a.store.Allocate(kvcache.Spec{
		Owner:  owner,
		Tokens: tokens,
		Format: kvcache.Format{Layer: kvcache.AllLayers, DType: a.config.DType, Layout: "cuda-paged-all-layers", KVHeads: a.config.KVHeads, HeadDim: a.config.HeadDim},
	}, kvcache.Residency{Backend: "cuda", Tier: "device", Domain: a.config.Domain})
	if err != nil {
		a.pages.FreeSequence(seqID)
		return Allocation{}, fmt.Errorf("cudakvlease: allocate logical block: %w", err)
	}
	id, generation := lease.ID()
	a.blocks[identity{id, generation}] = allocation{seqID: seqID, slot: table[0]}
	return Allocation{adapter: a, lease: lease}, nil
}

// Use pins a page while a CUDA consumer uses its block-table slot.
type Use struct {
	mu sync.Mutex

	adapter *Adapter
	lease   kvcache.Lease
	slot    int
	done    bool
}

// Bind acquires a consumer lease and returns the physical block-table slot.
func (a *Adapter) Bind(id kvcache.ID, generation kvcache.Generation) (*Use, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, ErrClosed
	}
	physical, ok := a.blocks[identity{id, generation}]
	if !ok {
		return nil, ErrStale
	}
	lease, err := a.store.Acquire(id, generation)
	if err != nil {
		return nil, fmt.Errorf("cudakvlease: acquire block: %w", err)
	}
	return &Use{adapter: a, lease: lease, slot: physical.slot}, nil
}

// Slot returns the physical slot while the Use remains live.
func (u *Use) Slot() (int, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.done {
		return 0, ErrStale
	}
	return u.slot, nil
}

// Page returns the physical K and V storage for layer. The caller must not
// retain or use the returned views after Complete.
func (u *Use) Page(layer int) (cudakv.Page, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.done {
		return cudakv.Page{}, ErrStale
	}
	pages, ok := u.adapter.pages.(interface {
		Page(layer, slot int) (cudakv.Page, error)
	})
	if !ok {
		return cudakv.Page{}, ErrNoView
	}
	page, err := pages.Page(layer, u.slot)
	if err != nil {
		return cudakv.Page{}, fmt.Errorf("cudakvlease: get physical page: %w", err)
	}
	return page, nil
}

// Complete waits for the consumer's CUDA work and then releases its lease.
// If waiting fails or ctx is canceled, the lease remains live and Complete may
// be retried. This fails closed: an uncertain device completion cannot permit
// physical page reuse.
func (u *Use) Complete(ctx context.Context, completion Completion) error {
	if ctx == nil || completion == nil {
		return ErrInvalid
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.done {
		return ErrStale
	}
	if err := completion.Synchronize(ctx); err != nil {
		return fmt.Errorf("cudakvlease: wait for consumer: %w", err)
	}
	if err := u.lease.Release(); err != nil {
		return fmt.Errorf("cudakvlease: release consumer: %w", err)
	}
	u.done = true
	return nil
}

// Retire waits for final CUDA work and retires the logical generation before
// returning its physical page to the allocator. It fails with kvcache.ErrBusy
// while any Use remains live.
func (a *Adapter) Retire(ctx context.Context, allocation Allocation, completion Completion) error {
	if ctx == nil || completion == nil || allocation.adapter != a {
		return ErrInvalid
	}
	id, generation := allocation.ID()
	key := identity{id, generation}
	return a.store.Evict(ctx, allocation.lease, func(ctx context.Context, _ kvcache.Descriptor) error {
		if err := completion.Synchronize(ctx); err != nil {
			return fmt.Errorf("cudakvlease: wait for retirement: %w", err)
		}
		a.mu.Lock()
		defer a.mu.Unlock()
		physical, ok := a.blocks[key]
		if !ok {
			return ErrStale
		}
		a.pages.FreeSequence(physical.seqID)
		delete(a.blocks, key)
		return nil
	})
}

// NumFreePages returns the physical allocator's current free-page count.
func (a *Adapter) NumFreePages() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.pages.NumFreeBlocks()
}

// Close releases the physical allocator. It returns kvcache.ErrBusy while an
// allocation or Use is live.
func (a *Adapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	if err := a.store.Close(); err != nil {
		return err
	}
	if err := a.pages.Close(); err != nil {
		return fmt.Errorf("cudakvlease: close pages: %w", err)
	}
	a.closed = true
	return nil
}
