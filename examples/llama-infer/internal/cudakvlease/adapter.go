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
	ErrClosed    = errors.New("cudakvlease: adapter closed")
	ErrInvalid   = errors.New("cudakvlease: invalid argument")
	ErrStale     = errors.New("cudakvlease: stale allocation")
	ErrNoView    = errors.New("cudakvlease: physical page views unavailable")
	ErrNotDevice = errors.New("cudakvlease: allocation is not device resident")
	ErrNotHost   = errors.New("cudakvlease: allocation is not host resident")
	ErrNoTiering = errors.New("cudakvlease: page allocator does not support host tiering")
)

// Pages is the physical page allocator used by Adapter.
type Pages interface {
	AllocSequence(seqID int64, promptTokens int) error
	FreeSequence(seqID int64)
	BlockTable(seqID int64) []int
	NumFreeBlocks() int
	Close() error
}

type hostBuffer interface {
	Slice() []float32
	Free() error
}

type tieredPages interface {
	Pages
	AllocHost(int) (hostBuffer, error)
	CopyPageToHost(layer, slot int, k, v []float32) error
	CopyHostToPage(layer, slot int, k, v []float32) error
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

// Stats reports logical residency and physical device use.
type Stats struct {
	DevicePages     int
	FreeDevicePages int
	HostBlocks      int
	DeviceBytes     int64
	PinnedHostBytes int64
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
	seqID        int64
	slot         int
	host         hostBuffer
	lastComplete uint64
	lease        kvcache.Lease
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
	clock   uint64
	blocks  map[identity]allocation
	closed  bool
}

// New returns an adapter over pages. The adapter takes ownership of pages.
func New(pages Pages, config Config) (*Adapter, error) {
	if pages == nil || !config.valid() {
		return nil, ErrInvalid
	}
	limit := config.TotalPages
	if _, ok := pages.(tieredPages); ok {
		// Logical blocks may outnumber physical pages when cold blocks reside in
		// pinned host memory. pages remains the hard device allocation bound.
		limit = 0
	}
	store, err := kvcache.NewStore(limit)
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
	cache, err := cudakv.NewPagedCache(dev, config.NumLayers, config.KVHeads, config.HeadDim, config.BlockSize, config.TotalPages)
	if err != nil {
		return nil, fmt.Errorf("cudakvlease: allocate CUDA pages: %w", err)
	}
	pages := &cudaPages{PagedCache: cache, dev: dev}
	a, err := New(pages, config)
	if err != nil {
		_ = cache.Close()
		return nil, err
	}
	return a, nil
}

// Allocation is the owning lease for one physical page. Its zero value is invalid.
type Allocation struct {
	adapter *Adapter
	lease   kvcache.Lease
}

// Reservation excludes consumers and retirement while one exact physical
// allocation is being considered for migration. Its zero value is invalid.
type Reservation struct {
	mu       sync.Mutex
	adapter  *Adapter
	logical  kvcache.Reservation
	lease    kvcache.Lease
	key      identity
	seqID    int64
	slot     int
	finished bool
}

// ReservationSet is an atomically selected set of victim generations.
type ReservationSet struct {
	reservations []*Reservation
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
	a.blocks[identity{id, generation}] = allocation{seqID: seqID, slot: table[0], lease: lease}
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
	if physical.host != nil {
		return nil, ErrNotDevice
	}
	lease, err := a.store.Acquire(id, generation)
	if err != nil {
		return nil, fmt.Errorf("cudakvlease: acquire block: %w", err)
	}
	return &Use{adapter: a, lease: lease, slot: physical.slot}, nil
}

// Residency returns the authoritative location of allocation.
func (a *Adapter) Residency(allocation Allocation) (kvcache.Residency, error) {
	if allocation.adapter != a {
		return kvcache.Residency{}, ErrInvalid
	}
	desc, err := a.store.Describe(allocation.lease)
	if err != nil {
		return kvcache.Residency{}, err
	}
	return desc.Residency, nil
}

// ReserveVictim reserves the least recently completed device allocation not
// listed in exclude. Ties are ordered by logical ID and generation. Busy
// generations are skipped without changing their order or state.
func (a *Adapter) ReserveVictim(exclude ...Allocation) (*Reservation, error) {
	set, err := a.ReserveVictims(1, exclude...)
	if err != nil {
		return nil, err
	}
	return set.reservations[0], nil
}

// ReserveVictims atomically reserves count victims in deterministic order.
// If count candidates cannot be reserved, every acquired reservation is
// aborted before ReserveVictims returns.
func (a *Adapter) ReserveVictims(count int, exclude ...Allocation) (*ReservationSet, error) {
	if count <= 0 {
		return nil, ErrInvalid
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, ErrClosed
	}
	excluded := make(map[identity]bool, len(exclude))
	for _, allocation := range exclude {
		if allocation.adapter != a {
			return nil, ErrInvalid
		}
		excluded[identityOf(allocation)] = true
	}
	reservations := make([]*Reservation, 0, count)
	for len(reservations) < count {
		var best identity
		var physical allocation
		found := false
		for key, candidate := range a.blocks {
			if excluded[key] || candidate.host != nil {
				continue
			}
			if !found || candidate.lastComplete < physical.lastComplete ||
				(candidate.lastComplete == physical.lastComplete && (key.id < best.id ||
					(key.id == best.id && key.generation < best.generation))) {
				best, physical, found = key, candidate, true
			}
		}
		if !found {
			for _, reservation := range reservations {
				_ = reservation.logical.Abort()
				reservation.finished = true
			}
			return nil, kvcache.ErrExhausted
		}
		logical, err := a.store.Reserve(physical.lease)
		if errors.Is(err, kvcache.ErrBusy) {
			excluded[best] = true
			continue
		}
		if err != nil {
			for _, reservation := range reservations {
				_ = reservation.logical.Abort()
				reservation.finished = true
			}
			return nil, fmt.Errorf("cudakvlease: reserve victim: %w", err)
		}
		reservations = append(reservations, &Reservation{adapter: a, logical: logical, lease: physical.lease, key: best, seqID: physical.seqID, slot: physical.slot})
		excluded[best] = true
	}
	return &ReservationSet{reservations: reservations}, nil
}

// Abort releases every unfinished reservation in the set.
func (s *ReservationSet) Abort() error {
	if s == nil || len(s.reservations) == 0 {
		return ErrInvalid
	}
	var errs []error
	for _, reservation := range s.reservations {
		if err := reservation.Abort(); err != nil && !errors.Is(err, ErrStale) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Offload migrates every reserved victim. If a later migration fails, earlier
// migrations are restored in reverse order before Offload returns.
func (s *ReservationSet) Offload(ctx context.Context, completion Completion) error {
	if s == nil || len(s.reservations) == 0 || ctx == nil || completion == nil {
		return ErrInvalid
	}
	completed := make([]Allocation, 0, len(s.reservations))
	for i, reservation := range s.reservations {
		if err := reservation.Offload(ctx, completion); err != nil {
			for _, pending := range s.reservations[i+1:] {
				_ = pending.Abort()
			}
			var rollback []error
			for j := len(completed) - 1; j >= 0; j-- {
				if restoreErr := completed[j].adapter.Restore(context.WithoutCancel(ctx), completed[j]); restoreErr != nil {
					rollback = append(rollback, fmt.Errorf("restore victim %d: %w", j, restoreErr))
				}
			}
			return errors.Join(err, errors.Join(rollback...))
		}
		completed = append(completed, Allocation{adapter: reservation.adapter, lease: reservation.lease})
	}
	return nil
}

func (a *Adapter) reserve(allocation Allocation) (*Reservation, error) {
	if allocation.adapter != a {
		return nil, ErrInvalid
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil, ErrClosed
	}
	key := identityOf(allocation)
	physical, ok := a.blocks[key]
	if !ok {
		return nil, kvcache.ErrStale
	}
	logical, err := a.store.Reserve(allocation.lease)
	if err != nil {
		return nil, err
	}
	return &Reservation{adapter: a, logical: logical, lease: allocation.lease, key: key, seqID: physical.seqID, slot: physical.slot}, nil
}

// Abort releases a victim reservation without changing physical residency.
func (r *Reservation) Abort() error {
	if r == nil || r.adapter == nil {
		return ErrInvalid
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.finished {
		return ErrStale
	}
	if err := r.logical.Abort(); err != nil {
		return err
	}
	r.finished = true
	return nil
}

// Offload migrates the reserved device allocation to pinned host memory.
// completion must cover every earlier CUDA use of the page.
func (r *Reservation) Offload(ctx context.Context, completion Completion) error {
	if r == nil || r.adapter == nil || ctx == nil || completion == nil {
		return ErrInvalid
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.finished {
		return ErrStale
	}
	a := r.adapter
	tiered, ok := a.pages.(tieredPages)
	if !ok {
		_ = r.logical.Abort()
		r.finished = true
		return ErrNoTiering
	}
	target := kvcache.Residency{Backend: "cuda", Tier: "pinned-host", Domain: a.config.Domain}
	err := r.logical.Migrate(ctx, target, func(ctx context.Context, _ kvcache.Descriptor, _ kvcache.Residency) error {
		if err := completion.Synchronize(ctx); err != nil {
			return fmt.Errorf("cudakvlease: wait before offload: %w", err)
		}
		a.mu.Lock()
		defer a.mu.Unlock()
		physical, ok := a.blocks[r.key]
		if !ok || physical.seqID != r.seqID || physical.slot != r.slot {
			return ErrStale
		}
		if physical.host != nil {
			return ErrNotDevice
		}
		values := a.valuesPerPage()
		host, err := tiered.AllocHost(2 * a.config.NumLayers * values)
		if err != nil {
			return fmt.Errorf("cudakvlease: allocate pinned host page: %w", err)
		}
		keep := false
		defer func() {
			if !keep {
				_ = host.Free()
			}
		}()
		data := host.Slice()
		for layer := 0; layer < a.config.NumLayers; layer++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			k, v := hostLayer(data, layer, values)
			if err := tiered.CopyPageToHost(layer, physical.slot, k, v); err != nil {
				return fmt.Errorf("cudakvlease: offload layer %d: %w", layer, err)
			}
		}
		a.pages.FreeSequence(physical.seqID)
		physical.seqID, physical.slot, physical.host = 0, -1, host
		a.blocks[r.key] = physical
		keep = true
		return nil
	})
	r.finished = true
	return err
}

// Restore migrates the reserved host allocation to a complete device page.
// The pinned source remains authoritative until every synchronous copy and
// the logical residency publication succeed.
func (r *Reservation) Restore(ctx context.Context) error {
	if r == nil || r.adapter == nil || ctx == nil {
		return ErrInvalid
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.finished {
		return ErrStale
	}
	a := r.adapter
	tiered, ok := a.pages.(tieredPages)
	if !ok {
		_ = r.logical.Abort()
		r.finished = true
		return ErrNoTiering
	}
	target := kvcache.Residency{Backend: "cuda", Tier: "device", Domain: a.config.Domain}
	err := r.logical.Migrate(ctx, target, func(ctx context.Context, desc kvcache.Descriptor, _ kvcache.Residency) error {
		a.mu.Lock()
		defer a.mu.Unlock()
		physical, ok := a.blocks[r.key]
		if !ok || physical.seqID != r.seqID || physical.slot != r.slot {
			return ErrStale
		}
		if physical.host == nil {
			return ErrNotHost
		}
		a.nextSeq++
		seqID := a.nextSeq
		tokens := desc.Spec.Tokens.End - desc.Spec.Tokens.Start
		if err := a.pages.AllocSequence(seqID, tokens); err != nil {
			return fmt.Errorf("cudakvlease: allocate restore page: %w", err)
		}
		table := a.pages.BlockTable(seqID)
		if len(table) != 1 {
			a.pages.FreeSequence(seqID)
			return fmt.Errorf("cudakvlease: allocator returned %d restore pages, want 1", len(table))
		}
		values := a.valuesPerPage()
		data := physical.host.Slice()
		for layer := 0; layer < a.config.NumLayers; layer++ {
			if err := ctx.Err(); err != nil {
				a.pages.FreeSequence(seqID)
				return err
			}
			k, v := hostLayer(data, layer, values)
			if err := tiered.CopyHostToPage(layer, table[0], k, v); err != nil {
				a.pages.FreeSequence(seqID)
				return fmt.Errorf("cudakvlease: restore layer %d: %w", layer, err)
			}
		}
		if err := physical.host.Free(); err != nil {
			a.pages.FreeSequence(seqID)
			return fmt.Errorf("cudakvlease: free pinned host page: %w", err)
		}
		physical.seqID, physical.slot, physical.host = seqID, table[0], nil
		a.blocks[r.key] = physical
		return nil
	})
	r.finished = true
	return err
}

// Offload copies one device-resident page for every layer to a single pinned
// host allocation, then returns the physical page to the allocator. Migration
// requires the owning lease to be the sole reference. completion must cover
// every earlier CUDA use of the page.
func (a *Adapter) Offload(ctx context.Context, allocation Allocation, completion Completion) error {
	reservation, err := a.reserve(allocation)
	if err != nil {
		return err
	}
	return reservation.Offload(ctx, completion)
}

// Restore copies an offloaded block into a newly allocated device page. The
// pinned source remains authoritative until all layer copies succeed.
func (a *Adapter) Restore(ctx context.Context, allocation Allocation) error {
	reservation, err := a.reserve(allocation)
	if err != nil {
		return err
	}
	return reservation.Restore(ctx)
}

func identityOf(a Allocation) identity {
	id, generation := a.ID()
	return identity{id, generation}
}

func (a *Adapter) valuesPerPage() int {
	return a.config.BlockSize * a.config.KVHeads * a.config.HeadDim
}

func hostLayer(data []float32, layer, values int) ([]float32, []float32) {
	start := 2 * layer * values
	return data[start : start+values], data[start+values : start+2*values]
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
	u.adapter.mu.Lock()
	defer u.adapter.mu.Unlock()
	if err := u.lease.Release(); err != nil {
		return fmt.Errorf("cudakvlease: release consumer: %w", err)
	}
	key := identityOf(Allocation{adapter: u.adapter, lease: u.lease})
	physical, ok := u.adapter.blocks[key]
	if !ok {
		return ErrStale
	}
	u.adapter.clock++
	physical.lastComplete = u.adapter.clock
	u.adapter.blocks[key] = physical
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
		if physical.host != nil {
			if err := physical.host.Free(); err != nil {
				return fmt.Errorf("cudakvlease: free pinned host page: %w", err)
			}
		} else {
			a.pages.FreeSequence(physical.seqID)
		}
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

// Stats returns a consistent snapshot of physical residency. DeviceBytes is
// bounded by Config.TotalPages regardless of the number of host blocks.
func (a *Adapter) Stats() Stats {
	a.mu.Lock()
	defer a.mu.Unlock()
	free := a.pages.NumFreeBlocks()
	var host int
	for _, block := range a.blocks {
		if block.host != nil {
			host++
		}
	}
	bytesPerBlock := int64(2*a.config.NumLayers*a.valuesPerPage()) * 4
	return Stats{
		DevicePages:     a.config.TotalPages - free,
		FreeDevicePages: free,
		HostBlocks:      host,
		DeviceBytes:     int64(a.config.TotalPages-free) * bytesPerBlock,
		PinnedHostBytes: int64(host) * bytesPerBlock,
	}
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
