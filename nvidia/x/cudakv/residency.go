package cudakv

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// ResidencyTier records completed placement or an outstanding transfer. A
// pending prediction/copy is never a usable resident sequence binding.
type ResidencyTier string

const (
	ResidencyHBM         ResidencyTier = "hbm"
	ResidencyEvicting    ResidencyTier = "evicting"
	ResidencyHost        ResidencyTier = "pinned_host"
	ResidencyPrefetching ResidencyTier = "prefetching"
	ResidencyLost        ResidencyTier = "lost"
	ResidencyClosed      ResidencyTier = "closed"
)

// ResidencyBinding is a detached whole-sequence binding for ValidateSequence.
// Generation changes on restore, including when the same physical slots return.
type ResidencyBinding struct {
	SequenceID int64
	Generation uint64
	Blocks     []int
	Tokens     int
}

// Residency owns asynchronous HBM/pinned-host transitions for one complete
// sequence and every layer in its pages. Create it with NewResidency; its zero
// value is not usable. Shared-prefix eviction and partial-layer/page plans are
// unsupported. Callers must quiesce existing kernel/page use before Evict and
// serialize any future page access through the returned resident binding.
// Previously obtained raw Page aliases cannot be revoked by this API.
//
// Successful eviction returns physical pool slots for reuse, not the reserved
// whole-layer CUDA allocation. Pinned host memory and device allocation borrows
// remain owned until Close succeeds. Failed waits/closes retain retry owners.
// Close abandons a nonresident sequence; it does not invent a usable binding.
type Residency struct {
	mu             sync.Mutex
	pc             *PagedCache
	allocator      *BlockAllocator
	id, generation uint64
	sequence       int64
	blocks         []int
	tokens         int
	tier           ResidencyTier
	closing        bool
	ops            residencyOps
	releases       []func()
	views          []residencyView
	host           []*residencyHost
	pending        []residencyCompletion
	held           []int
	transferErr    error
}

type residencyCompletion interface {
	Wait(context.Context) error
	Close() error
}
type residencyHost struct {
	allocation cudamem.HostAllocation[float32]
	free       func() error
}
type residencyView struct {
	slice       cudamem.Slice[float32]
	layer, slot int
	v           bool
	elements    int
}
type residencyOps struct {
	page     func(*PagedCache, int, int) (Page, error)
	borrow   func(residencyView, *cuda.Device) (func(), error)
	allocate func(*cuda.Device, int) (*residencyHost, error)
	copy     func(residencyView, *residencyHost, *cuda.Stream, bool) (residencyCompletion, error)
	stream   func(*cuda.Stream, *cuda.Device) error
}

func nativeResidencyOps() residencyOps {
	return residencyOps{
		page:   (*PagedCache).pageViewLocked,
		borrow: func(v residencyView, dev *cuda.Device) (func(), error) { return v.slice.Borrow(dev) },
		allocate: func(dev *cuda.Device, n int) (*residencyHost, error) {
			var allocation cudamem.HostAllocation[float32]
			err := dev.WithContext(func() error {
				var err error
				allocation, err = cudamem.AllocHost[float32](n)
				return err
			})
			if allocation.Ptr() == 0 {
				return nil, err
			}
			h := &residencyHost{allocation: allocation}
			h.free = func() error { return dev.WithContext(allocation.Free) }
			return h, err
		},
		copy: func(v residencyView, h *residencyHost, stream *cuda.Stream, toHost bool) (residencyCompletion, error) {
			var completion *cudamem.Completion
			var err error
			if toHost {
				completion, err = v.slice.CopyToPinnedAsync(h.allocation, stream)
			} else {
				completion, err = v.slice.CopyFromPinnedAsync(h.allocation, stream)
			}
			if completion == nil {
				return nil, err
			}
			return completion, err
		},
		stream: func(stream *cuda.Stream, dev *cuda.Device) error {
			if stream == nil || dev == nil || stream.Device() != dev {
				return fmt.Errorf("cudakv: residency stream device mismatch")
			}
			return nil
		},
	}
}

// NewResidency binds the exact currently resident allocation generation of
// sequence. It does not enqueue work or allocate pinned storage. A non-nil
// error-result owner, if any, must be closed. The caller supplies transfer
// streams to Evict and Prefetch; no stream is constructed or owned here.
func NewResidency(pc *PagedCache, sequence int64, generation uint64) (*Residency, error) {
	return newResidency(pc, sequence, generation, nativeResidencyOps())
}

func newResidency(pc *PagedCache, sequence int64, generation uint64, ops residencyOps) (*Residency, error) {
	if pc == nil {
		return nil, fmt.Errorf("cudakv: nil residency cache")
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed || pc.compactionOwners != 0 {
		return nil, cudamem.ErrInUse
	}
	a := pc.blocks
	if a == nil || a.residencyOwners[sequence] != 0 {
		return nil, fmt.Errorf("cudakv: sequence already has a residency owner")
	}
	blocks := a.Table(sequence)
	tokens, _ := a.SequenceLength(sequence)
	if err := a.ValidateSequence(sequence, generation, blocks, tokens); err != nil {
		return nil, err
	}
	r := &Residency{pc: pc, allocator: a, id: generation, generation: generation, sequence: sequence, blocks: blocks, tokens: tokens, tier: ResidencyHBM, ops: ops}
	views, err := r.pageViewsLocked(blocks)
	if err != nil {
		return nil, err
	}
	for _, view := range views {
		release, err := ops.borrow(view, pc.dev)
		if err != nil {
			for _, release := range r.releases {
				release()
			}
			return nil, err
		}
		r.releases = append(r.releases, release)
	}
	r.views = views
	r.host = make([]*residencyHost, len(views))
	pc.residencies++
	a.residencyOwners[sequence] = r.id
	return r, nil
}

func (r *Residency) pageViewsLocked(blocks []int) ([]residencyView, error) {
	var out []residencyView
	n := r.pc.blockSize * r.pc.numKVHeads * r.pc.headDim
	for _, block := range blocks {
		for layer := 0; layer < r.pc.numLayers; layer++ {
			page, err := r.ops.page(r.pc, layer, block)
			if err != nil {
				return nil, err
			}
			out = append(out, residencyView{page.K, layer, block, false, n}, residencyView{page.V, layer, block, true, n})
		}
	}
	return out, nil
}

// AllocationGeneration identifies a sequence allocation even while suspended.
// A generation alone is not a usable binding; ValidateSequence also checks
// completed residency, physical table and token capacity.
func (pc *PagedCache) AllocationGeneration(sequence int64) (uint64, bool) {
	if pc == nil {
		return 0, false
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed {
		return 0, false
	}
	return pc.blocks.AllocationGeneration(sequence)
}

// ValidateSequence refuses stale, suspended or incompletely restored bindings.
func (pc *PagedCache) ValidateSequence(binding ResidencyBinding) error {
	if pc == nil {
		return fmt.Errorf("cudakv: nil cache")
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.closed {
		return fmt.Errorf("cudakv: cache closed")
	}
	return pc.blocks.ValidateSequence(binding.SequenceID, binding.Generation, binding.Blocks, binding.Tokens)
}

// Tier reports placement. Only ResidencyHBM permits Binding to succeed.
func (r *Residency) Tier() ResidencyTier {
	if r == nil {
		return ResidencyClosed
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.tier
}

// Binding returns a detached currently usable whole-sequence binding.
func (r *Residency) Binding() (ResidencyBinding, error) {
	if r == nil {
		return ResidencyBinding{}, fmt.Errorf("cudakv: nil residency")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.pc == nil || r.closing || r.tier != ResidencyHBM {
		return ResidencyBinding{}, fmt.Errorf("cudakv: sequence is not resident")
	}
	r.pc.mu.Lock()
	defer r.pc.mu.Unlock()
	if err := r.validateLocked(false); err != nil {
		return ResidencyBinding{}, err
	}
	return ResidencyBinding{r.sequence, r.generation, slices.Clone(r.blocks), r.tokens}, nil
}

func (r *Residency) validateLocked(suspended bool) error {
	a := r.allocator
	if r.pc.closed || r.pc.blocks != a || a.allocations[r.sequence] != r.generation || a.residencyOwners[r.sequence] != r.id || a.suspended[r.sequence] != suspended || a.lens[r.sequence] != r.tokens {
		return fmt.Errorf("cudakv: residency allocation changed")
	}
	if !suspended && !slices.Equal(a.tables[r.sequence], r.blocks) {
		return fmt.Errorf("cudakv: residency table changed")
	}
	return nil
}

// Evict starts D2H copies of every selected page/layer. Until Wait succeeds the
// sequence is suspended and the original slots remain held. Host allocation
// exhaustion leaves the HBM binding valid; the caller can continue there.
func (r *Residency) Evict(stream *cuda.Stream) error { return r.begin(stream, true) }

// Prefetch reserves new pool slots and starts H2D copies. Slots and a new
// allocation generation become a usable binding only after successful Wait.
// Exhaustion leaves the pinned-host copy owned and the sequence suspended.
func (r *Residency) Prefetch(stream *cuda.Stream) error { return r.begin(stream, false) }

func (r *Residency) begin(stream *cuda.Stream, toHost bool) error {
	if r == nil {
		return fmt.Errorf("cudakv: nil residency")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.pc == nil || r.closing || (toHost && r.tier != ResidencyHBM) || (!toHost && r.tier != ResidencyHost) {
		return fmt.Errorf("cudakv: invalid residency transition")
	}
	if err := r.ops.stream(stream, r.pc.dev); err != nil {
		return err
	}
	r.pc.mu.Lock()
	defer r.pc.mu.Unlock()
	if err := r.validateLocked(!toHost); err != nil {
		return err
	}
	a := r.allocator
	if toHost {
		for _, b := range r.blocks {
			if a.holders[b] != 1 {
				return fmt.Errorf("cudakv: shared-prefix eviction unsupported")
			}
		}
		for i, v := range r.views {
			if r.host[i] != nil {
				continue
			}
			h, err := r.ops.allocate(r.pc.dev, v.elements)
			r.host[i] = h
			if err != nil {
				return fmt.Errorf("cudakv: allocate pinned residency: %w", err)
			}
			if h == nil {
				return fmt.Errorf("cudakv: missing pinned residency allocation")
			}
		}
		a.suspended[r.sequence] = true
		for _, b := range r.blocks {
			a.holders[b]++
			a.unavailable[b] = true
		}
		r.held = slices.Clone(r.blocks)
		r.tier = ResidencyEvicting
	} else {
		if a.nextAllocation == ^uint64(0) {
			return fmt.Errorf("cudakv: allocation identity exhausted")
		}
		if len(a.free) < len(r.blocks) {
			return fmt.Errorf("cudakv: prefetch pool exhausted")
		}
		for range r.blocks {
			b := a.take()
			a.unavailable[b] = true
			r.held = append(r.held, b)
		}
		views, err := r.pageViewsLocked(r.held)
		if err != nil {
			r.dropHeldLocked()
			return err
		}
		r.views = views
		r.tier = ResidencyPrefetching
	}
	r.pending = make([]residencyCompletion, len(r.views))
	r.transferErr = nil
	for i, v := range r.views {
		completion, err := r.ops.copy(v, r.host[i], stream, toHost)
		r.pending[i] = completion
		if err != nil {
			r.transferErr = fmt.Errorf("cudakv: enqueue residency copy %d: %w", i, err)
			return r.transferErr
		}
		if completion == nil {
			r.transferErr = fmt.Errorf("cudakv: missing copy completion")
			return r.transferErr
		}
	}
	return nil
}

// Wait establishes actual transfer completion. Cancellation only stops waiting;
// it does not cancel CUDA work, free owners or publish residency. Event cleanup
// and native wait failures keep the completion owner for a later retry.
func (r *Residency) Wait(ctx context.Context) error {
	if r == nil {
		return fmt.Errorf("cudakv: nil residency")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.waitLocked(ctx)
}
func (r *Residency) waitLocked(ctx context.Context) error {
	if r.pc == nil {
		return fmt.Errorf("cudakv: uninitialized residency")
	}
	if r.tier != ResidencyEvicting && r.tier != ResidencyPrefetching {
		return nil
	}
	for i, c := range r.pending {
		if c == nil {
			continue
		}
		if ctx != nil && ctx.Err() != nil {
			return ctx.Err()
		}
		if err := c.Wait(ctx); err != nil {
			return err
		}
		r.pending[i] = nil
	}
	r.pending = nil
	r.pc.mu.Lock()
	defer r.pc.mu.Unlock()
	toHost := r.tier == ResidencyEvicting
	if err := r.validateLocked(true); err != nil {
		r.dropHeldLocked()
		r.tier = ResidencyLost
		return errors.Join(err, r.transferErr)
	}
	if r.transferErr != nil {
		err := r.transferErr
		r.transferErr = nil
		r.dropHeldLocked()
		if toHost {
			delete(r.allocator.suspended, r.sequence)
			r.tier = ResidencyHBM
		} else {
			r.tier = ResidencyHost
		}
		return err
	}
	a := r.allocator
	if toHost {
		// Remove the sequence's ownership before dropping the transfer's hold.
		for _, b := range r.blocks {
			a.drop(b)
		}
		a.tables[r.sequence] = nil
		r.dropHeldLocked()
		r.tier = ResidencyHost
	} else {
		if a.nextAllocation == ^uint64(0) {
			r.dropHeldLocked()
			r.tier = ResidencyHost
			return fmt.Errorf("cudakv: allocation identity exhausted")
		}
		a.nextAllocation++
		r.generation = a.nextAllocation
		a.allocations[r.sequence] = r.generation
		r.blocks = slices.Clone(r.held)
		a.tables[r.sequence] = slices.Clone(r.held)
		for _, b := range r.held {
			delete(a.unavailable, b)
		}
		r.held = nil
		delete(a.suspended, r.sequence)
		r.tier = ResidencyHBM
	}
	return nil
}
func (r *Residency) dropHeldLocked() {
	for _, b := range r.held {
		delete(r.allocator.unavailable, b)
		r.allocator.drop(b)
	}
	r.held = nil
}

// Close waits for pending transfers and releases pinned buffers and device
// borrows. Failed completion/free cleanup retains the owner for retry. Closing
// a completed host-resident sequence abandons its suspended allocation. A
// successful resident HBM sequence stays allocated for its caller.
func (r *Residency) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.tier == ResidencyClosed {
		return nil
	}
	if r.pc == nil {
		return fmt.Errorf("cudakv: uninitialized residency")
	}
	r.closing = true
	if err := r.waitLocked(nil); err != nil {
		return err
	}
	var err error
	for i, h := range r.host {
		if h == nil {
			continue
		}
		if e := h.free(); e != nil {
			err = errors.Join(err, e)
		} else {
			r.host[i] = nil
		}
	}
	if err != nil {
		return err
	}
	r.pc.mu.Lock()
	defer r.pc.mu.Unlock()
	if r.allocator.allocations[r.sequence] == r.generation && r.allocator.suspended[r.sequence] {
		r.allocator.FreeSequence(r.sequence)
	}
	if r.allocator.residencyOwners[r.sequence] == r.id {
		delete(r.allocator.residencyOwners, r.sequence)
	}
	for _, release := range r.releases {
		release()
	}
	r.releases = nil
	r.pc.residencies--
	r.tier = ResidencyClosed
	return nil
}
