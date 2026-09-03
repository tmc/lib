// SPDX-License-Identifier: Apache-2.0

package kvcache

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// ID identifies a logical KV block for the lifetime of a Store.
type ID uint64

// Generation distinguishes different allocations that use the same ID.
type Generation uint64

// Range is the half-open logical token range stored in a block.
type Range struct {
	Start int
	End   int
}

// Format describes the interpretation of a block's K and V data.
// Layout and DType are backend-defined names; the store does not interpret them.
type Format struct {
	Layer   int
	DType   string
	Layout  string
	KVHeads int
	HeadDim int
}

// Residency names the backend location containing a block.
//
// Backend, Tier, and Domain are an opaque tuple. In particular, Tier is not an
// ordered or portable memory hierarchy: "unified" on Metal and "device" on
// CUDA have different synchronization and transfer rules.
type Residency struct {
	Backend string
	Tier    string
	Domain  string
}

// Spec is the immutable description supplied when allocating a block.
type Spec struct {
	Owner  string
	Tokens Range
	Format Format
}

// Descriptor is a consistent snapshot of a live block's metadata.
type Descriptor struct {
	ID         ID
	Generation Generation
	Spec       Spec
	Residency  Residency
	References int
}

var (
	// ErrClosed reports an operation on a closed store.
	ErrClosed = errors.New("kvcache: store closed")
	// ErrInvalid reports malformed metadata or a zero lease.
	ErrInvalid = errors.New("kvcache: invalid argument")
	// ErrStale reports a released lease or a retired block generation.
	ErrStale = errors.New("kvcache: stale lease")
	// ErrBusy reports a block in transition or one that still has shared users.
	ErrBusy = errors.New("kvcache: block busy")
	// ErrExhausted reports that the store's configured block limit was reached.
	ErrExhausted = errors.New("kvcache: block limit reached")
)

// Lease pins one generation of a block. Its zero value is invalid.
// A Lease may be copied, but all copies denote the same reference and Release
// succeeds only once. Use Store.Share to obtain another reference.
type Lease struct {
	store      *Store
	id         ID
	generation Generation
	serial     uint64
}

// ID returns the leased block identity and generation.
func (l Lease) ID() (ID, Generation) { return l.id, l.generation }

// Release drops this reference. The last reference must instead be passed to
// Store.Evict, which couples retirement to backend reclamation. Releasing a
// zero, stale, or already released lease returns ErrStale.
func (l Lease) Release() error {
	if l.store == nil || l.serial == 0 {
		return ErrStale
	}
	return l.store.release(l)
}

type block struct {
	desc      Descriptor
	leases    map[uint64]struct{}
	migrating bool
}

// Store coordinates logical KV block identity, leases, and residency changes.
// It does not allocate, copy, synchronize, or free backend memory.
type Store struct {
	mu sync.Mutex

	limit      int
	nextID     ID
	nextGen    Generation
	nextSerial uint64
	freeIDs    []ID
	blocks     map[ID]*block
	closed     bool
}

// NewStore returns a block store. A zero limit means no configured limit.
func NewStore(limit int) (*Store, error) {
	if limit < 0 {
		return nil, fmt.Errorf("%w: negative block limit", ErrInvalid)
	}
	return &Store{limit: limit, blocks: make(map[ID]*block)}, nil
}

func validSpec(spec Spec) bool {
	return spec.Owner != "" && spec.Tokens.Start >= 0 && spec.Tokens.End > spec.Tokens.Start &&
		spec.Format.Layer >= 0 && spec.Format.DType != "" && spec.Format.Layout != "" &&
		spec.Format.KVHeads > 0 && spec.Format.HeadDim > 0
}

func validResidency(r Residency) bool {
	return r.Backend != "" && r.Tier != "" && r.Domain != ""
}

// Allocate records a new block and returns its owning reference.
func (s *Store) Allocate(spec Spec, residency Residency) (Lease, error) {
	if !validSpec(spec) || !validResidency(residency) {
		return Lease{}, ErrInvalid
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Lease{}, ErrClosed
	}
	if s.limit != 0 && len(s.blocks) >= s.limit {
		return Lease{}, ErrExhausted
	}
	var id ID
	if n := len(s.freeIDs); n != 0 {
		id = s.freeIDs[n-1]
		s.freeIDs = s.freeIDs[:n-1]
	} else {
		s.nextID++
		id = s.nextID
	}
	s.nextGen++
	s.nextSerial++
	l := Lease{s, id, s.nextGen, s.nextSerial}
	s.blocks[l.id] = &block{
		desc:   Descriptor{ID: l.id, Generation: l.generation, Spec: spec, Residency: residency, References: 1},
		leases: map[uint64]struct{}{l.serial: {}},
	}
	return l, nil
}

// Acquire pins an existing identity and generation.
func (s *Store) Acquire(id ID, generation Generation) (Lease, error) {
	if id == 0 || generation == 0 {
		return Lease{}, ErrInvalid
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return Lease{}, ErrClosed
	}
	b := s.blocks[id]
	if b == nil || b.desc.Generation != generation {
		return Lease{}, ErrStale
	}
	if b.migrating {
		return Lease{}, ErrBusy
	}
	return s.newLeaseLocked(b), nil
}

// Share creates an independent reference to the same block generation.
func (s *Store) Share(l Lease) (Lease, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := s.blockLocked(l)
	if err != nil {
		return Lease{}, err
	}
	if b.migrating {
		return Lease{}, ErrBusy
	}
	return s.newLeaseLocked(b), nil
}

func (s *Store) newLeaseLocked(b *block) Lease {
	s.nextSerial++
	l := Lease{s, b.desc.ID, b.desc.Generation, s.nextSerial}
	b.leases[l.serial] = struct{}{}
	b.desc.References++
	return l
}

// Describe returns the current metadata for a lease.
func (s *Store) Describe(l Lease) (Descriptor, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := s.blockLocked(l)
	if err != nil {
		return Descriptor{}, err
	}
	return b.desc, nil
}

func (s *Store) blockLocked(l Lease) (*block, error) {
	if s.closed {
		return nil, ErrClosed
	}
	if l.store != s || l.id == 0 || l.generation == 0 || l.serial == 0 {
		return nil, ErrStale
	}
	b := s.blocks[l.id]
	if b == nil || b.desc.Generation != l.generation {
		return nil, ErrStale
	}
	if _, ok := b.leases[l.serial]; !ok {
		return nil, ErrStale
	}
	return b, nil
}

func (s *Store) release(l Lease) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := s.blockLocked(l)
	if err != nil {
		return err
	}
	if b.migrating {
		return ErrBusy
	}
	if b.desc.References == 1 {
		return ErrBusy
	}
	delete(b.leases, l.serial)
	b.desc.References--
	return nil
}

// Migrate runs move while the source residency remains authoritative, then
// atomically publishes target. A move error leaves metadata at the source;
// move must clean up any unpublished target allocation before returning an
// error. Once move returns nil the migration commits even if ctx is canceled
// concurrently, because rolling back metadata after a successful destructive
// transfer could expose a hole.
// Migration requires the sole lease, so a backend cannot move or retire source
// storage while another consumer may still use it. New Acquire and Share calls
// fail with ErrBusy until the move finishes.
func (s *Store) Migrate(ctx context.Context, l Lease, target Residency, move func(context.Context, Descriptor, Residency) error) error {
	if ctx == nil || !validResidency(target) || move == nil {
		return ErrInvalid
	}
	s.mu.Lock()
	b, err := s.blockLocked(l)
	if err == nil && (b.migrating || b.desc.References != 1) {
		err = ErrBusy
	}
	if err != nil {
		s.mu.Unlock()
		return err
	}
	b.migrating = true
	before := b.desc
	s.mu.Unlock()

	if err = ctx.Err(); err == nil {
		err = move(ctx, before, target)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	b = s.blocks[l.id]
	if b == nil || b.desc.Generation != l.generation {
		return ErrStale
	}
	b.migrating = false
	if err != nil {
		return err
	}
	b.desc.Residency = target
	return nil
}

// Evict runs discard and retires a block generation atomically. Eviction is
// allowed only through its sole lease. On failure, the block and lease remain
// live. A successful eviction invalidates the lease.
func (s *Store) Evict(ctx context.Context, l Lease, discard func(context.Context, Descriptor) error) error {
	if ctx == nil || discard == nil {
		return ErrInvalid
	}
	s.mu.Lock()
	b, err := s.blockLocked(l)
	if err == nil && (b.migrating || b.desc.References != 1) {
		err = ErrBusy
	}
	if err != nil {
		s.mu.Unlock()
		return err
	}
	b.migrating = true
	before := b.desc
	s.mu.Unlock()

	if err = ctx.Err(); err == nil {
		err = discard(ctx, before)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	b = s.blocks[l.id]
	if b == nil || b.desc.Generation != l.generation {
		return ErrStale
	}
	b.migrating = false
	if err != nil {
		return err
	}
	delete(s.blocks, l.id)
	s.freeIDs = append(s.freeIDs, l.id)
	return nil
}

// Close marks an empty store closed. It returns ErrBusy while any block is
// live, so shutdown cannot silently invalidate a backend allocation.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if len(s.blocks) != 0 {
		return ErrBusy
	}
	s.closed = true
	return nil
}
