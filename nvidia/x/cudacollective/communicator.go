package cudacollective

import (
	"errors"
	"fmt"
	"sync"
)

// Generation represents an incarnation counter for a communicator clique.
// When a rank fails or a communicator is aborted, its Clique marks that
// generation and every older one failed. Subsequent operations or re-creation
// with a failed generation fail closed. Generation 0 is invalid.
type Generation uint64

// Communicator is the coordinated control plane interface for a rank.
// Its zero value is not usable; create one with NewCommunicator.
type Communicator struct {
	// When nested, locks are acquired in the order clique.mu, mu, inFlightMu.
	// clique.mu is never held across transport, native, Close or completion
	// calls.
	mu         sync.RWMutex
	closeMu    sync.Mutex
	rank       int
	clique     *Clique
	membership Membership
	generation Generation
	transport  Transport
	closed     bool
	closeErr   error
	failed     bool
	failErr    error

	// In-flight buffer tracking to prevent premature buffer reuse and enforce backpressure.
	inFlightMu sync.Mutex
	inFlight   map[uintptr]*InFlightTransfer

	// Participant heartbeat / liveness tracking
	activeParticipants map[int]bool
}

// InFlightTransfer tracks an ongoing transfer buffer and its completion.
type InFlightTransfer struct {
	Slot       int // 0 or 1 for double buffering
	BufferAddr uintptr
	Bytes      int
	Done       chan struct{}
	Err        error // read only after Done closes
}

// CommunicatorConfig provides parameters for creating a Communicator.
// Clique supplies the membership and failure authority; every rank and
// every reconstruction of one clique must share the same Clique.
type CommunicatorConfig struct {
	Rank       int
	Clique     *Clique
	Generation Generation
	Transport  Transport
}

// NewCommunicator creates and validates a Communicator for a specific rank and generation.
// It refuses a generation at or below the Clique's failed generation.
func NewCommunicator(cfg CommunicatorConfig) (*Communicator, error) {
	q := cfg.Clique
	if q == nil {
		return nil, errors.New("cudacollective: clique cannot be nil")
	}
	m := q.membership
	if m.WorldSize() <= 0 {
		return nil, fmt.Errorf("%w: clique has no membership", ErrInvalidWorldSize)
	}
	if cfg.Rank < 0 || cfg.Rank >= m.WorldSize() {
		return nil, fmt.Errorf("%w: rank %d out of world bounds [0, %d)", ErrInvalidRank, cfg.Rank, m.WorldSize())
	}
	if cfg.Generation == 0 {
		return nil, fmt.Errorf("%w: generation 0 is invalid", ErrCommunicatorGeneration)
	}
	if cfg.Transport == nil {
		return nil, errors.New("cudacollective: transport cannot be nil")
	}

	participants := make(map[int]bool, m.WorldSize())
	for i := 0; i < m.WorldSize(); i++ {
		participants[i] = true
	}

	q.mu.RLock()
	defer q.mu.RUnlock()
	if err := q.check(cfg.Generation); err != nil {
		return nil, err
	}
	return &Communicator{
		rank:               cfg.Rank,
		clique:             q,
		membership:         m,
		generation:         cfg.Generation,
		transport:          cfg.Transport,
		inFlight:           make(map[uintptr]*InFlightTransfer),
		activeParticipants: participants,
	}, nil
}

// Rank returns the rank ID of this communicator.
func (c *Communicator) Rank() int {
	return c.rank
}

// Generation returns the generation counter.
func (c *Communicator) Generation() Generation {
	return c.generation
}

// Membership returns the rank membership.
func (c *Communicator) Membership() Membership {
	return c.membership
}

// Transport returns the active transport.
func (c *Communicator) Transport() Transport {
	return c.transport
}

// errUnspecifiedFailure records a MarkFailed call with a nil error.
var errUnspecifiedFailure = errors.New("cudacollective: unspecified failure")

// MarkFailed transitions the communicator to a permanently failed state with err.
// It also marks the communicator's generation failed in its Clique before
// returning, so every communicator of that generation or an older one refuses
// admission.
// Future operations on this communicator return ErrCommunicatorGeneration
// wrapping the first err; those on other communicators do not wrap err.
// Work admitted before the failure keeps its reservations; Close remains
// available.
func (c *Communicator) MarkFailed(err error) {
	if err == nil {
		err = errUnspecifiedFailure
	}
	if q := c.clique; q != nil {
		q.mu.Lock()
		defer q.mu.Unlock()
		q.fail(c.generation, c.rank)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failed = true
	if c.failErr == nil {
		c.failErr = err
	}
}

// ReportLostParticipant records that a remote participant dropped out or failed.
// Like MarkFailed, it marks the communicator's generation failed in its Clique
// before returning.
func (c *Communicator) ReportLostParticipant(rank int) {
	if q := c.clique; q != nil {
		q.mu.Lock()
		defer q.mu.Unlock()
		q.fail(c.generation, c.rank)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.activeParticipants, rank)
	c.failed = true
	c.failErr = fmt.Errorf("%w: rank %d", ErrLostParticipant, rank)
}

// CheckAlive verifies that all members are present and the generation is active.
// The result is a point-in-time observation, not a lease on later work.
func (c *Communicator) CheckAlive() error {
	if q := c.clique; q != nil {
		q.mu.RLock()
		defer q.mu.RUnlock()
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return ErrCommunicatorClosed
	}
	if c.failed {
		return fmt.Errorf("%w: %w", ErrCommunicatorGeneration, c.failErr)
	}
	if err := c.cliqueCheck(); err != nil {
		return err
	}
	if len(c.activeParticipants) < c.membership.WorldSize() {
		return fmt.Errorf("%w: missing ranks", ErrLostParticipant)
	}
	return nil
}

// cliqueCheck reports whether the Clique has marked c's generation failed.
// The caller holds c.clique.mu.
func (c *Communicator) cliqueCheck() error {
	if c.clique == nil {
		return nil
	}
	return c.clique.check(c.generation)
}

// ValidateRange verifies that [addr, addr+bytes) is non-nil, does not wrap
// the address space, and lies completely within the reserved transfer buffer.
func (tr *InFlightTransfer) ValidateRange(addr uintptr, bytes int) error {
	if tr == nil || addr == 0 || tr.BufferAddr == 0 {
		return ErrNilBuffer
	}
	if bytes <= 0 || tr.Bytes <= 0 {
		return ErrBufferLengthMismatch
	}
	lo := uint64(addr)
	n := uint64(bytes)
	if lo > ^uint64(0)-n {
		return fmt.Errorf("%w: buffer range [0x%x, +%d) wraps the address space", ErrBufferLengthMismatch, lo, bytes)
	}
	hi := lo + n
	resLo := uint64(tr.BufferAddr)
	resN := uint64(tr.Bytes)
	if resLo > ^uint64(0)-resN {
		return fmt.Errorf("%w: reservation range [0x%x, +%d) wraps the address space", ErrBufferLengthMismatch, resLo, tr.Bytes)
	}
	resHi := resLo + resN
	if lo >= resLo && lo < resHi && hi > resHi {
		return fmt.Errorf("%w: buffer range [0x%x, 0x%x) overruns reserved slot %d [0x%x, 0x%x)", ErrBufferLengthMismatch, lo, hi, tr.Slot, resLo, resHi)
	}
	if lo < resLo && hi > resLo {
		return fmt.Errorf("%w: %w: buffer range [0x%x, 0x%x) partially overlaps reserved slot %d [0x%x, 0x%x)", ErrBufferOverlap, ErrBufferLengthMismatch, lo, hi, tr.Slot, resLo, resHi)
	}
	if lo < resLo || hi > resHi {
		if n > resN {
			return fmt.Errorf("%w: buffer extent %d exceeds reserved slot %d capacity %d", ErrBufferLengthMismatch, bytes, tr.Slot, tr.Bytes)
		}
		return fmt.Errorf("%w: buffer range [0x%x, 0x%x) outside reserved slot %d [0x%x, 0x%x)", ErrBufferUnbound, lo, hi, tr.Slot, resLo, resHi)
	}
	return nil
}

// AcquireTransferBuffer marks a buffer range [addr, addr+bytes) as actively transferring in a double-buffered slot.
// If the buffer or any overlapping subrange is already registered in flight, it returns ErrBufferInUse.
// It refuses a generation its Clique has marked failed: once MarkFailed or
// ReportLostParticipant has returned on any communicator of the Clique, no
// reservation of that generation or an older one is admitted.
func (c *Communicator) AcquireTransferBuffer(addr uintptr, bytes int, slot int) (*InFlightTransfer, error) {
	// Hold the Clique admission through reservation insertion.
	if q := c.clique; q != nil {
		q.mu.RLock()
		defer q.mu.RUnlock()
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, ErrCommunicatorClosed
	}
	if c.failed {
		return nil, fmt.Errorf("%w: %w", ErrCommunicatorGeneration, c.failErr)
	}
	if err := c.cliqueCheck(); err != nil {
		return nil, err
	}
	if len(c.activeParticipants) < c.membership.WorldSize() {
		return nil, ErrLostParticipant
	}
	if addr == 0 {
		return nil, ErrNilBuffer
	}
	if bytes <= 0 {
		return nil, ErrBufferLengthMismatch
	}
	lo := uint64(addr)
	n := uint64(bytes)
	if lo > ^uint64(0)-n {
		return nil, fmt.Errorf("%w: buffer range [0x%x, +%d) wraps the address space", ErrBufferLengthMismatch, lo, bytes)
	}
	hi := lo + n

	c.inFlightMu.Lock()
	defer c.inFlightMu.Unlock()

	for base, existing := range c.inFlight {
		select {
		case <-existing.Done:
			// Completed, safe to prune
			delete(c.inFlight, base)
			continue
		default:
		}
		existingLo := uint64(existing.BufferAddr)
		existingHi := existingLo + uint64(existing.Bytes)
		if lo < existingHi && existingLo < hi {
			if lo == existingLo && hi == existingHi {
				return nil, fmt.Errorf("%w: address 0x%x in slot %d", ErrBufferInUse, addr, existing.Slot)
			}
			return nil, fmt.Errorf("%w: %w: range [0x%x, 0x%x) overlaps active transfer [0x%x, 0x%x) in slot %d",
				ErrBufferInUse, ErrBufferOverlap, lo, hi, existingLo, existingHi, existing.Slot)
		}
	}

	tr := &InFlightTransfer{
		Slot:       slot,
		BufferAddr: addr,
		Bytes:      bytes,
		Done:       make(chan struct{}),
	}
	c.inFlight[addr] = tr
	return tr, nil
}

// ReleaseTransferBuffer marks an in-flight transfer completed and frees the buffer address for reuse.
func (c *Communicator) ReleaseTransferBuffer(tr *InFlightTransfer, err error) {
	if tr == nil {
		return
	}
	c.inFlightMu.Lock()
	defer c.inFlightMu.Unlock()

	if c.inFlight[tr.BufferAddr] != tr {
		return
	}
	select {
	case <-tr.Done:
		return
	default:
	}
	tr.Err = err
	close(tr.Done)
	delete(c.inFlight, tr.BufferAddr)
}

// Close stops admission and waits for the transport to finish accepted work.
// A transport Close error retains every in-flight reservation and permits an
// explicit Close retry. Concurrent attempts are serialized. Buffers and native
// handles remain borrowed until Close succeeds; native waits may be unbounded.
func (c *Communicator) Close() error {
	c.closeMu.Lock()
	defer c.closeMu.Unlock()
	c.mu.Lock()
	if c.closed && c.closeErr == nil {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	c.mu.Unlock()
	// Do not hold c.mu while waiting: completion and failure reporting can
	// still need the communicator while accepted native work drains.
	err := c.transport.Close()
	if err != nil {
		c.mu.Lock()
		c.closeErr = err
		c.mu.Unlock()
		return err
	}
	c.inFlightMu.Lock()
	for _, tr := range c.inFlight {
		select {
		case <-tr.Done:
		default:
			if tr.Err == nil {
				tr.Err = ErrCommunicatorClosed
			} else {
				tr.Err = errors.Join(tr.Err, ErrCommunicatorClosed)
			}
			close(tr.Done)
		}
	}
	c.inFlight = nil
	c.inFlightMu.Unlock()
	c.mu.Lock()
	c.closeErr = err
	c.mu.Unlock()
	return err
}
