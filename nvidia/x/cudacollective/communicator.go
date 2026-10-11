package cudacollective

import (
	"errors"
	"fmt"
	"sync"
)

// Generation represents an incarnation counter for a communicator clique.
// When a rank fails or a communicator is aborted, the generation is marked failed.
// Subsequent operations or re-creation with that generation fail closed.
type Generation uint64

// Communicator is the coordinated control plane interface for a rank.
type Communicator struct {
	mu         sync.RWMutex
	rank       int
	membership Membership
	generation Generation
	transport  Transport
	closed     bool
	closeDone  chan struct{}
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
	Err        error
}

// CommunicatorConfig provides parameters for creating a Communicator.
type CommunicatorConfig struct {
	Rank       int
	Membership Membership
	Generation Generation
	Transport  Transport
}

// NewCommunicator creates and validates a Communicator for a specific rank and generation.
func NewCommunicator(cfg CommunicatorConfig) (*Communicator, error) {
	if cfg.Membership.WorldSize() <= 0 {
		return nil, ErrInvalidWorldSize
	}
	if cfg.Rank < 0 || cfg.Rank >= cfg.Membership.WorldSize() {
		return nil, fmt.Errorf("%w: rank %d out of world bounds [0, %d)", ErrInvalidRank, cfg.Rank, cfg.Membership.WorldSize())
	}
	if cfg.Generation == 0 {
		return nil, fmt.Errorf("%w: generation 0 is invalid", ErrCommunicatorGeneration)
	}
	if cfg.Transport == nil {
		return nil, errors.New("cudacollective: transport cannot be nil")
	}

	participants := make(map[int]bool, cfg.Membership.WorldSize())
	for i := 0; i < cfg.Membership.WorldSize(); i++ {
		participants[i] = true
	}

	return &Communicator{
		rank:               cfg.Rank,
		closeDone:          make(chan struct{}),
		membership:         cfg.Membership,
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

// MarkFailed transitions the communicator to a permanently failed state with err.
// Any future operation on this communicator will immediately return ErrCommunicatorGeneration.
func (c *Communicator) MarkFailed(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.failed = true
	if c.failErr == nil {
		c.failErr = err
	}
}

// ReportLostParticipant records that a remote participant dropped out or failed.
func (c *Communicator) ReportLostParticipant(rank int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.activeParticipants, rank)
	c.failed = true
	c.failErr = fmt.Errorf("%w: rank %d", ErrLostParticipant, rank)
}

// CheckAlive verifies that all members are present and the generation is active.
func (c *Communicator) CheckAlive() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return ErrCommunicatorClosed
	}
	if c.failed {
		return fmt.Errorf("%w: %w", ErrCommunicatorGeneration, c.failErr)
	}
	if len(c.activeParticipants) < c.membership.WorldSize() {
		return fmt.Errorf("%w: missing ranks", ErrLostParticipant)
	}
	return nil
}

// AcquireTransferBuffer marks a buffer address as actively transferring in a double-buffered slot.
// If the buffer is already registered in flight, it returns ErrBufferInUse.
func (c *Communicator) AcquireTransferBuffer(addr uintptr, bytes int, slot int) (*InFlightTransfer, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.closed {
		return nil, ErrCommunicatorClosed
	}
	if c.failed {
		return nil, fmt.Errorf("%w: %w", ErrCommunicatorGeneration, c.failErr)
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

	c.inFlightMu.Lock()
	defer c.inFlightMu.Unlock()

	if existing, found := c.inFlight[addr]; found {
		select {
		case <-existing.Done:
			// Completed, safe to replace
		default:
			return nil, fmt.Errorf("%w: address 0x%x in slot %d", ErrBufferInUse, addr, existing.Slot)
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
// For native transports this wait is only bounded when native work completes.
func (c *Communicator) Close() error {
	c.mu.Lock()
	if c.closed {
		done := c.closeDone
		c.mu.Unlock()
		<-done
		c.mu.RLock()
		defer c.mu.RUnlock()
		return c.closeErr
	}
	c.closed = true
	c.mu.Unlock()
	// Do not hold c.mu while waiting: completion and failure reporting can
	// still need the communicator while accepted native work drains.
	err := c.transport.Close()
	c.inFlightMu.Lock()
	for _, tr := range c.inFlight {
		select {
		case <-tr.Done:
		default:
			tr.Err = ErrCommunicatorClosed
			close(tr.Done)
		}
	}
	c.inFlight = nil
	c.inFlightMu.Unlock()
	c.mu.Lock()
	c.closeErr = err
	close(c.closeDone)
	c.mu.Unlock()
	return err
}
