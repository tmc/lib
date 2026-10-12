package cudatma

import (
	"errors"
	"fmt"
	"sync"
)

// Errors reported by [DescriptorSlot].
var (
	// ErrInFlight reports an update to a descriptor that transfers issued
	// from it are still reading.
	ErrInFlight = errors.New("descriptor has transfers in flight")

	// ErrStaleGeneration reports a handle or issue request naming a
	// generation the slot has since replaced.
	ErrStaleGeneration = errors.New("stale descriptor generation")

	// ErrFutureGeneration reports an issue request naming a generation the
	// slot has not reached.
	ErrFutureGeneration = errors.New("future descriptor generation")

	// ErrDescriptorMismatch reports a transfer whose descriptor differs from
	// the one the slot holds at the requested generation.
	ErrDescriptorMismatch = errors.New("descriptor differs from slot")

	// ErrNotInFlight reports a completion for a transfer that was already
	// completed.
	ErrNotInFlight = errors.New("transfer not in flight")

	// ErrForeignHandle reports a handle that the slot did not issue,
	// including the zero TransferHandle.
	ErrForeignHandle = errors.New("handle not issued by this slot")
)

// A TransferHandle names one transfer issued by a [DescriptorSlot]. Only
// [DescriptorSlot.Issue] creates a usable handle; the zero TransferHandle
// names no transfer.
type TransferHandle struct {
	slot *DescriptorSlot
	gen  uint64
	seq  uint64
}

// Generation returns the descriptor generation the transfer was issued
// under, or 0 for the zero TransferHandle.
func (h TransferHandle) Generation() uint64 { return h.gen }

// A DescriptorSlot owns one [Descriptor] and the transfers issued through it.
// Each descriptor the slot holds has a generation, starting at 1. Issue
// records a transfer as in flight under the current generation; Complete
// retires it. Update installs a new descriptor and generation, and is refused
// while any transfer issued through the slot is in flight.
//
// The slot is opt-in. It governs only transfers issued through it: a
// [Descriptor] or [Transfer] value used directly, and any tensor map built
// with [Descriptor.EncodeTiled], are outside it and can be changed or reused
// without the slot knowing.
//
// DescriptorSlot is a host model of descriptor ownership. It does not observe
// device execution: Complete is called by the host once it has evidence, such
// as an mbarrier phase, that the transfer finished. Whether native TMA
// transfers in fact stop reading the descriptor by then is a native gate this
// type does not check.
//
// A DescriptorSlot is safe for concurrent use. Use [NewDescriptorSlot]; the
// zero value is not usable.
type DescriptorSlot struct {
	mu       sync.Mutex
	desc     Descriptor
	gen      uint64
	next     uint64
	inFlight map[uint64]bool
}

// NewDescriptorSlot returns a slot owning d at generation 1.
func NewDescriptorSlot(d Descriptor) (*DescriptorSlot, error) {
	if err := d.Validate(); err != nil {
		return nil, fmt.Errorf("new descriptor slot: %w", err)
	}
	return &DescriptorSlot{desc: d, gen: 1, inFlight: make(map[uint64]bool)}, nil
}

// Descriptor returns the descriptor of the current generation.
func (s *DescriptorSlot) Descriptor() Descriptor {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.desc
}

// Current returns the current descriptor and its generation, read together.
// Callers building a transfer should take both from one call: separate
// calls to [DescriptorSlot.Descriptor] and [DescriptorSlot.Generation] can
// straddle an Update and pair a descriptor with the wrong generation.
func (s *DescriptorSlot) Current() (Descriptor, uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.desc, s.gen
}

// Generation returns the current descriptor generation.
func (s *DescriptorSlot) Generation() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gen
}

// InFlight returns the number of transfers issued and not yet completed.
func (s *DescriptorSlot) InFlight() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.inFlight)
}

// Issue validates t and records it as in flight under generation, which must
// be the slot's current generation, as returned with the descriptor by
// [DescriptorSlot.Current]. A request for an earlier generation is refused
// with [ErrStaleGeneration], even if the descriptor was since updated to an
// equal value, and one for a later generation with [ErrFutureGeneration].
// The transfer's descriptor must equal the current one, or Issue returns an
// error wrapping [ErrDescriptorMismatch].
//
// The generation check is only as strong as the caller's value: a
// generation read again at issue time, rather than saved with the
// descriptor, will match whatever the slot holds then.
func (s *DescriptorSlot) Issue(generation uint64, t Transfer) (TransferHandle, error) {
	if err := t.Validate(); err != nil {
		return TransferHandle{}, fmt.Errorf("issue transfer: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if generation > s.gen {
		return TransferHandle{}, fmt.Errorf("issue transfer: generation %d, slot is at %d: %w", generation, s.gen, ErrFutureGeneration)
	}
	if generation < s.gen {
		return TransferHandle{}, fmt.Errorf("issue transfer: generation %d, slot is at %d: %w", generation, s.gen, ErrStaleGeneration)
	}
	if t.Descriptor != s.desc {
		return TransferHandle{}, fmt.Errorf("issue transfer: generation %d: %w", s.gen, ErrDescriptorMismatch)
	}
	h := TransferHandle{slot: s, gen: s.gen, seq: s.next}
	s.next++
	s.inFlight[h.seq] = true
	return h, nil
}

// Complete retires the transfer h names. It returns an error wrapping
// [ErrForeignHandle] if this slot did not issue h, [ErrStaleGeneration] if h
// was issued under an earlier generation, and [ErrNotInFlight] if h was
// already completed.
func (s *DescriptorSlot) Complete(h TransferHandle) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if h.slot != s {
		return fmt.Errorf("complete transfer: %w", ErrForeignHandle)
	}
	if h.gen != s.gen {
		return fmt.Errorf("complete transfer: generation %d, slot is at %d: %w", h.gen, s.gen, ErrStaleGeneration)
	}
	if !s.inFlight[h.seq] {
		return fmt.Errorf("complete transfer %d: %w", h.seq, ErrNotInFlight)
	}
	delete(s.inFlight, h.seq)
	return nil
}

// Update installs d as a new generation. It returns an error wrapping
// [ErrInFlight], and leaves the slot unchanged, while any transfer issued
// through the slot has not completed. Transfer sequence numbers continue
// across generations, so no handle is ever reissued.
func (s *DescriptorSlot) Update(d Descriptor) error {
	if err := d.Validate(); err != nil {
		return fmt.Errorf("update descriptor: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if n := len(s.inFlight); n > 0 {
		return fmt.Errorf("update descriptor: %d transfers: %w", n, ErrInFlight)
	}
	s.desc = d
	s.gen++
	return nil
}
