package cudatmem

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudagroup"
	"github.com/tmc/lib/nvidia/x/cudalayout"
)

// Row counts logical TMEM rows, each containing one 32-bit cell per column.
type Row uint32

// Column counts logical TMEM columns, not bytes or hardware allocation units.
type Column uint32

// Address is a modeled cell coordinate in the allocator's column pool. It is
// not a device pointer or an encoded taddr, and does not grant access authority.
type Address struct {
	Row    Row
	Column Column
}

// Slice is an immutable logical view of a checked allocation. Overlapping
// views are allowed; they do not establish disjointness or device ordering.
// Copies share the parent's lifetime. An idle view does not prevent Release.
// The zero value is invalid.
type Slice struct {
	allocation    *Allocation
	row, column   uint32
	rows, columns uint32
}

func (a *Allocation) liveLocked() error {
	if a.released || a.owner.closed {
		return ErrClosed
	}
	if _, ok := a.owner.allocations[a]; !ok {
		return ErrClosed
	}
	if a.owner.collective == nil {
		return ErrPermission
	}
	return nil
}

func within(start, count, limit uint32) bool {
	return count != 0 && start <= limit && count <= limit-start
}

// Slice selects a positive rectangular view within a's logical row window
// and columns. The origin is relative to a. Legacy allocations are refused.
func (a *Allocation) Slice(row Row, column Column, rows, columns uint32) (Slice, error) {
	if a == nil || a.owner == nil {
		return Slice{}, ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if err := a.liveLocked(); err != nil {
		return Slice{}, err
	}
	if !within(uint32(row), rows, a.rows) || !within(uint32(column), columns, a.columns) {
		return Slice{}, ErrInvalidUnit
	}
	return Slice{a, uint32(row), uint32(column), rows, columns}, nil
}

// Sub selects a positive rectangular subview, relative to s's origin.
func (s Slice) Sub(row Row, column Column, rows, columns uint32) (Slice, error) {
	a := s.allocation
	if a == nil || a.owner == nil {
		return Slice{}, ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if err := a.liveLocked(); err != nil {
		return Slice{}, err
	}
	if !within(uint32(row), rows, s.rows) || !within(uint32(column), columns, s.columns) {
		return Slice{}, ErrInvalidUnit
	}
	return Slice{a, s.row + uint32(row), s.column + uint32(column), rows, columns}, nil
}

// Address returns the modeled pool coordinate of an element relative to s.
// It validates lifetime and bounds; it does not authorize a scalar instruction.
func (s Slice) Address(row Row, column Column) (Address, error) {
	a := s.allocation
	if a == nil || a.owner == nil {
		return Address{}, ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if err := a.liveLocked(); err != nil {
		return Address{}, err
	}
	if uint32(row) >= s.rows || uint32(column) >= s.columns {
		return Address{}, ErrInvalidUnit
	}
	return Address{Row: Row(s.row + uint32(row)), Column: Column(a.offset + s.column + uint32(column))}, nil
}

// Access owns one declared asynchronous use of a slice. Copies share completion
// state; Complete consumes that use at most once. The zero value is invalid.
// Access neither issues an instruction nor observes its device completion.
type Access struct{ state *accessState }

type accessState struct {
	allocation *Allocation
	complete   bool // guarded by allocation.owner.mu
}

// BeginAccess admits the unmodified .32x32b.x1 or .x2 .b32 load/store footprint:
// layout must be dense, unswizzled, 32 rows by one or two 32-bit columns, and
// match s. Other shapes, packing, register mappings, and native taddr encoding
// are unsupported. Tile.Owner and Fragment are not used as hardware mappings.
//
// warp is the linear warp index in the actual launch block. All its 32 declared
// issuers must be present exactly once in CTA 0, with the same modeled base at
// s's origin. The row chunk is 32*(warp%4) through 32*(warp%4)+31. These rules
// follow PTX ISA 9.4 sections 9.7.18.2.3.1.1 and 9.7.18.8.1, .3, .4:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
//
// Success holds the parent live until Complete, including after its permit is
// relinquished. Callers still arrange actual uniform execution, synchronization
// and wait completion; declared metadata alone is not evidence of those events.
func (s Slice) BeginAccess(warp uint32, layout cudalayout.Tile, issuers []cudagroup.Issuer, bases []Address) (*Access, error) {
	a := s.allocation
	if a == nil || a.owner == nil {
		return nil, ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if err := a.liveLocked(); err != nil {
		return nil, err
	}
	if s.rows != 32 || (s.columns != 1 && s.columns != 2) || layout.Rows() != 32 || layout.Cols() != int(s.columns) || layout.ElementBytes() != 4 || layout.StrideElements() != int(s.columns) || layout.Swizzle() != cudalayout.SwizzleNone {
		return nil, ErrInvalidUnit
	}
	if _, err := layout.Index(31, int(s.columns)-1); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidUnit, err)
	}
	block := a.owner.collective.launch.Block
	threads := uint64(block.X) * uint64(block.Y) * uint64(block.Z) // constructor checked CTA size
	first := uint64(warp) * 32
	if first+32 > threads || s.row != 32*(warp%4) || len(issuers) != 32 || len(bases) != 32 {
		return nil, ErrParticipation
	}
	base := Address{Row: Row(s.row), Column: Column(a.offset + s.column)}
	var seen uint32
	for i, issuer := range issuers {
		if issuer.CTA != 0 || issuer.Thread < first || issuer.Thread >= first+32 || bases[i] != base {
			return nil, ErrParticipation
		}
		bit := uint32(1) << uint32(issuer.Thread-first)
		if seen&bit != 0 {
			return nil, ErrParticipation
		}
		seen |= bit
	}
	if a.users == ^uint32(0) {
		return nil, ErrBusy
	}
	a.users++
	return &Access{state: &accessState{allocation: a}}, nil
}

// Complete declares the caller's asynchronous operation complete. It is
// idempotent even across copies and concurrent calls, and does not consume
// another access's use. It does not perform a device wait.
func (a *Access) Complete() error {
	if a == nil || a.state == nil || a.state.allocation == nil {
		return ErrClosed
	}
	s := a.state
	x := s.allocation
	x.owner.mu.Lock()
	defer x.owner.mu.Unlock()
	if s.complete {
		return nil
	}
	s.complete = true
	x.users--
	return nil
}
