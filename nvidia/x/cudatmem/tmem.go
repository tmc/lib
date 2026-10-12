package cudatmem

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudagroup"
	"sort"
	"sync"
)

var (
	ErrUnsupportedTarget = errors.New("cudatmem: target does not support tmem")
	ErrInvalidUnit       = errors.New("cudatmem: invalid allocation unit")
	ErrPermission        = errors.New("cudatmem: allocation permission required")
	ErrParticipation     = errors.New("cudatmem: collective participation mismatch")
	ErrBusy              = errors.New("cudatmem: allocation still has asynchronous users")
	ErrClosed            = errors.New("cudatmem: allocator is closed")
)

// Target is an exact architecture token, not a marketing family name.
type Target string

const (
	SM100A Target = "sm_100a"
	SM121A Target = "sm_121a"
)

func ParseTarget(s string) (Target, error) {
	switch Target(s) {
	case SM100A:
		return SM100A, nil
	case SM121A:
		return SM121A, ErrUnsupportedTarget
	default:
		return "", fmt.Errorf("%w: %q", ErrUnsupportedTarget, s)
	}
}

// Capability describes the target-specific hardware allocation limits used by
// this bounded model. Values are package policy, not a universal ISA promise.
type Capability struct {
	Target     Target
	MaxColumns uint32
	ColumnUnit uint32
	RowUnit    uint32
}

func capability(target Target) (Capability, error) {
	if target != SM100A {
		return Capability{}, fmt.Errorf("%w: %s", ErrUnsupportedTarget, target)
	}
	return Capability{Target: target, MaxColumns: 512, ColumnUnit: 32, RowUnit: 1}, nil
}

// Permit models one CTA's tcgen05 allocation permit and the allocation
// sequence issued under it. Allocations made with a permit are owned by that
// permit and can be released only through it.
//
// Permit identity is host-side bookkeeping: holding a *Permit does not prove
// that the corresponding device warp participates in the collective.
// Participation is checked only as the declared participant count.
type Permit struct {
	participants uint32
	relinquished bool
	lastColumns  uint32 // columns of the most recent allocation; 0 means none
	owner        *Allocator
}

// Allocation is a TMEM region owned by the permit that allocated it. Users
// must call CompleteUser once for each asynchronous user before Release.
type Allocation struct {
	columns, rows, offset uint32
	users                 uint32
	released              bool
	owner                 *Allocator
	permit                *Permit
}

func (a *Allocation) Columns() uint32 { return a.columns }
func (a *Allocation) Rows() uint32    { return a.rows }
func (a *Allocation) Offset() uint32  { return a.offset }

func (a *Allocation) BeginUser() error {
	if a == nil || a.owner == nil {
		return ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if a.released {
		return ErrClosed
	}
	if _, ok := a.owner.allocations[a]; !ok {
		return ErrClosed
	}
	if a.users == ^uint32(0) {
		return ErrBusy
	}
	a.users++
	return nil
}
func (a *Allocation) CompleteUser() error {
	if a == nil || a.owner == nil {
		return ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if a.released {
		return ErrClosed
	}
	if _, ok := a.owner.allocations[a]; !ok {
		return ErrClosed
	}
	if a.users == 0 {
		return ErrBusy
	}
	a.users--
	return nil
}

// Allocator owns a finite target-specific TMEM column pool.
type Allocator struct {
	mu            sync.Mutex
	cap           Capability
	next          uint32
	free          []columnRange
	permits       uint32
	allocations   map[*Allocation]struct{}
	activePermits map[*Permit]struct{}
	closed        bool
}

// ValidateCollective consumes cudagroup's exact TMEM participation contract.
func ValidateCollective(c cudagroup.TMEMCollective, issuers []cudagroup.Issuer, args []uint32) error {
	return c.ValidateParticipation(issuers, args)
}

type columnRange struct{ offset, columns uint32 }

// checkPermit reports whether p belongs to a. It reads only p.owner, which
// is fixed when p is created, so callers must use it before reading any other
// permit field: those fields are guarded by p.owner's lock, not a's.
func (a *Allocator) checkPermit(p *Permit) error {
	if p == nil {
		return fmt.Errorf("%w: nil permit", ErrPermission)
	}
	if p.owner != a {
		return fmt.Errorf("%w: permit from another allocator", ErrPermission)
	}
	return nil
}

func New(target Target) (*Allocator, error) {
	c, err := capability(target)
	if err != nil {
		return nil, err
	}
	return &Allocator{cap: c, allocations: make(map[*Allocation]struct{}), activePermits: make(map[*Permit]struct{})}, nil
}
func (a *Allocator) Capability() Capability {
	if a == nil {
		return Capability{}
	}
	return a.cap
}
func (a *Allocator) AcquirePermit(participants uint32) (*Permit, error) {
	if a == nil {
		return nil, ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed || a.cap.ColumnUnit == 0 || a.activePermits == nil || a.allocations == nil {
		return nil, ErrClosed
	}
	if participants != 32 {
		return nil, ErrParticipation
	}
	if a.permits == ^uint32(0) {
		return nil, ErrBusy
	}
	a.permits++
	p := &Permit{participants: participants, owner: a}
	a.activePermits[p] = struct{}{}
	return p, nil
}

// Allocate allocates columns by rows of TMEM under p.
//
// As in PTX ISA 9.4 tcgen05.alloc, the column count must not increase between
// two allocations in the permit's execution order. The order covers every
// successful allocation under p, including ones already released, and
// violations return an error wrapping ErrInvalidUnit. Allocate returns
// ErrPermission after p has been relinquished.
func (a *Allocator) Allocate(p *Permit, participants, columns, rows uint32) (*Allocation, error) {
	if a == nil {
		return nil, ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed || a.cap.ColumnUnit == 0 || a.cap.RowUnit == 0 {
		return nil, ErrClosed
	}
	if err := a.checkPermit(p); err != nil {
		return nil, err
	}
	if p.relinquished {
		return nil, fmt.Errorf("%w: permit relinquished", ErrPermission)
	}
	if _, ok := a.activePermits[p]; !ok {
		return nil, fmt.Errorf("%w: unknown permit", ErrPermission)
	}
	if participants != p.participants {
		return nil, ErrParticipation
	}
	if columns == 0 || rows == 0 || rows > 128 || columns%a.cap.ColumnUnit != 0 || columns&(columns-1) != 0 || rows%a.cap.RowUnit != 0 || columns > a.cap.MaxColumns {
		return nil, ErrInvalidUnit
	}
	if p.lastColumns != 0 && columns > p.lastColumns {
		return nil, fmt.Errorf("%w: %d columns after %d", ErrInvalidUnit, columns, p.lastColumns)
	}
	offset := a.next
	for i, r := range a.free {
		if r.columns < columns {
			continue
		}
		offset = r.offset
		if r.columns == columns {
			a.free = append(a.free[:i], a.free[i+1:]...)
		} else {
			a.free[i] = columnRange{offset: r.offset + columns, columns: r.columns - columns}
		}
		break
	}
	if offset == a.next {
		if a.next > a.cap.MaxColumns-columns {
			return nil, ErrInvalidUnit
		}
		a.next += columns
	}
	x := &Allocation{columns: columns, rows: rows, offset: offset, owner: a, permit: p}
	a.allocations[x] = struct{}{}
	p.lastColumns = columns
	return x, nil
}

// Release deallocates x. Only the permit that allocated x may release it,
// including after that permit has been relinquished; another permit gets
// ErrPermission. Release returns ErrBusy while x has asynchronous users and
// ErrClosed if x was already released.
func (a *Allocator) Release(p *Permit, x *Allocation, participants uint32) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkPermit(p); err != nil {
		return err
	}
	if x == nil || x.owner != a || x.released {
		return ErrClosed
	}
	if _, ok := a.allocations[x]; !ok {
		return ErrClosed
	}
	if x.permit != p {
		return fmt.Errorf("%w: allocation owned by another permit", ErrPermission)
	}
	if participants != p.participants {
		return ErrParticipation
	}
	if x.users != 0 {
		return ErrBusy
	}
	x.released = true
	delete(a.allocations, x)
	a.free = append(a.free, columnRange{offset: x.offset, columns: x.columns})
	sort.Slice(a.free, func(i, j int) bool { return a.free[i].offset < a.free[j].offset })
	merged := a.free[:0]
	for _, r := range a.free {
		if len(merged) > 0 && merged[len(merged)-1].offset+merged[len(merged)-1].columns == r.offset {
			merged[len(merged)-1].columns += r.columns
		} else {
			merged = append(merged, r)
		}
	}
	a.free = merged
	return nil
}

// Relinquish revokes p's authority to allocate, as tcgen05.relinquish_alloc_permit
// does for a CTA. It does not release p's live allocations: p remains their
// owner and must still release each one.
func (a *Allocator) Relinquish(p *Permit) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkPermit(p); err != nil {
		return err
	}
	if p.relinquished {
		return fmt.Errorf("%w: permit already relinquished", ErrPermission)
	}
	if _, ok := a.activePermits[p]; !ok {
		return fmt.Errorf("%w: unknown permit", ErrPermission)
	}
	p.relinquished = true
	delete(a.activePermits, p)
	if a.permits > 0 {
		a.permits--
	}
	return nil
}

// Close closes the allocator. It returns ErrBusy while any permit has not
// been relinquished or any allocation is live, including an allocation whose
// permit has been relinquished. Closing a nil or closed allocator is a no-op.
func (a *Allocator) Close() error {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	if len(a.allocations) != 0 || a.permits != 0 {
		return ErrBusy
	}
	a.closed = true
	return nil
}
