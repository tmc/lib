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

// Permit is the collective allocation authority. It must be held by the
// participating CTA group for the entire allocation/deallocation protocol.
type Permit struct {
	participants uint32
	released     bool
	owner        *Allocator
}

// Allocation is an owned TMEM region. Users must call CompleteUser once for
// each asynchronous user before Release.
type Allocation struct {
	columns, rows, offset uint32
	users                 uint32
	released              bool
	owner                 *Allocator
}

func (a Allocation) Columns() uint32 { return a.columns }
func (a Allocation) Rows() uint32    { return a.rows }
func (a Allocation) Offset() uint32  { return a.offset }

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
func (a *Allocator) Allocate(p *Permit, participants, columns, rows uint32) (*Allocation, error) {
	if a == nil {
		return nil, ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed || a.cap.ColumnUnit == 0 || a.cap.RowUnit == 0 {
		return nil, ErrClosed
	}
	if p == nil || p.released || p.owner != a {
		return nil, ErrPermission
	}
	if _, ok := a.activePermits[p]; !ok {
		return nil, ErrPermission
	}
	if participants != p.participants {
		return nil, ErrParticipation
	}
	if columns == 0 || rows == 0 || rows > 128 || columns%a.cap.ColumnUnit != 0 || columns&(columns-1) != 0 || rows%a.cap.RowUnit != 0 || columns > a.cap.MaxColumns {
		return nil, ErrInvalidUnit
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
	x := &Allocation{columns: columns, rows: rows, offset: offset, owner: a}
	a.allocations[x] = struct{}{}
	return x, nil
}
func (a *Allocator) Release(p *Permit, x *Allocation, participants uint32) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if p == nil || p.released || p.owner != a {
		return ErrPermission
	}
	if _, ok := a.activePermits[p]; !ok {
		return ErrPermission
	}
	if participants != p.participants {
		return ErrParticipation
	}
	if x == nil || x.owner != a || x.released {
		return ErrClosed
	}
	if _, ok := a.allocations[x]; !ok {
		return ErrClosed
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
func (a *Allocator) Relinquish(p *Permit) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if p == nil || p.released || p.owner != a {
		return ErrPermission
	}
	if _, ok := a.activePermits[p]; !ok {
		return ErrPermission
	}
	if len(a.allocations) != 0 {
		return ErrBusy
	}
	p.released = true
	delete(a.activePermits, p)
	if a.permits > 0 {
		a.permits--
	}
	return nil
}
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
