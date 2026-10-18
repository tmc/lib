package cudatmem

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagroup"
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
// this bounded model. MaxRows and MaxColumns follow PTX ISA 9.4 section
// 9.7.18.1 for sm_100a; target admission remains package policy.
type Capability struct {
	Target     Target
	MaxColumns uint32
	MaxRows    uint32
	ColumnUnit uint32
	RowUnit    uint32
}

func capability(target Target) (Capability, error) {
	if target != SM100A {
		return Capability{}, fmt.Errorf("%w: %s", ErrUnsupportedTarget, target)
	}
	return Capability{Target: target, MaxColumns: 512, MaxRows: 128, ColumnUnit: 32, RowUnit: 1}, nil
}

// Permit models one CTA's tcgen05 allocation permit and the allocation
// sequence issued under it. Allocations made with a permit are owned by that
// permit and can be released only through it.
//
// Permit identity is host-side bookkeeping: holding a *Permit does not prove
// that the corresponding device warp participates in the collective.
// New uses unverified participant counts. NewCollective checks the declared
// issuer identities; neither constructor observes device execution.
type Permit struct {
	participants uint32
	relinquished bool
	lastColumns  uint32 // columns of the most recent allocation; 0 means none
	owner        *Allocator
}

// Allocation is a TMEM region owned by the permit that allocated it. Rows is
// the logical row window; tcgen05.alloc allocates columns across all 128 rows
// (PTX ISA 9.4, section 9.7.18.1.2). Checked users obtain slices and Access
// handles. Legacy users balance BeginUser and CompleteUser before Release.
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

// BeginUser starts an unverified legacy user. Checked allocations require
// Slice.BeginAccess instead.
func (a *Allocation) BeginUser() error {
	if a == nil || a.owner == nil {
		return ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if a.owner.collective != nil {
		return ErrPermission
	}
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

// CompleteUser completes an unverified legacy user. Checked allocations require
// the specific Access.Complete handle instead.
func (a *Allocation) CompleteUser() error {
	if a == nil || a.owner == nil {
		return ErrClosed
	}
	a.owner.mu.Lock()
	defer a.owner.mu.Unlock()
	if a.owner.collective != nil {
		return ErrPermission
	}
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
	collective    *collectiveState
}

type collectiveState struct {
	issuers cudagroup.TMEMCollective
	launch  cudagroup.LaunchRequirements // actual launch, including an ordinary zero cluster
	issued  bool
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

// New constructs the legacy, unverified count-only host model. Its permits
// cannot be used with the checked collective or slice APIs.
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

// AcquirePermit starts a legacy execution with a declared participant count.
// It returns ErrPermission for a checked allocator.
func (a *Allocator) AcquirePermit(participants uint32) (*Permit, error) {
	if a == nil {
		return nil, ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed || a.cap.ColumnUnit == 0 || a.activePermits == nil || a.allocations == nil {
		return nil, ErrClosed
	}
	if a.collective != nil {
		return nil, ErrPermission
	}
	if participants != 32 {
		return nil, ErrParticipation
	}
	if a.permits == ^uint32(0) {
		return nil, ErrBusy
	}
	return a.acquirePermitLocked(participants), nil
}

func (a *Allocator) acquirePermitLocked(participants uint32) *Permit {
	a.permits++
	p := &Permit{participants: participants, owner: a}
	a.activePermits[p] = struct{}{}
	return p
}

// Allocate allocates a legacy logical row window under p. Checked allocators
// return ErrPermission; use AllocateCollective for exact issuer admission.
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
	if a.collective != nil {
		return nil, ErrPermission
	}
	return a.allocateLocked(p, participants, columns, rows)
}

func (a *Allocator) allocateLocked(p *Permit, participants, columns, rows uint32) (*Allocation, error) {
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
	if columns == 0 || rows == 0 || rows > a.cap.MaxRows || columns%a.cap.ColumnUnit != 0 || columns&(columns-1) != 0 || rows%a.cap.RowUnit != 0 || columns > a.cap.MaxColumns {
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
// ErrClosed if x was already released. Checked allocators return ErrPermission.
func (a *Allocator) Release(p *Permit, x *Allocation, participants uint32) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.collective != nil {
		return ErrPermission
	}
	return a.releaseLocked(p, x, participants)
}

func (a *Allocator) releaseLocked(p *Permit, x *Allocation, participants uint32) error {
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
// owner and must still release each one. Checked allocators return ErrPermission.
func (a *Allocator) Relinquish(p *Permit) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.collective != nil {
		return ErrPermission
	}
	return a.relinquishLocked(p)
}

func (a *Allocator) relinquishLocked(p *Permit) error {
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

// NewCollective constructs a checked, nonexclusive group1 model for one CTA
// on sm_100a. The actual grid must be 1x1x1. A zero launch.Cluster denotes an
// ordinary launch: only the validation copy is normalized to a singleton
// cluster so cudagroup can validate its logical CTA rank. This does not change
// launch, request a physical cluster, or prove native execution.
//
// One execution permit may be acquired, once, to retain the CTA's allocation
// history and CTA-wide relinquishment. See PTX ISA 9.4, section 9.7.18.7.1:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html.
func NewCollective(target Target, c cudagroup.TMEMCollective, launch cudagroup.LaunchRequirements) (*Allocator, error) {
	a, err := New(target)
	if err != nil {
		return nil, err
	}
	one := driver.Dim3{X: 1, Y: 1, Z: 1}
	if c.Peers() != cudagroup.TMEMGroup1 || launch.Grid != one || (launch.Cluster != (driver.Dim3{}) && launch.Cluster != one) {
		return nil, ErrParticipation
	}
	logical := launch
	logical.Cluster = one
	if err := c.ValidateLaunch(logical); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParticipation, err)
	}
	a.collective = &collectiveState{issuers: c, launch: launch}
	return a, nil
}

func (a *Allocator) checkedLocked() error {
	if a.closed || a.cap.ColumnUnit == 0 || a.activePermits == nil || a.allocations == nil {
		return ErrClosed
	}
	if a.collective == nil {
		return ErrPermission
	}
	return nil
}

func (a *Allocator) participationLocked(issuers []cudagroup.Issuer, args []uint32) error {
	if err := a.collective.issuers.ValidateParticipation(issuers, args); err != nil {
		return fmt.Errorf("%w: %v", ErrParticipation, err)
	}
	return nil
}

// AcquireCollectivePermit checks the exact declared issuing warp. The permit
// is the allocator's sole execution history; even relinquishment cannot reset it.
func (a *Allocator) AcquireCollectivePermit(issuers []cudagroup.Issuer) (*Permit, error) {
	if a == nil {
		return nil, ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkedLocked(); err != nil {
		return nil, err
	}
	var args [32]uint32
	if err := a.participationLocked(issuers, args[:]); err != nil {
		return nil, err
	}
	if a.collective.issued {
		return nil, ErrPermission
	}
	a.collective.issued = true
	return a.acquirePermitLocked(32), nil
}

// AllocateCollective validates issuing identities and uniform per-issuer
// column counts before allocating a logical row window. Counts must be legal
// allocation units and must not increase during the permit's execution.
func (a *Allocator) AllocateCollective(p *Permit, issuers []cudagroup.Issuer, columns []uint32, rows uint32) (*Allocation, error) {
	if a == nil {
		return nil, ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkedLocked(); err != nil {
		return nil, err
	}
	if err := a.participationLocked(issuers, columns); err != nil {
		return nil, err
	}
	return a.allocateLocked(p, 32, columns[0], rows)
}

// ReleaseCollective checks the original column count and issuing identities.
// The owning permit can deallocate after relinquishment, once all accesses
// complete. The count-only Release method refuses checked allocations.
func (a *Allocator) ReleaseCollective(p *Permit, x *Allocation, issuers []cudagroup.Issuer, columns []uint32) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkedLocked(); err != nil {
		return err
	}
	if err := a.participationLocked(issuers, columns); err != nil {
		return err
	}
	if x == nil || x.owner != a {
		return ErrClosed
	}
	if columns[0] != x.columns {
		return ErrInvalidUnit
	}
	return a.releaseLocked(p, x, 32)
}

// RelinquishCollective checks the issuing identities and permanently revokes
// allocation authority. Existing accesses and owning deallocation remain legal.
func (a *Allocator) RelinquishCollective(p *Permit, issuers []cudagroup.Issuer) error {
	if a == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if err := a.checkedLocked(); err != nil {
		return err
	}
	var args [32]uint32
	if err := a.participationLocked(issuers, args[:]); err != nil {
		return err
	}
	return a.relinquishLocked(p)
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
