package cudagroup

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
)

// Kind identifies an execution group. A warpgroup is four warps (128 threads).
type Kind uint8

const (
	Thread Kind = iota + 1
	Warp
	WarpGroup
	CTA
	Cluster
)

func (k Kind) String() string {
	switch k {
	case Thread:
		return "thread"
	case Warp:
		return "warp"
	case WarpGroup:
		return "warpgroup"
	case CTA:
		return "cta"
	case Cluster:
		return "cluster"
	default:
		return fmt.Sprintf("kind(%d)", k)
	}
}

// count returns the number of elements in d.
func count(d driver.Dim3) (uint64, error) {
	if d.X == 0 || d.Y == 0 || d.Z == 0 {
		return 0, errors.New("group dimensions must be nonzero")
	}
	n := uint64(d.X) * uint64(d.Y)
	if n > math.MaxUint64/uint64(d.Z) {
		return 0, errors.New("group dimensions overflow")
	}
	return n * uint64(d.Z), nil
}

// Group is an immutable execution membership description.
type Group struct {
	kind   Kind
	dims   driver.Dim3
	parent *Group
	base   uint64
}

// New constructs a group. Warp and warpgroup dimensions are fixed by CUDA.
func New(kind Kind, dims driver.Dim3) (Group, error) {
	if kind < Thread || kind > Cluster {
		return Group{}, errors.New("unsupported execution group kind")
	}
	n, err := count(dims)
	if err != nil {
		return Group{}, err
	}
	switch kind {
	case Thread:
		if dims.X != 1 || dims.Y != 1 || dims.Z != 1 {
			return Group{}, errors.New("thread group must have one participant")
		}
	case Warp:
		if dims.X != 32 || dims.Y != 1 || dims.Z != 1 {
			return Group{}, errors.New("warp group must have 32 participants")
		}
	case WarpGroup:
		if dims.X != 128 || dims.Y != 1 || dims.Z != 1 {
			return Group{}, errors.New("warpgroup must have 128 participants")
		}
	case CTA:
		if n > 1024 || dims.X > 1024 || dims.Y > 1024 || dims.Z > 64 {
			return Group{}, errors.New("cta exceeds supported block dimensions")
		}
	case Cluster:
		if n > 8 {
			return Group{}, errors.New("cluster exceeds portable eight-cta subset")
		}
	}
	return Group{kind: kind, dims: dims}, nil
}

// Kind returns the execution membership kind.
func (g Group) Kind() Kind { return g.kind }

// Dims returns thread dimensions, or CTA dimensions for a cluster.
func (g Group) Dims() driver.Dim3 { return g.dims }

// Size returns the number of participants, or zero for an invalid group.
func (g Group) Size() uint64 { n, _ := count(g.dims); return n }

// Rank returns the linear participant rank for a group-local coordinate.
// A subgroup's ranks are relative to its parent CTA; other ranks start at zero.
func (g Group) Rank(index driver.Dim3) (uint64, error) {
	if g.kind == 0 {
		return 0, errors.New("zero execution group")
	}
	if index.X >= g.dims.X || index.Y >= g.dims.Y || index.Z >= g.dims.Z {
		return 0, errors.New("participant coordinate outside group")
	}
	return g.base + uint64(index.Z)*uint64(g.dims.Y)*uint64(g.dims.X) + uint64(index.Y)*uint64(g.dims.X) + uint64(index.X), nil
}

// Contains reports whether index is a member of g.
func (g Group) Contains(index driver.Dim3) bool { _, err := g.Rank(index); return err == nil }

// WarpGroupForCTA returns the warpgroup containing warp in a CTA.
func WarpGroupForCTA(cta Group, warp uint32) (Group, error) {
	if cta.kind != CTA {
		return Group{}, errors.New("warpgroup parent must be a cta")
	}
	warps := cta.Size() / 32
	if uint64(warp) >= warps || uint64(warp/4)*4+3 >= warps {
		return Group{}, errors.New("cta does not contain a complete warpgroup")
	}
	g, _ := New(WarpGroup, driver.Dim3{X: 128, Y: 1, Z: 1})
	g.parent = &cta
	g.base = uint64(warp/4) * 128
	return g, nil
}

// MemoryScope is a visibility/ordering scope, not an execution membership.
type MemoryScope uint8

const (
	MemoryThread MemoryScope = iota + 1
	MemoryCTA
	MemoryCluster
)

func (s MemoryScope) String() string {
	switch s {
	case MemoryThread:
		return "thread"
	case MemoryCTA:
		return "cta"
	case MemoryCluster:
		return "cluster"
	default:
		return fmt.Sprintf("memory-scope(%d)", s)
	}
}

// CollectiveKind identifies an operation-specific synchronization contract.
type CollectiveKind uint8

const (
	Rendezvous CollectiveKind = iota + 1
	ReduceSum
	Broadcast
	// WarpgroupUniformity checks uniform issue; it admits no instruction.
	WarpgroupUniformity
)

// Collective describes participants, required uniformity and memory scope.
type Collective struct {
	Kind    CollectiveKind
	Group   Group
	Scope   MemoryScope
	Uniform bool
}

// NewCollective describes warp or CTA rendezvous, warp reduction or broadcast,
// or warpgroup uniform issue. Transaction barriers are not modeled here.
func NewCollective(kind CollectiveKind, group Group, scope MemoryScope) (Collective, error) {
	if kind < Rendezvous || kind > WarpgroupUniformity {
		return Collective{}, errors.New("unsupported collective kind")
	}
	if group.kind == 0 {
		return Collective{}, errors.New("collective requires an execution group")
	}
	if scope < MemoryThread || scope > MemoryCluster {
		return Collective{}, errors.New("unsupported memory scope")
	}
	if (kind == Rendezvous && group.kind != Warp && group.kind != CTA) ||
		((kind == ReduceSum || kind == Broadcast) && group.kind != Warp) ||
		(kind == WarpgroupUniformity && group.kind != WarpGroup) {
		return Collective{}, errors.New("operation does not support execution group")
	}
	return Collective{Kind: kind, Group: group, Scope: scope, Uniform: true}, nil
}

// ValidateParticipation checks the exact participant count, catching dropped
// participants before a generated collective is launched.
func (c Collective) ValidateParticipation(participants uint64) error {
	if _, err := NewCollective(c.Kind, c.Group, c.Scope); err != nil {
		return err
	}
	if !c.Uniform {
		return errors.New("collective requires uniform issue")
	}
	if participants != c.Group.Size() {
		return fmt.Errorf("collective participation: got %d participants, want %d", participants, c.Group.Size())
	}
	return nil
}

// ValidateRanks checks that every participant rank appears exactly once.
// A count alone cannot detect a duplicate lane replacing a missing lane.
func (c Collective) ValidateRanks(ranks []uint64) error {
	if err := c.ValidateParticipation(uint64(len(ranks))); err != nil {
		return err
	}
	seen := make(map[uint64]struct{}, len(ranks))
	for _, rank := range ranks {
		if rank >= c.Group.base+c.Group.Size() || rank < c.Group.base {
			return errors.New("collective participant rank outside group")
		}
		if _, ok := seen[rank]; ok {
			return errors.New("collective participant rank repeated")
		}
		seen[rank] = struct{}{}
	}
	return nil
}

// ValidateUniform checks one operation argument per participant. All values
// must agree. Instruction-specific value ranges remain the owner's contract.
func (c Collective) ValidateUniform(ranks []uint64, values []uint32) error {
	if err := c.ValidateRanks(ranks); err != nil {
		return err
	}
	if len(values) != len(ranks) {
		return errors.New("uniform argument count differs from participation")
	}
	for _, v := range values {
		if v != values[0] {
			return errors.New("collective argument is not uniform")
		}
	}
	return nil
}
