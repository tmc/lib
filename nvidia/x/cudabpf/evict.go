package cudabpf

import "fmt"

// A BlockState is the KV cache's own view of whether a block is still
// referenced. The signature is cudakv's: block ownership, allocation and
// refcounting stay there, and this package supplies events and decisions
// rather than holding any of it. It is an interface so that nothing here
// imports the KV cache, and so a parity test can drive both sides from the
// same allocator.
type BlockState interface {
	// Holders returns how many sequences reference a block. An unknown
	// block is an error, not zero holders -- zero holders means the block
	// is free, which is the opposite claim.
	Holders(block int) (int, error)
}

// A HolderPolicy places a root chunk according to whether the KV block behind
// it is still referenced. It is the reference placement policy: unreferenced
// blocks are moved toward the unused list so the driver reaches them first,
// referenced ones toward the used list so it reaches them last.
//
// It expresses a preference and nothing more. The driver keeps final
// authority, the request may be refused or collide, and a block the policy
// wanted kept can still be evicted.
type HolderPolicy struct {
	State BlockState

	// Block maps the driver's view of a root chunk to a KV block. It
	// reports false for a chunk that is not KV cache at all, and such a
	// chunk gets no opinion rather than a guessed one.
	Block func(PMMSnapshot) (int, bool)

	// StepCost is charged per invocation against the fault budget.
	StepCost int
}

func (p *HolderPolicy) cost() int {
	if p.StepCost <= 0 {
		return 1
	}
	return p.StepCost
}

// Decide returns the placement for one observed chunk.
func (p *HolderPolicy) Decide(s PMMSnapshot, b *Budget) (Reorder, bool, error) {
	if b != nil {
		if err := b.Step(p.cost()); err != nil {
			return Reorder{}, false, err
		}
	}
	if p.State == nil || p.Block == nil {
		return Reorder{}, false, nil
	}
	block, ok := p.Block(s)
	if !ok {
		return Reorder{}, false, nil
	}
	n, err := p.State.Holders(block)
	if err != nil {
		return Reorder{}, false, fmt.Errorf("cudabpf: block %d: %w", block, err)
	}
	if n < 0 {
		return Reorder{}, false, fmt.Errorf("cudabpf: block %d: negative holder count", block)
	}
	if n == 0 {
		return Reorder{Destination: DestinationUnused, Position: PositionHead}, true, nil
	}
	return Reorder{Destination: DestinationUsed, Position: PositionTail}, true, nil
}

// BlockAccess implements [MemoryPolicy].
func (p *HolderPolicy) BlockAccess(d *PMMDecision, b *Budget) (Reorder, bool, error) {
	if d == nil {
		return Reorder{}, false, nil
	}
	return p.Decide(d.Observed, b)
}

// BlockActivate implements [MemoryPolicy]. Activation and access take the
// same decision: both ask where a chunk belongs given who still holds it.
func (p *HolderPolicy) BlockActivate(d *PMMDecision, b *Budget) (Reorder, bool, error) {
	if d == nil {
		return Reorder{}, false, nil
	}
	return p.Decide(d.Observed, b)
}

// Prefetch implements [MemoryPolicy]. A holder policy has no view on
// prefetching, and says so rather than inventing a region.
func (p *HolderPolicy) Prefetch(*PrefetchDecision, *Budget) (Region, bool, error) {
	return Region{}, false, nil
}
