package cudabpf_test

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudabpf"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

// TestBlockEvictionDecisionParityWithCudakv drives the placement policy from
// the live KV allocator rather than a model of it. Block ownership,
// allocation and refcounting stay in cudakv; this package supplies events and
// decisions. Parity is the property that matters: a block cudakv says is
// unreferenced is exactly the block this policy moves toward the unused list,
// and no other.
//
// *cudakv.BlockAllocator satisfies [cudabpf.BlockState] structurally --
// nothing here imports it outside this test, and nothing in cudakv knows
// about this package.
func TestBlockEvictionDecisionParityWithCudakv(t *testing.T) {
	const blockSize, totalBlocks = 4, 16
	alloc := cudakv.NewBlockAllocator(blockSize, totalBlocks)

	var _ cudabpf.BlockState = alloc

	if err := alloc.AllocSequence(1, nil, 10); err != nil {
		t.Fatalf("AllocSequence(1): %v", err)
	}
	if err := alloc.AllocSequence(2, nil, 6); err != nil {
		t.Fatalf("AllocSequence(2): %v", err)
	}
	held := append(append([]int(nil), alloc.Table(1)...), alloc.Table(2)...)
	if len(held) == 0 {
		t.Fatal("the allocator handed out no blocks; the parity check would be vacuous")
	}
	alloc.FreeSequence(2)

	policy := &cudabpf.HolderPolicy{
		State: alloc,
		Block: func(s cudabpf.PMMSnapshot) (int, bool) { return int(s.RootID), true },
	}

	freed, kept := 0, 0
	for block := 0; block < totalBlocks; block++ {
		holders, err := alloc.Holders(block)
		if err != nil {
			t.Fatalf("Holders(%d): %v", block, err)
		}
		r, ok, err := policy.Decide(cudabpf.PMMSnapshot{RootID: uint64(block)}, cudabpf.NewBudget(8))
		if err != nil {
			t.Fatalf("Decide(%d): %v", block, err)
		}
		if !ok {
			t.Fatalf("no decision for block %d", block)
		}
		switch holders {
		case 0:
			freed++
			if r.Destination != cudabpf.DestinationUnused {
				t.Errorf("block %d has no holders; policy said %v, want unused", block, r)
			}
		default:
			kept++
			if r.Destination != cudabpf.DestinationUsed {
				t.Errorf("block %d has %d holders; policy said %v, want used", block, holders, r)
			}
		}
	}
	if freed == 0 || kept == 0 {
		t.Fatalf("parity checked %d free and %d held blocks; both sides must be exercised", freed, kept)
	}

	// Releasing the remaining sequence must move its blocks across, or
	// the policy is reading something other than the live refcount.
	before := alloc.Table(1)
	alloc.FreeSequence(1)
	for _, block := range before {
		r, ok, err := policy.Decide(cudabpf.PMMSnapshot{RootID: uint64(block)}, cudabpf.NewBudget(8))
		if err != nil || !ok {
			t.Fatalf("Decide(%d) after free: %v %v", block, r, err)
		}
		if r.Destination != cudabpf.DestinationUnused {
			t.Errorf("block %d was freed; policy still says %v", block, r)
		}
	}

	// A block the allocator does not know is an error, not zero holders:
	// zero holders means free, which is the opposite claim.
	if _, _, err := policy.Decide(cudabpf.PMMSnapshot{RootID: totalBlocks + 5}, cudabpf.NewBudget(8)); err == nil {
		t.Error("an unknown block was treated as a decision rather than an error")
	}
}
