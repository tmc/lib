package cudaspec

import (
	"errors"
	"slices"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

// BranchKV holds a speculative sequence and its share of an existing prefix.
// The allocator must be exclusively owned by this consumer; BlockAllocator is
// not safe for unrelated concurrent mutation. Close releases only this sequence.
// The zero value is closed. Do not close while a device still reads its table.
type BranchKV struct {
	mu             sync.Mutex
	allocator      *cudakv.BlockAllocator
	sequence       int64
	table, private []int
	closed         bool
}

// NewBranchKV shares a full-block prefix and reserves space for draft tokens.
// Partial prefix blocks are refused because sharing them for writes would
// modify another sequence's KV. Native KV contents are not copied here.
func NewBranchKV(a *cudakv.BlockAllocator, prefixSequence, sequence int64, prefixTokens, draftTokens int) (*BranchKV, error) {
	if a == nil || a.BlockSize() <= 0 || prefixTokens <= 0 || prefixTokens > 4096 || draftTokens <= 0 || draftTokens > 16 || prefixSequence == sequence {
		return nil, errors.New("cudaspec: invalid KV branch configuration")
	}
	n, ok := a.SequenceLength(prefixSequence)
	if !ok || n != prefixTokens || prefixTokens%a.BlockSize() != 0 {
		return nil, errors.New("cudaspec: missing, changed, or partial-block KV prefix")
	}
	shared := a.Table(prefixSequence)
	if len(shared) != prefixTokens/a.BlockSize() {
		return nil, errors.New("cudaspec: invalid prefix block table")
	}
	seen := make(map[int]bool, len(shared))
	for _, block := range shared {
		if seen[block] {
			return nil, errors.New("cudaspec: duplicate prefix block")
		}
		seen[block] = true
		holders, err := a.Holders(block)
		if err != nil || holders < 1 {
			return nil, errors.New("cudaspec: unheld prefix block")
		}
	}
	if err := a.AllocSequence(sequence, shared, prefixTokens+draftTokens); err != nil {
		return nil, err
	}
	table := a.Table(sequence)
	return &BranchKV{allocator: a, sequence: sequence, table: table, private: append([]int(nil), table[len(shared):]...)}, nil
}

// Table returns a copy of the live branch's block table.
func (b *BranchKV) Table() ([]int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.allocator == nil || b.closed {
		return nil, errors.New("cudaspec: KV branch is closed")
	}
	if !slices.Equal(b.table, b.allocator.Table(b.sequence)) {
		return nil, errors.New("cudaspec: KV branch table changed")
	}
	return append([]int(nil), b.table...), nil
}

// Release implements KVBlockReleaser. It refuses borrowed-prefix or foreign
// blocks and drops the sequence's references through the allocator exactly once.
func (b *BranchKV) Release(blocks []int) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.allocator == nil {
		return errors.New("cudaspec: uninitialized KV branch")
	}
	if !slices.Equal(blocks, b.private) {
		return errors.New("cudaspec: release does not match private branch blocks")
	}
	return b.closeLocked()
}

// Close releases the branch's private blocks and its prefix references.
// The original prefix sequence and other borrowers remain live.
func (b *BranchKV) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.allocator == nil {
		return nil
	}
	return b.closeLocked()
}
func (b *BranchKV) closeLocked() error {
	if b.closed {
		return nil
	}
	if !slices.Equal(b.table, b.allocator.Table(b.sequence)) {
		return errors.New("cudaspec: KV branch table changed")
	}
	b.allocator.FreeSequence(b.sequence)
	b.closed = true
	return nil
}
