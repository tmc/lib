package cudabatch

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"sync"
)

// KVBlockManager abstracts KV block allocation and reference counting.
// cudakv.BlockAllocator and cudakv.PagedCache satisfy or interface with this.
type KVBlockManager interface {
	AllocSequence(seqID int64, shared []int, tokens int) error
	AppendToken(seqID int64) error
	FreeSequence(seqID int64)
	Retain(blocks []int) error
	Release(blocks []int) error
	NumFreeBlocks() int
	Table(seqID int64) []int
	BlockSize() int
}

// MemoryBudgetTracker tracks admission and allocation limits against KV pool blocks.
type MemoryBudgetTracker struct {
	mu           sync.Mutex
	kvMgr        KVBlockManager
	reservedReqs map[int64]int // seqID -> allocated block count
	allocations  map[int64]uint64
}

// NewMemoryBudgetTracker wraps a KVBlockManager. Allocation-bound scheduling
// requires an actual *cudakv.BlockAllocator; unsupported managers refuse admission
// and binding rather than inventing pool ownership.
func NewMemoryBudgetTracker(kvMgr KVBlockManager) *MemoryBudgetTracker {
	return &MemoryBudgetTracker{
		kvMgr:        kvMgr,
		reservedReqs: make(map[int64]int),
		allocations:  make(map[int64]uint64),
	}
}

// CanAdmit checks if the block manager has sufficient free blocks to admit a request
// with promptTokens tokens and shared prefix blocks.
func (t *MemoryBudgetTracker) CanAdmit(promptTokens int, sharedBlocks []int) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	pool, ok := t.kvMgr.(*cudakv.BlockAllocator)
	if !ok || pool == nil || pool.BlockSize() <= 0 || promptTokens <= 0 || pool.TotalBlocks() < 0 {
		return false
	}
	blockSize := pool.BlockSize()
	neededBlocks := (promptTokens-1)/blockSize + 1 - len(sharedBlocks)
	if len(sharedBlocks) > promptTokens/blockSize {
		return false
	}
	seen := map[int]bool{}
	for _, block := range sharedBlocks {
		n, err := pool.Holders(block)
		if err != nil || n <= 0 || seen[block] {
			return false
		}
		seen[block] = true
	}
	if neededBlocks <= 0 {
		return true
	}
	return t.kvMgr.NumFreeBlocks() >= neededBlocks
}

// AllocSequence allocates initial physical KV blocks for seqID.
func (t *MemoryBudgetTracker) AllocSequence(seqID int64, shared []int, promptTokens int) ([]int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	pool, ok := t.kvMgr.(*cudakv.BlockAllocator)
	if !ok || pool == nil || pool.BlockSize() <= 0 || promptTokens <= 0 {
		return nil, fmt.Errorf("%w: actual initialized block allocator required", ErrCorruptBlockTable)
	}

	if err := t.kvMgr.AllocSequence(seqID, shared, promptTokens); err != nil {
		return nil, fmt.Errorf("cudabatch: alloc sequence %d: %w", seqID, err)
	}
	tbl := t.kvMgr.Table(seqID)
	t.reservedReqs[seqID] = len(tbl)
	t.allocations[seqID], _ = pool.AllocationGeneration(seqID)
	return tbl, nil
}

// AppendToken reserves room for an additional generated token.
func (t *MemoryBudgetTracker) AppendToken(seqID int64) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	pool, ok := t.kvMgr.(*cudakv.BlockAllocator)
	if !ok || pool == nil {
		return ErrCorruptBlockTable
	}
	length, ok := pool.SequenceLength(seqID)
	if !ok || pool.ValidateSequence(seqID, t.allocations[seqID], pool.Table(seqID), length) != nil {
		return ErrCorruptBlockTable
	}

	if err := t.kvMgr.AppendToken(seqID); err != nil {
		return fmt.Errorf("cudabatch: append token %d: %w", seqID, err)
	}
	tbl := t.kvMgr.Table(seqID)
	t.reservedReqs[seqID] = len(tbl)
	return nil
}

// FreeSequence frees sequence blocks when all execution holders complete.
func (t *MemoryBudgetTracker) FreeSequence(seqID int64) {
	if t.kvMgr == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	pool, ok := t.kvMgr.(*cudakv.BlockAllocator)
	if !ok || pool == nil {
		return
	}
	length, ok := pool.SequenceLength(seqID)
	if !ok || pool.ValidateSequence(seqID, t.allocations[seqID], pool.Table(seqID), length) != nil {
		return
	}

	t.kvMgr.FreeSequence(seqID)
	delete(t.reservedReqs, seqID)
	delete(t.allocations, seqID)
}

// RetainPrefix marks prefix blocks as retained (e.g. for multi-turn continuations).
func (t *MemoryBudgetTracker) RetainPrefix(blocks []int) error {
	if t.kvMgr == nil || len(blocks) == 0 {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.kvMgr.Retain(blocks)
}

// ReleasePrefix releases a previously retained prefix.
func (t *MemoryBudgetTracker) ReleasePrefix(blocks []int) error {
	if t.kvMgr == nil || len(blocks) == 0 {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.kvMgr.Release(blocks)
}
