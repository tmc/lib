package cudabatch

import (
	"fmt"
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
}

// NewMemoryBudgetTracker creates a tracker wrapping a KVBlockManager.
func NewMemoryBudgetTracker(kvMgr KVBlockManager) *MemoryBudgetTracker {
	return &MemoryBudgetTracker{
		kvMgr:        kvMgr,
		reservedReqs: make(map[int64]int),
	}
}

// CanAdmit checks if the block manager has sufficient free blocks to admit a request
// with promptTokens tokens and shared prefix blocks.
func (t *MemoryBudgetTracker) CanAdmit(promptTokens int, sharedBlocks []int) bool {
	if t.kvMgr == nil {
		return true
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	blockSize := t.kvMgr.BlockSize()
	if blockSize <= 0 {
		blockSize = 16
	}
	neededBlocks := (promptTokens+blockSize-1)/blockSize - len(sharedBlocks)
	if neededBlocks <= 0 {
		return true
	}
	return t.kvMgr.NumFreeBlocks() >= neededBlocks
}

// AllocSequence allocates initial physical KV blocks for seqID.
func (t *MemoryBudgetTracker) AllocSequence(seqID int64, shared []int, promptTokens int) ([]int, error) {
	if t.kvMgr == nil {
		return nil, nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	if err := t.kvMgr.AllocSequence(seqID, shared, promptTokens); err != nil {
		return nil, fmt.Errorf("cudabatch: alloc sequence %d: %w", seqID, err)
	}
	tbl := t.kvMgr.Table(seqID)
	t.reservedReqs[seqID] = len(tbl)
	return tbl, nil
}

// AppendToken reserves room for an additional generated token.
func (t *MemoryBudgetTracker) AppendToken(seqID int64) error {
	if t.kvMgr == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()

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

	t.kvMgr.FreeSequence(seqID)
	delete(t.reservedReqs, seqID)
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
