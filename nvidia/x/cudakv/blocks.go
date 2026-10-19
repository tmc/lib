// SPDX-License-Identifier: Apache-2.0

package cudakv

import (
	"errors"
	"fmt"
	"strconv"
)

// ErrBlockAllocatorUninitialized is returned when an allocation method is
// called on a BlockAllocator that was not created with NewBlockAllocator.
var ErrBlockAllocatorUninitialized = errors.New("cudakv: block allocator must be created with NewBlockAllocator")

// BlockAllocator hands out the physical slots of a block pool and counts the
// holders of each. It owns no device storage, so PagedCache's bookkeeping can
// be tested without a device. Create a BlockAllocator with NewBlockAllocator;
// its zero value is not usable.
//
// A block is free when nothing holds it. A sequence holds every block in its
// table; a prefix cache that keeps a block alive after the sequence that wrote
// it has finished holds it too, through retain. The block returns to the free
// list when the last holder lets go, whichever kind of holder that is. Two
// unconnected owners of the same block, one recycling it while the other still
// reads it, is the failure this exists to rule out.
type BlockAllocator struct {
	blockSize      int
	totalBlocks    int
	free           []int
	holders        []int // per block; 0 means free
	tables         map[int64][]int
	lens           map[int64]int
	err            error // non-nil when the constructor arguments or extent were invalid
	allocations    map[int64]uint64
	nextAllocation uint64
}

// blockAllocator is an alias for BlockAllocator preserving package-internal compatibility.
type blockAllocator = BlockAllocator

// NewBlockAllocator creates a new allocator with totalBlocks physical slots of
// blockSize tokens. If blockSize is not positive, totalBlocks is negative, or
// the block storage size is not representable, the allocator has no slots and
// AllocSequence and AppendToken report the invalid arguments.
func NewBlockAllocator(blockSize, totalBlocks int) *BlockAllocator {
	if blockSize <= 0 || totalBlocks < 0 || totalBlocks > maxBlockSliceLen() {
		return &BlockAllocator{
			blockSize: blockSize,
			tables:    make(map[int64][]int),
			lens:      make(map[int64]int),
			err:       fmt.Errorf("%s: invalid block allocator: block size %d, total blocks %d", pkgName, blockSize, totalBlocks),
		}
	}
	free := make([]int, totalBlocks)
	for i := range free {
		free[i] = i
	}
	return &BlockAllocator{
		blockSize:   blockSize,
		totalBlocks: totalBlocks,
		free:        free,
		holders:     make([]int, totalBlocks),
		tables:      make(map[int64][]int),
		lens:        make(map[int64]int),
		allocations: make(map[int64]uint64),
	}
}

func maxBlockSliceLen() int {
	return int(^uint(0)>>1) / (strconv.IntSize / 8)
}

func (a *BlockAllocator) admissionError() error {
	if a.err != nil {
		return a.err
	}
	if a.blockSize <= 0 || a.totalBlocks < 0 || len(a.holders) != a.totalBlocks || a.tables == nil || a.lens == nil {
		return ErrBlockAllocatorUninitialized
	}
	return nil
}

func newBlockAllocator(blockSize, totalBlocks int) *BlockAllocator {
	return NewBlockAllocator(blockSize, totalBlocks)
}

// AllocSequence gives seqID a block table covering tokens. The first
// len(shared) blocks are shared with whoever already holds them and are
// counted as full; the rest come from the free list. Each shared block may
// appear at most once.
func (a *BlockAllocator) AllocSequence(seqID int64, shared []int, tokens int) error {
	if err := a.admissionError(); err != nil {
		return err
	}
	if _, exists := a.tables[seqID]; exists {
		return fmt.Errorf("%s: sequence %d already allocated", pkgName, seqID)
	}
	if a.nextAllocation == ^uint64(0) {
		return fmt.Errorf("%s: allocation identity exhausted", pkgName)
	}
	if tokens < 0 {
		return fmt.Errorf("%s: token count %d is negative", pkgName, tokens)
	}
	if len(shared) > tokens/a.blockSize {
		return fmt.Errorf("%s: %d tokens do not cover %d shared blocks of %d", pkgName, tokens, len(shared), a.blockSize)
	}
	blocks := tokens / a.blockSize
	if tokens%a.blockSize != 0 {
		blocks++
	}
	if blocks > maxBlockSliceLen() {
		return fmt.Errorf("%s: token count %d requires an unrepresentable block table", pkgName, tokens)
	}
	seen := make(map[int]bool, len(shared))
	for _, b := range shared {
		if err := a.held(b); err != nil {
			return fmt.Errorf("%s: sharing: %w", pkgName, err)
		}
		if seen[b] {
			return fmt.Errorf("%s: sharing: block %d appears more than once", pkgName, b)
		}
		seen[b] = true
	}
	needed := blocks - len(shared)
	if len(a.free) < needed {
		return fmt.Errorf("%s: out of physical KV blocks (needed %d, available %d)", pkgName, needed, len(a.free))
	}
	table := make([]int, 0, blocks)
	for _, b := range shared {
		a.holders[b]++
		table = append(table, b)
	}
	for i := 0; i < needed; i++ {
		table = append(table, a.take())
	}
	a.tables[seqID] = table
	a.lens[seqID] = tokens
	if a.allocations == nil {
		a.allocations = make(map[int64]uint64)
	}
	a.nextAllocation++
	a.allocations[seqID] = a.nextAllocation
	return nil
}

func (a *BlockAllocator) allocSequence(seqID int64, shared []int, tokens int) error {
	return a.AllocSequence(seqID, shared, tokens)
}

// AppendToken reserves room for one more token in seqID, taking a fresh block
// at each block boundary.
func (a *BlockAllocator) AppendToken(seqID int64) error {
	if err := a.admissionError(); err != nil {
		return err
	}
	table, exists := a.tables[seqID]
	if !exists {
		return fmt.Errorf("%s: sequence %d not found", pkgName, seqID)
	}
	if a.lens[seqID] > 0 {
		if err := a.ValidateSequence(seqID, a.allocations[seqID], table, a.lens[seqID]); err != nil {
			return err
		}
	} else if len(table) != 0 {
		return fmt.Errorf("%s: corrupt empty sequence table", pkgName)
	}
	if a.lens[seqID] == int(^uint(0)>>1) {
		return fmt.Errorf("%s: sequence %d token count would overflow", pkgName, seqID)
	}
	if a.lens[seqID]%a.blockSize == 0 {
		if len(a.free) == 0 {
			return fmt.Errorf("%s: out of KV blocks during generation", pkgName)
		}
		a.tables[seqID] = append(table, a.take())
	}
	a.lens[seqID]++
	return nil
}

func (a *BlockAllocator) appendToken(seqID int64) error {
	return a.AppendToken(seqID)
}

// FreeSequence drops seqID's hold on its blocks. Blocks nothing else holds
// return to the free list; shared ones stay with their other holders.
func (a *BlockAllocator) FreeSequence(seqID int64) {
	table, exists := a.tables[seqID]
	if !exists {
		return
	}
	for _, b := range table {
		a.drop(b)
	}
	delete(a.tables, seqID)
	delete(a.lens, seqID)
	delete(a.allocations, seqID)
}

// AllocationGeneration identifies the current allocation of seqID. Freeing and
// reallocating an ID changes its generation even if the same blocks are reused.
// Like the allocator's other methods, calls must be externally serialized.
func (a *BlockAllocator) AllocationGeneration(seqID int64) (uint64, bool) {
	if a == nil {
		return 0, false
	}
	g, ok := a.allocations[seqID]
	return g, ok
}

// ValidateSequence checks a complete physical table against this pool's live
// sequence allocation and token capacity. It changes no allocation or holder.
func (a *BlockAllocator) ValidateSequence(seqID int64, generation uint64, blocks []int, tokens int) error {
	if a == nil {
		return ErrBlockAllocatorUninitialized
	}
	if err := a.admissionError(); err != nil {
		return err
	}
	if generation == 0 || a.allocations[seqID] != generation {
		return fmt.Errorf("%s: sequence allocation changed", pkgName)
	}
	owned, ok := a.tables[seqID]
	n, exists := a.lens[seqID]
	if !ok || !exists || tokens <= 0 || tokens > n || len(blocks) != len(owned) {
		return fmt.Errorf("%s: sequence table or capacity mismatch", pkgName)
	}
	needed := n / a.blockSize
	if n%a.blockSize != 0 {
		needed++
	}
	if len(owned) != needed {
		return fmt.Errorf("%s: corrupt sequence capacity", pkgName)
	}
	seen := make(map[int]bool, len(blocks))
	for i, b := range blocks {
		if b != owned[i] || b < 0 || b >= a.totalBlocks || a.holders[b] <= 0 || seen[b] {
			return fmt.Errorf("%s: foreign, free or duplicate sequence block", pkgName)
		}
		seen[b] = true
	}
	return nil
}

func (a *BlockAllocator) freeSequence(seqID int64) {
	a.FreeSequence(seqID)
}

// Retain adds a holder to each block. Every block must already be held: a free
// block has no contents worth keeping, so retaining one is a caller bug.
func (a *BlockAllocator) Retain(blocks []int) error {
	for _, b := range blocks {
		if err := a.held(b); err != nil {
			return fmt.Errorf("%s: retain: %w", pkgName, err)
		}
	}
	for _, b := range blocks {
		a.holders[b]++
	}
	return nil
}

func (a *BlockAllocator) retain(blocks []int) error {
	return a.Retain(blocks)
}

// Release removes one holder for each occurrence of a block, freeing those
// left with none. If any block has fewer holders than requested, Release
// returns an error without changing any block.
func (a *BlockAllocator) Release(blocks []int) error {
	releases := make(map[int]int)
	for _, b := range blocks {
		if err := a.held(b); err != nil {
			return fmt.Errorf("%s: release: %w", pkgName, err)
		}
		releases[b]++
		if releases[b] > a.holders[b] {
			return fmt.Errorf("%s: release: block %d has %d holders, cannot release %d", pkgName, b, a.holders[b], releases[b])
		}
	}
	for _, b := range blocks {
		a.drop(b)
	}
	return nil
}

func (a *BlockAllocator) release(blocks []int) error {
	return a.Release(blocks)
}

// Holders reports how many sequences and retainers hold block b.
func (a *BlockAllocator) Holders(b int) (int, error) {
	if b < 0 || b >= len(a.holders) {
		return 0, fmt.Errorf("%s: block %d out of range (total %d)", pkgName, b, len(a.holders))
	}
	return a.holders[b], nil
}

// NumFreeBlocks returns the count of currently unallocated blocks.
func (a *BlockAllocator) NumFreeBlocks() int {
	return len(a.free)
}

// TotalBlocks returns the total capacity of physical blocks in the pool.
func (a *BlockAllocator) TotalBlocks() int {
	return a.totalBlocks
}

// BlockSize returns the token capacity per block.
func (a *BlockAllocator) BlockSize() int {
	return a.blockSize
}

// SequenceLength returns the number of tokens allocated to seqID.
func (a *BlockAllocator) SequenceLength(seqID int64) (int, bool) {
	l, ok := a.lens[seqID]
	return l, ok
}

// Table returns a copy of the block table for sequence seqID.
func (a *BlockAllocator) Table(seqID int64) []int {
	return append([]int(nil), a.tables[seqID]...)
}

func (a *BlockAllocator) table(seqID int64) []int {
	return a.Table(seqID)
}

func (a *BlockAllocator) held(b int) error {
	if b < 0 || b >= len(a.holders) {
		return fmt.Errorf("block %d out of range (total %d)", b, len(a.holders))
	}
	if a.holders[b] == 0 {
		return fmt.Errorf("block %d is free", b)
	}
	return nil
}

func (a *BlockAllocator) take() int {
	b := a.free[len(a.free)-1]
	a.free = a.free[:len(a.free)-1]
	a.holders[b] = 1
	return b
}

func (a *BlockAllocator) drop(b int) {
	a.holders[b]--
	if a.holders[b] == 0 {
		a.free = append(a.free, b)
	}
}

// largestContiguousFree calculates the maximum number of consecutive free blocks.
func (a *BlockAllocator) largestContiguousFree() int {
	return largestContiguousFreeIn(a.holders)
}

func largestContiguousFreeIn(holders []int) int {
	maxRun := 0
	currRun := 0
	for _, h := range holders {
		if h == 0 {
			currRun++
			if currRun > maxRun {
				maxRun = currRun
			}
		} else {
			currRun = 0
		}
	}
	return maxRun
}

// FragmentationStats returns current fragmentation statistics for the allocator.
func (a *BlockAllocator) FragmentationStats() FragmentationStats {
	totalFree := len(a.free)
	allocated := a.totalBlocks - totalFree
	largest := a.largestContiguousFree()
	return FragmentationStats{
		TotalBlocks:        a.totalBlocks,
		AllocatedBlocks:    allocated,
		FreeBlocks:         totalFree,
		LargestContiguous:  largest,
		FragmentationIndex: Fragmentation(totalFree, largest),
	}
}

// FragmentationIndex returns the current fragmentation ratio (0.0 to 1.0).
func (a *BlockAllocator) FragmentationIndex() float64 {
	return a.FragmentationStats().FragmentationIndex
}

// PlanCompaction computes the optimal set of block relocations to compact all active
// blocks into contiguous low-index slots [0..ActiveBlocks-1].
func (a *BlockAllocator) PlanCompaction(maxMoves int) CompactionPlan {
	stats := a.FragmentationStats()
	activeBlocks := stats.AllocatedBlocks
	total := a.totalBlocks

	if activeBlocks == 0 || activeBlocks == total {
		return CompactionPlan{
			InitialFragIndex:   stats.FragmentationIndex,
			EstimatedFragIndex: 0.0,
			ActiveBlocks:       activeBlocks,
			TotalBlocks:        total,
		}
	}

	// Identify free destination slots in [0..activeBlocks-1].
	var freeDsts []int
	for i := 0; i < activeBlocks; i++ {
		if a.holders[i] == 0 {
			freeDsts = append(freeDsts, i)
		}
	}

	// Identify active source blocks in [activeBlocks..total-1].
	// Scan from highest index downwards so the most scattered blocks are compacted first.
	var activeSrcs []int
	for i := total - 1; i >= activeBlocks; i-- {
		if a.holders[i] > 0 {
			activeSrcs = append(activeSrcs, i)
		}
	}

	numMoves := len(freeDsts)
	if maxMoves > 0 && maxMoves < numMoves {
		numMoves = maxMoves
	}

	moves := make([]BlockMove, numMoves)
	remappings := make(map[int]int, numMoves)
	for j := 0; j < numMoves; j++ {
		src := activeSrcs[j]
		dst := freeDsts[j]
		moves[j] = BlockMove{SrcBlock: src, DstBlock: dst}
		remappings[src] = dst
	}

	// Simulate resulting fragmentation
	simHolders := make([]int, total)
	copy(simHolders, a.holders)
	for _, m := range moves {
		simHolders[m.DstBlock] = simHolders[m.SrcBlock]
		simHolders[m.SrcBlock] = 0
	}
	simLargest := largestContiguousFreeIn(simHolders)
	simFrag := Fragmentation(stats.FreeBlocks, simLargest)

	return CompactionPlan{
		Moves:              moves,
		Remappings:         remappings,
		InitialFragIndex:   stats.FragmentationIndex,
		EstimatedFragIndex: simFrag,
		ActiveBlocks:       activeBlocks,
		TotalBlocks:        total,
	}
}

// ApplyCompaction applies the relocations in plan to the allocator's internal tables,
// holders, and free block list.
func (a *BlockAllocator) ApplyCompaction(plan CompactionPlan) {
	if plan.Empty() {
		return
	}

	// Relocate holders
	for _, m := range plan.Moves {
		a.holders[m.DstBlock] = a.holders[m.SrcBlock]
		a.holders[m.SrcBlock] = 0
	}

	// Update sequence tables
	for seqID, table := range a.tables {
		for i, b := range table {
			if newB, ok := plan.Remappings[b]; ok {
				table[i] = newB
			}
		}
		a.tables[seqID] = table
	}

	// Rebuild free list: free blocks are all i where holders[i] == 0.
	// We keep them in descending order so take() pops the lowest free block next.
	a.free = a.free[:0]
	for i := a.totalBlocks - 1; i >= 0; i-- {
		if a.holders[i] == 0 {
			a.free = append(a.free, i)
		}
	}
}

// VirtualPageTable returns the virtual-to-physical block table for seqID.
func (a *BlockAllocator) VirtualPageTable(seqID int64) (VirtualPageTable, bool) {
	table, ok := a.tables[seqID]
	if !ok {
		return VirtualPageTable{}, false
	}
	return VirtualPageTable{
		SeqID:  seqID,
		Blocks: append([]int(nil), table...),
	}, true
}

// VirtualPageTables returns virtual page tables for all sequences.
func (a *BlockAllocator) VirtualPageTables() []VirtualPageTable {
	out := make([]VirtualPageTable, 0, len(a.tables))
	for seqID, table := range a.tables {
		out = append(out, VirtualPageTable{
			SeqID:  seqID,
			Blocks: append([]int(nil), table...),
		})
	}
	return out
}
