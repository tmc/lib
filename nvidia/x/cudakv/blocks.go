// SPDX-License-Identifier: Apache-2.0

package cudakv

import "fmt"

// blockAllocator hands out the physical slots of a block pool and counts the
// holders of each. It owns no device storage, so PagedCache's bookkeeping can
// be tested without a device.
//
// A block is free when nothing holds it. A sequence holds every block in its
// table; a prefix cache that keeps a block alive after the sequence that wrote
// it has finished holds it too, through retain. The block returns to the free
// list when the last holder lets go, whichever kind of holder that is. Two
// unconnected owners of the same block, one recycling it while the other still
// reads it, is the failure this exists to rule out.
type blockAllocator struct {
	blockSize int
	free      []int
	holders   []int // per block; 0 means free
	tables    map[int64][]int
	lens      map[int64]int
}

func newBlockAllocator(blockSize, totalBlocks int) *blockAllocator {
	free := make([]int, totalBlocks)
	for i := range free {
		free[i] = i
	}
	return &blockAllocator{
		blockSize: blockSize,
		free:      free,
		holders:   make([]int, totalBlocks),
		tables:    make(map[int64][]int),
		lens:      make(map[int64]int),
	}
}

// allocSequence gives seqID a block table covering tokens. The first
// len(shared) blocks are shared with whoever already holds them and are
// counted as full; the rest come from the free list.
func (a *blockAllocator) allocSequence(seqID int64, shared []int, tokens int) error {
	if _, exists := a.tables[seqID]; exists {
		return fmt.Errorf("%s: sequence %d already allocated", pkgName, seqID)
	}
	if tokens < len(shared)*a.blockSize {
		return fmt.Errorf("%s: %d tokens do not cover %d shared blocks of %d", pkgName, tokens, len(shared), a.blockSize)
	}
	for _, b := range shared {
		if err := a.held(b); err != nil {
			return fmt.Errorf("%s: sharing: %w", pkgName, err)
		}
	}
	needed := (tokens+a.blockSize-1)/a.blockSize - len(shared)
	if len(a.free) < needed {
		return fmt.Errorf("%s: out of physical KV blocks (needed %d, available %d)", pkgName, needed, len(a.free))
	}
	table := make([]int, 0, len(shared)+needed)
	for _, b := range shared {
		a.holders[b]++
		table = append(table, b)
	}
	for i := 0; i < needed; i++ {
		table = append(table, a.take())
	}
	a.tables[seqID] = table
	a.lens[seqID] = tokens
	return nil
}

// appendToken reserves room for one more token in seqID, taking a fresh block
// at each block boundary.
func (a *blockAllocator) appendToken(seqID int64) error {
	table, exists := a.tables[seqID]
	if !exists {
		return fmt.Errorf("%s: sequence %d not found", pkgName, seqID)
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

// freeSequence drops seqID's hold on its blocks. Blocks nothing else holds
// return to the free list; shared ones stay with their other holders.
func (a *blockAllocator) freeSequence(seqID int64) {
	table, exists := a.tables[seqID]
	if !exists {
		return
	}
	for _, b := range table {
		a.drop(b)
	}
	delete(a.tables, seqID)
	delete(a.lens, seqID)
}

// retain adds a holder to each block. Every block must already be held: a free
// block has no contents worth keeping, so retaining one is a caller bug.
func (a *blockAllocator) retain(blocks []int) error {
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

// release removes one holder from each block, freeing those left with none.
func (a *blockAllocator) release(blocks []int) error {
	for _, b := range blocks {
		if err := a.held(b); err != nil {
			return fmt.Errorf("%s: release: %w", pkgName, err)
		}
	}
	for _, b := range blocks {
		a.drop(b)
	}
	return nil
}

func (a *blockAllocator) held(b int) error {
	if b < 0 || b >= len(a.holders) {
		return fmt.Errorf("block %d out of range (total %d)", b, len(a.holders))
	}
	if a.holders[b] == 0 {
		return fmt.Errorf("block %d is free", b)
	}
	return nil
}

func (a *blockAllocator) take() int {
	b := a.free[len(a.free)-1]
	a.free = a.free[:len(a.free)-1]
	a.holders[b] = 1
	return b
}

func (a *blockAllocator) drop(b int) {
	a.holders[b]--
	if a.holders[b] == 0 {
		a.free = append(a.free, b)
	}
}

func (a *blockAllocator) table(seqID int64) []int {
	return append([]int(nil), a.tables[seqID]...)
}
