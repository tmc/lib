package cudair

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

// ArenaPlan represents the planned memory layout for tensor scratchpads.
// Offsets contains arena-backed nodes, including their storage aliases.
// External inputs and aliases backed by external inputs have no offset.
// Snapshot accessors return detached keys; compare their stable node IDs.
type ArenaPlan struct {
	TotalBytes int
	Offsets    map[*Node]int
}

// PlanMemory plans a memory arena reusing intermediate buffers after their last
// consumer completes. Graph outputs remain live through graph completion.
// It returns an empty plan on error; use Graph.Snapshot for checked planning.
func PlanMemory(g *Graph) *ArenaPlan {
	nodes := g.Nodes()
	if len(nodes) == 0 {
		return &ArenaPlan{
			TotalBytes: 0,
			Offsets:    make(map[*Node]int),
		}
	}

	// Calculate consumers
	consumers := make(map[*Node][]*Node)
	for _, n := range nodes {
		for _, inp := range n.Inputs {
			consumers[inp] = append(consumers[inp], n)
		}
	}

	plan, err := planMemorySafe(nodes, consumers)
	if err != nil {
		// Preserve the legacy unchecked API's empty result on planning errors.
		return &ArenaPlan{
			TotalBytes: 0,
			Offsets:    make(map[*Node]int),
		}
	}
	return plan
}

func planMemorySafe(order []*Node, consumers map[*Node][]*Node) (*ArenaPlan, error) {
	if len(order) == 0 {
		return &ArenaPlan{
			TotalBytes: 0,
			Offsets:    make(map[*Node]int),
		}, nil
	}

	// Calculate step index for each node
	stepMap := make(map[*Node]int, len(order))
	for i, n := range order {
		stepMap[n] = i
	}

	// Calculate last consumer step for each node
	lastUse := make(map[*Node]int, len(order))
	for i, n := range order {
		lastUse[n] = i
		for _, cons := range consumers[n] {
			if s, ok := stepMap[cons]; ok && s > lastUse[n] {
				lastUse[n] = s
			}
		}
	}

	// Outputs outlive every graph step. Propagate alias lifetimes before any
	// block is assigned so an output view keeps its whole storage chain live.
	if graph := order[0].graph; graph != nil {
		for _, out := range graph.Outputs() {
			if _, ok := stepMap[out]; ok {
				lastUse[out] = len(order)
			}
		}
	}
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if n.StorageAlias != nil && lastUse[n] > lastUse[n.StorageAlias] {
			lastUse[n.StorageAlias] = lastUse[n]
		}
	}

	offsets := make(map[*Node]int, len(order))
	external := make(map[*Node]bool)
	var totalBytes uint64

	type block struct {
		offset   int
		size     int
		freeStep int
	}
	var allocated []block

	for i, n := range order {
		if n.Kind == KindInput {
			external[n] = true
			continue // External inputs are backed by caller memory.
		}

		if n.StorageAlias != nil {
			if external[n.StorageAlias] {
				external[n] = true
				continue
			}
			aliasOffset, ok := offsets[n.StorageAlias]
			if !ok {
				return nil, fmt.Errorf("%w: storage alias for node %s has no arena allocation", ErrUnsupportedEffect, n.Name)
			}
			offsets[n] = aliasOffset
			continue
		}

		size := n.SizeBytes()
		if size < 0 {
			return nil, fmt.Errorf("%w: negative size for node %s", ErrShapeMismatch, n.Name)
		}

		// 256-byte alignment for fast GPU vectorized loads
		align := 256
		if size > math.MaxInt-(align-1) {
			return nil, fmt.Errorf("%w: aligned size for node %s exceeds address limit", ErrArenaOverflow, n.Name)
		}
		alignedSize := (size + align - 1) &^ (align - 1)

		// Search for reusable free block
		found := -1
		for bIdx, b := range allocated {
			if b.freeStep < i && b.size >= alignedSize {
				found = bIdx
				break
			}
		}

		if found >= 0 {
			offsets[n] = allocated[found].offset
			allocated[found].freeStep = lastUse[n]
		} else {
			offset := totalBytes
			if offset > cudashape.MaxByteLimit || uint64(alignedSize) > cudashape.MaxByteLimit {
				return nil, fmt.Errorf("%w: node %s offset exceeds memory limit", ErrArenaOverflow, n.Name)
			}
			newTotal, overflow := cudashape.SafeMul(1, totalBytes+uint64(alignedSize))
			if overflow || newTotal > cudashape.MaxByteLimit || newTotal > math.MaxInt {
				return nil, fmt.Errorf("%w: total scratchpad allocation %d exceeds limit", ErrArenaOverflow, newTotal)
			}
			offsets[n] = int(offset)
			totalBytes = newTotal
			allocated = append(allocated, block{
				offset:   int(offset),
				size:     alignedSize,
				freeStep: lastUse[n],
			})
		}
	}

	return &ArenaPlan{
		TotalBytes: int(totalBytes),
		Offsets:    offsets,
	}, nil
}
