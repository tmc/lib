package cudair

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

// ArenaPlan represents the planned memory layout for tensor scratchpads.
type ArenaPlan struct {
	TotalBytes int
	Offsets    map[*Node]int
}

// PlanMemory computes an optimal memory arena layout reusing intermediate buffers
// based on tensor lifetime analysis along the topological sequential schedule.
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
		// Fallback to non-overlapping baseline on error
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

	offsets := make(map[*Node]int, len(order))
	var totalBytes uint64

	type block struct {
		offset   int
		size     int
		freeStep int
	}
	var allocated []block

	for i, n := range order {
		if n.Kind == KindInput {
			continue // External inputs are backed by caller memory
		}

		// Check for storage alias reuse (e.g. Reshape or in-place operation)
		if n.StorageAlias != nil {
			if aliasOffset, ok := offsets[n.StorageAlias]; ok {
				offsets[n] = aliasOffset
				// Extend lifetime of the underlying allocated block
				for bIdx := range allocated {
					if allocated[bIdx].offset == aliasOffset {
						if lastUse[n] > allocated[bIdx].freeStep {
							allocated[bIdx].freeStep = lastUse[n]
						}
						break
					}
				}
				continue
			}
		}

		size := n.SizeBytes()
		if size < 0 {
			return nil, fmt.Errorf("%w: negative size for node %s", ErrShapeMismatch, n.Name)
		}

		// 256-byte alignment for fast GPU vectorized loads
		align := 256
		alignedSize := (size + align - 1) &^ (align - 1)

		// Search for reusable free block
		found := -1
		for bIdx, b := range allocated {
			if b.freeStep <= i && b.size >= alignedSize {
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
