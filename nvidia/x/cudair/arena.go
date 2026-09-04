package cudair

// ArenaPlan represents the planned memory layout for tensor scratchpads.
type ArenaPlan struct {
	TotalBytes int
	Offsets    map[*Node]int
}

// PlanMemory computes a memory arena layout that reuses intermediate buffers
// after their last consumer has finished.
func PlanMemory(g *Graph) *ArenaPlan {
	nodes := g.Nodes()
	if len(nodes) == 0 {
		return &ArenaPlan{
			TotalBytes: 0,
			Offsets:    make(map[*Node]int),
		}
	}

	// Calculate last consumer index for each node
	lastUse := make(map[*Node]int)
	for i, n := range nodes {
		for _, inp := range n.Inputs {
			if i > lastUse[inp] {
				lastUse[inp] = i
			}
		}
	}

	offsets := make(map[*Node]int)
	totalBytes := 0

	type block struct {
		offset   int
		size     int
		freeStep int
	}
	var allocated []block

	for i, n := range nodes {
		if n.Kind == KindInput {
			continue // External inputs are backed by caller memory
		}

		size := n.SizeBytes()
		// 256-byte alignment for fast GPU vectorized loads
		align := 256
		alignedSize := (size + align - 1) &^ (align - 1)

		// Search for reusable free block
		found := -1
		for bIdx, b := range allocated {
			if b.freeStep < i && b.size >= alignedSize {
				found = bIdx
				break
			}
		}
		freeStep, ok := lastUse[n]
		if !ok {
			freeStep = len(nodes)
		}

		if found >= 0 {
			offsets[n] = allocated[found].offset
			allocated[found].freeStep = freeStep
		} else {
			offset := totalBytes
			offsets[n] = offset
			totalBytes += alignedSize
			allocated = append(allocated, block{
				offset:   offset,
				size:     alignedSize,
				freeStep: freeStep,
			})
		}
	}

	return &ArenaPlan{
		TotalBytes: totalBytes,
		Offsets:    offsets,
	}
}
