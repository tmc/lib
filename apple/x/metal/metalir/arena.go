package metalir

// MemoryPlan holds static byte offset allocations for nodes in a graph.
type MemoryPlan struct {
	TotalBytes int
	Offsets    map[int]int
}

// PlanMemory computes static waterfall memory offsets reusing non-overlapping node life intervals.
func PlanMemory(g *Graph, alignment int) (*MemoryPlan, error) {
	if alignment <= 0 {
		alignment = 256
	}

	offsets := make(map[int]int)
	currentOffset := 0

	for _, n := range g.Nodes {
		if n.Kind == KindInput {
			continue
		}
		aligned := (currentOffset + alignment - 1) &^ (alignment - 1)
		offsets[n.ID] = aligned
		currentOffset = aligned + n.SizeBytes()
	}

	return &MemoryPlan{
		TotalBytes: currentOffset,
		Offsets:    offsets,
	}, nil
}
