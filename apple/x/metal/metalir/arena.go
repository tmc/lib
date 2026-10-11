package metalir

import "fmt"

// MemoryPlan holds static byte offset allocations for nodes in a graph.
type MemoryPlan struct {
	TotalBytes int
	Offsets    map[int]int
}

// PlanMemory assigns aligned, non-overlapping byte ranges to non-input nodes.
func PlanMemory(g *Graph, alignment int) (*MemoryPlan, error) {
	if g == nil {
		return nil, fmt.Errorf("metalir: nil graph")
	}
	if alignment <= 0 {
		alignment = 256
	}

	offsets := make(map[int]int)
	currentOffset := 0
	seen := make(map[int]bool)

	for _, n := range g.Nodes {
		if n == nil {
			return nil, fmt.Errorf("metalir: graph contains nil node")
		}
		if seen[n.ID] {
			return nil, fmt.Errorf("metalir: duplicate node ID %d", n.ID)
		}
		seen[n.ID] = true
		if n.Kind == KindInput {
			continue
		}
		if currentOffset > int(^uint(0)>>1)-(alignment-1) {
			return nil, fmt.Errorf("metalir: memory alignment overflows at node %q", n.Name)
		}
		aligned := ((currentOffset + alignment - 1) / alignment) * alignment
		size, err := n.SizeBytesChecked(nil)
		if err != nil {
			return nil, fmt.Errorf("metalir: size node %q: %w", n.Name, err)
		}
		if aligned > int(^uint(0)>>1)-size {
			return nil, fmt.Errorf("metalir: memory plan overflows at node %q", n.Name)
		}
		offsets[n.ID] = aligned
		currentOffset = aligned + size
	}

	return &MemoryPlan{
		TotalBytes: currentOffset,
		Offsets:    offsets,
	}, nil
}
