package metalir

import "fmt"

// MemoryPlan holds static byte offset allocations for nodes in a graph.
type MemoryPlan struct {
	TotalBytes int
	Offsets    map[int]int
}

// PlanMemory computes static waterfall memory offsets reusing non-overlapping node life intervals.
func PlanMemory(g *Graph, alignment int) (*MemoryPlan, error) {
	if alignment == 0 {
		alignment = 256
	}
	if alignment < 0 || alignment&(alignment-1) != 0 {
		return nil, fmt.Errorf("alignment must be a positive power of two")
	}
	if err := g.Validate(); err != nil {
		return nil, fmt.Errorf("plan memory: %w", err)
	}

	offsets := make(map[int]int)
	total := 0
	last := make(map[*Node]int)
	for i, n := range g.Nodes {
		for _, in := range n.Inputs {
			last[in] = i
		}
	}
	type block struct{ offset, size, until int }
	var blocks []block

	for i, n := range g.Nodes {
		if n.Kind == KindInput {
			continue
		}
		size := (n.SizeBytes() + alignment - 1) &^ (alignment - 1)
		chosen := -1
		for j := range blocks {
			if blocks[j].until < i && blocks[j].size >= size {
				chosen = j
				break
			}
		}
		until, ok := last[n]
		if !ok {
			until = len(g.Nodes)
		}
		if chosen >= 0 {
			offsets[n.ID] = blocks[chosen].offset
			blocks[chosen].until = until
		} else {
			offsets[n.ID] = total
			blocks = append(blocks, block{total, size, until})
			total += size
		}
	}

	return &MemoryPlan{
		TotalBytes: total,
		Offsets:    offsets,
	}, nil
}
