package metalgraphexec

import (
	"fmt"
)

// CommandNode represents a single compute dispatch command in the Metal graph.
type CommandNode struct {
	Name            string
	PipelineName    string
	BufferIndices   []int
	GridSize        [3]int
	ThreadgroupSize [3]int
}

// ExecutionGraph holds a static sequence of compute command nodes ready for single-pass execution.
type ExecutionGraph struct {
	Nodes []CommandNode
}

// New creates an empty Metal execution graph.
func New() *ExecutionGraph {
	return &ExecutionGraph{
		Nodes: make([]CommandNode, 0),
	}
}

// AddDispatch appends a compute dispatch step to the execution graph.
func (g *ExecutionGraph) AddDispatch(name, pipeline string, buffers []int, grid, tg [3]int) {
	g.Nodes = append(g.Nodes, CommandNode{
		Name:            name,
		PipelineName:    pipeline,
		BufferIndices:   buffers,
		GridSize:        grid,
		ThreadgroupSize: tg,
	})
}

// NodeCount returns the number of command dispatches in the graph.
func (g *ExecutionGraph) NodeCount() int {
	return len(g.Nodes)
}

// Validate checks that all nodes have valid dimensions.
func (g *ExecutionGraph) Validate() error {
	for i, node := range g.Nodes {
		if node.PipelineName == "" {
			return fmt.Errorf("metalgraphexec: node %d (%q) has empty pipeline name", i, node.Name)
		}
		if node.GridSize[0] <= 0 || node.ThreadgroupSize[0] <= 0 {
			return fmt.Errorf("metalgraphexec: node %d (%q) has invalid grid/tg dimensions: %v / %v",
				i, node.Name, node.GridSize, node.ThreadgroupSize)
		}
	}
	return nil
}
