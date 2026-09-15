package driver

import (
	"fmt"
	"runtime"
)

// NodesBounded returns the graph's nodes, rejecting counts above limit.
// The graph must not be changed concurrently. The returned slice owns the
// complete output array required by cuGraphGetNodes.
func (g Graph) NodesBounded(limit int) ([]GraphNode, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("driver: invalid graph node limit %d", limit)
	}
	if err := checkLibSym("cuGraphGetNodes"); err != nil {
		return nil, err
	}
	return graphNodes(g, limit, fngraphGetNodes)
}

func graphNodes(g Graph, limit int, query func(Graph, *GraphNode, *uint) Result) ([]GraphNode, error) {
	if limit <= 0 {
		return nil, fmt.Errorf("driver: invalid graph node limit %d", limit)
	}
	var n uint
	if err := toError(query(g, nil, &n)); err != nil {
		return nil, err
	}
	if n > uint(limit) {
		return nil, fmt.Errorf("driver: graph node count %d exceeds limit %d", n, limit)
	}
	if n == 0 {
		return nil, nil
	}
	nodes := make([]GraphNode, n)
	var pin runtime.Pinner
	pin.Pin(&nodes[0])
	defer pin.Unpin()
	if err := toError(query(g, &nodes[0], &n)); err != nil {
		return nil, err
	}
	if n != uint(len(nodes)) {
		return nil, fmt.Errorf("driver: graph node count changed during query")
	}
	return nodes, nil
}
