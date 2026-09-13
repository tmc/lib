package driver

// Nodes returns all nodes in the graph.
func (g Graph) Nodes() ([]GraphNode, error) {
	if err := checkLibSym("cuGraphGetNodes"); err != nil {
		return nil, err
	}
	var numNodes uint
	if err := toError(fngraphGetNodes(g, nil, &numNodes)); err != nil {
		return nil, err
	}
	if numNodes == 0 {
		return nil, nil
	}
	nodes := make([]GraphNode, numNodes)
	if err := toError(fngraphGetNodes(g, &nodes[0], &numNodes)); err != nil {
		return nil, err
	}
	return nodes[:numNodes], nil
}
