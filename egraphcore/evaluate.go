package egraphcore

// EvaluateNodeCost calls model with a backend-cloned node and a copy of the
// child costs. cloneNode should copy mutable node data that the model could
// otherwise use to mutate graph state.
func EvaluateNodeCost[Node any](
	model CostModel[Node],
	node Node,
	childCosts []float64,
	cloneNode func(Node) Node,
) float64 {
	return model.NodeCost(cloneNode(node), append([]float64(nil), childCosts...))
}
