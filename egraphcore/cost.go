package egraphcore

// ASTSizeCost returns one unit for the current node plus the supplied costs
// for each child use.
func ASTSizeCost(childCosts []float64) float64 {
	cost := 1.0
	for _, childCost := range childCosts {
		cost += childCost
	}
	return cost
}
