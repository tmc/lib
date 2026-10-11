package egraph

import "github.com/tmc/lib/apple/x/metal/metalshape"

// DimensionExpr is an immutable non-negative symbolic dimension expression.
type DimensionExpr = metalshape.Expr

// DimensionInterval is a closed, non-negative interval assumption.
type DimensionInterval = metalshape.Interval

func DimensionConstant(value int64) (DimensionExpr, error) {
	return metalshape.DimensionConstant(value)
}
func DimensionSymbol(name string) (DimensionExpr, error) { return metalshape.DimensionSymbol(name) }
func DimensionAdd(a, b DimensionExpr) DimensionExpr      { return metalshape.DimensionAdd(a, b) }
func DimensionMul(a, b DimensionExpr) DimensionExpr      { return metalshape.DimensionMul(a, b) }
func DimensionCeilDiv(a, b DimensionExpr) (DimensionExpr, error) {
	return metalshape.DimensionCeilDiv(a, b)
}
func DimensionFloorDiv(a, b DimensionExpr) (DimensionExpr, error) {
	return metalshape.DimensionFloorDiv(a, b)
}
func DimensionMod(a, b DimensionExpr) (DimensionExpr, error) { return metalshape.DimensionMod(a, b) }
func DimensionMin(a, b DimensionExpr) DimensionExpr          { return metalshape.DimensionMin(a, b) }
func DimensionMax(a, b DimensionExpr) DimensionExpr          { return metalshape.DimensionMax(a, b) }
