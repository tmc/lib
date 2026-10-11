// Package egraph implements equality graphs, rewrite rules, bounded
// saturation, cost-based extraction, and Metal Shading Language generation.
//
// Build a graph with New, add nodes with Add, and define rewrites with
// NewRewrite or the pattern constructors. Run rules with SaturateWithReport or
// ExecuteSchedule, then select expressions with ExtractBest or ExtractTopK.
//
// The package also provides analyses, typed relations and function tables,
// dimension search spaces, candidate search, and operator registries.
// An EGraph is mutable and is not safe for concurrent use.
//
//	g := egraph.New()
//	x := g.Add(egraph.NewLeaf("x"))
//	y := g.Add(egraph.NewLeaf("y"))
//	root := g.Add(egraph.NewNode("add", x, y))
//	expr, _, err := egraph.ExtractBest(g, root, egraph.AstSizeCost{})
//	_ = expr
//	_ = err
package egraph
