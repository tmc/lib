// Package egraph implements a lightweight, pure-Go E-Graph (equivalence graph)
// and equality saturation engine for term rewriting, program optimization, and
// kernel fusion.
//
// An E-Graph efficiently represents an exponential number of equivalent
// expressions as equivalence classes (e-classes). Equality saturation repeatedly
// applies non-destructive rewrite rules until a fixed point is reached, followed
// by extraction of the optimal expression according to a cost model.
//
// Basic usage:
//
//	eg := egraph.New()
//	x := eg.Add(egraph.NewLeaf("x"))
//	two := eg.Add(egraph.NewLeaf(2))
//	mul := eg.Add(egraph.NewNode("*", x, two))
//	shift := eg.Add(egraph.NewNode("<<", x, eg.Add(egraph.NewLeaf(1))))
//	eg.Union(mul, shift)
//	eg.Rebuild()
//
// Graph construction and rewriting run in Go. GPU candidate benchmarking
// additionally uses the CUDA driver and runtime compiler through the CUDA wrappers.
package egraph
