// Package egraph implements a lightweight, pure-Go E-Graph (equivalence graph)
// and equality saturation engine for term rewriting, program optimization,
// kernel fusion, declarative C++ Tile IR lowering, and sub-byte quantized inference.
//
// An E-Graph efficiently represents an exponential number of equivalent
// expressions as equivalence classes (e-classes). Equality saturation repeatedly
// applies non-destructive rewrite rules until a fixed point is reached, followed
// by extraction of the optimal expression according to a cost model.
//
// Rewrite rule suites:
//   - [TransformerTileRules]: fuses composite transformer layer subgraphs (RMSNorm -> QKV, Matmul -> SwiGLU / GeLU)
//     into unified declarative Tile IR operations (__tile_global__).
//   - [TMALayoutRules]: introduces hardware layout hints satisfying Tensor Memory Accelerator (TMA)
//     and Tensor Core alignment constraints (16-byte alignment, 2D box extents).
//   - [SubByteRules]: expands coverage for sub-byte quantized weight patterns (Q8_0, IQ2_XXS, FP8 e4m3/e5m2, FP4/e2m1),
//     cleanly separating PTX bit-unswizzling rewrites from Tensor Core tile matrix multiplies (HMMA / MMA).
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
