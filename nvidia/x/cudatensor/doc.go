// Package cudatensor provides a fluent, checked tensor frontend for constructing
// computation graphs with deferred symbolic execution, shape verification, and frontend
// error propagation.
//
// # Fluent Graph Construction
//
// Tensors wrap underlying [cudair.Node] definitions. Methods like [Tensor.Add],
// [Tensor.Mul], and [Tensor.MatMul] record operations into an associated [cudair.Graph].
//
// # Error Propagation
//
// The fluent API follows standard Go builder conventions with sticky error propagation:
// operations on an invalid or errored tensor preserve the error ([Tensor.Err]),
// eliminating runtime panics on shape mismatches, cross-graph operands, or invalid axes.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// # Einsum Graphs and SOL Bounds
//
// [CanonicalizeRanks] converts ordered einsum DAGs into stable rank labels for
// comparison and [ExportTimeloopWorkload] writes canonical graphs in Timeloop
// v0.5 form. [ComputeSOLBounds] reports per-operation unfused and fused bounds
// alongside SOLAR-style whole-graph estimates. SOLAR's shipped unfused and
// fused runtimes apply one roofline to graph totals, which is optimistic
// relative to the stated per-operation design intent. Its fused and
// fused-prefetched byte totals are identical; those estimates differ only in
// how the roofline is combined. [CanonicalGEMMDataMovement] maps only supported
// rank-2 or batched GEMM equations to the B/M/N/K dimensions and A/B/C tensor
// roles accepted by cudaroofline; other equations and unmapped dimensions fail.
package cudatensor
