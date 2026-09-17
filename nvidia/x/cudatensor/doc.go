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
package cudatensor
