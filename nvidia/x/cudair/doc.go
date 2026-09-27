// Package cudair provides a typed Intermediate Representation (IR) graph,
// immutable validated snapshots, and waterfall memory arena planning for GPU tensor graphs.
//
// # Graph Representation
//
// Computation graphs are constructed from an authoritative operation set:
// Input, Constant, Add, Multiply, Sub, Div, Neg, Exp, Log, Sqrt, Relu, Gelu,
// Matmul, ReduceSum, ReduceMax, Reshape, Transpose, Slice, and Concat.
//
// Every node carries explicit metadata including [DataType], shape extents,
// [cudashape.Layout], strides, and storage effects ([StorageEffect]).
//
// # Immutable Validated Snapshots
//
// A mutable [Graph] can be frozen and validated using [Graph.Snapshot].
// Validation ensures:
//   - The graph is a directed acyclic graph (no cycles).
//   - Nodes have unmixed single-graph ownership (cross-graph inputs are rejected).
//   - All operations conform to the authoritative semantic registry.
//   - Shapes, ranks, reduction axes, permutations, and dtypes are strictly compatible.
//   - Graph outputs are valid and reachable.
//
// # Arena Allocation Planning
//
// The [PlanMemory] function calculates tensor live ranges and overlays non-overlapping
// intermediate tensor buffers into a single contiguous device memory scratchpad,
// minimizing GPU VRAM allocations while respecting storage aliases and memory limits.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudair
