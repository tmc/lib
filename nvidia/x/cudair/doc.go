// Package cudair provides a typed Intermediate Representation (IR) graph and waterfall
// memory arena planner for GPU tensor graphs.
//
// # Graph Representation
//
// Computation graphs are constructed from elementary tensor operations (Elementwise,
// MatMul, Reduce). The topological structure allows compiler passes such as dead-code
// elimination, kernel fusion ([github.com/tmc/lib/nvidia/x/cudafuse]), and optimal
// memory buffer reuse planning.
//
// # Arena Allocation Planning
//
// The [ArenaPlanner] calculates tensor live ranges and overlays non-overlapping intermediate
// tensor buffers into a single contiguous device memory scratchpad, minimizing GPU VRAM allocations.
package cudair
