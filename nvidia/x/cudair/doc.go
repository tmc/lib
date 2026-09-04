// Package cudair provides a typed intermediate representation and memory arena
// planner for CUDA source-generation tools.
//
// # Graph Representation
//
// Computation graphs are constructed from elementary tensor operations (Elementwise,
// MatMul, Reduce). The topological structure allows compiler passes such as dead-code
// elimination, kernel fusion ([github.com/tmc/lib/nvidia/x/cudafuse]), and
// memory buffer reuse planning.
// Graph construction does not report errors. Call [Graph.Validate] before
// consuming a graph; [PlanMemory] and cudafuse do this automatically. Graphs
// and their nodes must not be mutated concurrently.
//
// # Arena Allocation Planning
//
// [PlanMemory] calculates tensor live ranges and assigns non-overlapping
// intermediate buffers offsets in one arena. The package does not allocate
// device memory, compile generated source, or execute CUDA kernels.
package cudair
