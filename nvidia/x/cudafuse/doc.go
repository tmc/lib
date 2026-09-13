// Package cudafuse compiles sub-graphs of elementwise operations into
// single fused CUDA kernels.
//
// Fusing multiple memory-bandwidth-bound operators (e.g. Add + ReLU + Multiply + Exp)
// keeps intermediate values entirely in GPU register files, eliminating
// global memory DRAM round-trips and achieving near-roofline memory efficiency.
package cudafuse
