// Package cudafuse compiles sub-graphs of elementwise and reduction operations into
// single fused CUDA mega-kernels.
//
// Fusing multiple memory-bandwidth-bound operators (e.g. Add + ReLU + Multiply + Exp)
// keeps intermediate values entirely in GPU register files and shared memory, eliminating
// global memory DRAM round-trips and achieving near-roofline memory efficiency.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudafuse
