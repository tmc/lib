// Package cudablock provides high-performance CUDA thread-block and warp-level primitives
// for collective reductions, prefix scans, and cooperative tile staging into shared memory.
//
// The embedded header ([Header]) can be directly passed to [github.com/tmc/lib/nvidia/x/nvrtcx]
// using [nvrtcx.WithHeader]("cudablock.cuh", cudablock.Header) to enable optimized cooperative
// block algorithms in JIT-compiled kernels without external CUDA SDK dependencies.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudablock
