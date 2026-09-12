// Package cudalayer provides experimental CUDA decoder-layer kernels and JIT
// compilation for the llama inference example.
//
// [Compile] uses NVRTC to produce a compiled module for a compute capability.
// Callers load the module and launch its kernels; compilation alone does not
// execute a decoder layer. [ProjectionNormConfig.Validate] checks projection
// shape constraints, and [ProjectionNormConfig.SharedMemoryBytes] reports the
// dynamic shared memory required by ProjectionResidualRMSNorm.
//
// The API and kernel layout are not yet stable. The package is exercised by a
// CUDA KV-cache proof path, not by the example's production serving backend.
package cudalayer
