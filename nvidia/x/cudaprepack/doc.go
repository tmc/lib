// Package cudaprepack provides ahead-of-time (AOT) hardware layout transformations
// for NVIDIA Blackwell TMA (cp.async.bulk.tensor.2d) tensor boxes and microscaled sub-byte formats.
//
// # Hardware Disambiguation
//
// NVIDIA Blackwell architectures have distinct hardware execution capabilities:
//
//   - Blackwell GB10 (sm_121a, cc 12.1): Supports Tensor Memory Accelerator
//     (TMA, cp.async.bulk), mbarrier, and multimem, but rejects tcgen05 and wgmma.
//     Tensor boxes must adhere to 16-byte address alignment and 128-byte box boundary limits.
//
//   - Blackwell GB200 / B100 (sm_100a): Supports 5th-generation Tensor Cores (tcgen05),
//     hardware microscaling swizzling, and 2:4 structured sparsity (tcgen05.mma.ws.sp).
//
// # Sub-Byte Formats
//
// The package repacks sub-byte formats (IQ2_XXS, Q8_0, FP8, and MXFP4) into
// contiguous hardware tiles so TMA bulk transfers can stream tiles directly
// into shared memory with zero register bit-manipulation.
package cudaprepack
