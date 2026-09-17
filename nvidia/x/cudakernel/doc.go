// Package cudakernel provides immutable kernel artifact descriptors, ABI parameter
// schemas, fixed-width argument packing, and hardware resource contracts for CUDA execution.
//
// Artifact represents a compiled CUDA kernel image (PTX, CUBIN, LTOIR, or TileIR)
// along with its cryptographic digest, entry point symbol, parameter schema, target
// compute architecture, and hardware resource requirements.
//
// # Hardware Resource Model
//
// Following the NVIDIA CUBIN inspection report schema, cudakernel strictly distinguishes
// between:
//   - PeakLiveGPR: the maximum number of general-purpose registers simultaneously live
//     during kernel execution, as measured by register liveness analysis.
//   - AllocatedRegisters: the physical register resources allocated to each thread
//     by the compiler/driver, determined by hardware allocation quanta (e.g. 256 registers
//     per warp unit on NVIDIA Blackwell SM121).
//
// # ABI and Argument Packing
//
// Schema describes the parameter layout of a kernel according to NVIDIA parameter
// bank (c[0x0]) alignment rules. Fixed-width packing verifies scalar sizes and overflow
// bounds, rejects unresolved or Go heap pointers, and produces memory layouts suitable
// for cuLaunchKernel or cuLaunchKernelEx without persisting host heap addresses.
//
// # Two-Tier JIT Options
//
// JITOptions captures driver-side JIT compilation parameters (such as MaxRegisters,
// ThreadsPerBlock, and OptLevel) that must be passed to cuModuleLoadDataEx when
// loading PTX artifacts to prevent unconstrained register allocation and SM occupancy drops.
package cudakernel
