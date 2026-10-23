// Package hopper describes typed tensor kernels as e-graph terms and lowers
// them to CUDA C++ for sm_90a (H100, H200).
//
// Tensors are stored as F32, BF16, E4M3 (float8_e4m3fn), E2M1 (NVFP4 codes)
// or integers; values in registers are FP32. The Go codecs ([BF16Bits],
// [E4M3Bits], [FromE2M1], [DequantNVFP4]) round exactly as the device
// conversions do, so CPU references and kernels agree bit for bit where the
// arithmetic order allows.
//
// Emitted CUDA uses raw integer types for low-precision data and routes every
// PTX instruction through a small device prelude, so the same source compiles
// with nvcc or NVRTC for sm_90a and, for tests, as host C++ against an
// emulation header (one thread per CUDA thread, emulated shuffles, barriers
// and tensor-core fragments).
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package hopper
