// Package hopper describes typed tensor kernels as e-graph terms and lowers
// them to CUDA C++ for sm_90a (H100, H200).
//
// Tensors are stored as F32, BF16, E4M3 (float8_e4m3fn), E2M1 (NVFP4 codes)
// or integers; values in registers are FP32. The Go codecs ([BF16Bits],
// [E4M3Bits], [FromE2M1], [DequantNVFP4]) round exactly as the device
// conversions do, so CPU references and kernels agree bit for bit where the
// arithmetic order allows.
//
// Terms ([Builder]) describe tensor programs: typed Tensor leaves, slices
// and reshapes, elementwise [Lambda] maps with NumPy broadcasting, last-axis
// reductions, casts, and x·Wᵀ matmuls over BF16, BF16 hi+lo or NVFP4
// weights. [Eval] is their CPU reference. [Optimize] saturates them with
// [Rules] and [FuseEpilogues] (maps folded into a GEMM's epilogue) and
// extracts by [Cost]; [Compile] lowers the result to a [Plan] of
// [SkinnyGEMM], [TileGEMM] and [RowKernel] launches. For wide rows, setting
// [RowKernel.Cluster] emits an 8-CTA distributed shared memory reduction
// schedule with a matching single-block wide schedule for large batches, and
// [SourceTVMFFI] emits TVM-FFI host wrapper headers for load_jit.
//
// Kernels that terms do not yet describe are emitted directly:
// [PagedAttention] (paged FP8 split-KV attention), [RowTopK] (biased top-k
// routing) and [ExpertAlign] (expert-sorted MoE routes for a grouped
// [TileGEMM]).
//
// Emitted CUDA uses raw integer types for low-precision data and routes every
// PTX instruction through a small device prelude, so the same source compiles
// with nvcc or NVRTC for sm_90a and, for tests, as host C++ against an
// emulation header (one thread per CUDA thread, emulated shuffles, barriers
// and tensor-core fragments).
//
// # Testing generated kernels without a GPU
//
// The tests use four tools that work for any kernel written against the
// prelude.
//
// The emulator, testdata/emu.h, is included first (clang++ -std=c++20
// -include emu.h) and runs a launch on the host. Each CUDA thread is a
// std::thread and blocks run one after another, so it finds layout,
// indexing and synchronization errors, not inter-block races. It is
// adversarial where a GPU may differ from a naive host:
//
//   - cp.async writes 0xff (NaN) to its destination when issued and the
//     data lands only when cp.async.wait_group retires its group, so a
//     too-shallow wait or a read ahead of its barrier reads NaN;
//   - each block starts with its static and dynamic shared memory filled
//     with 0xff, so a kernel that reads shared memory it did not write in
//     that block reads NaN (EG_EMU_ZERO_SMEM turns this off, as a control);
//   - the tests pass output buffers filled with 0xff, as a fresh
//     allocation might hold, so an unwritten output is NaN.
//
// Results are compared with float64 references written independently of
// the kernel ([PagedAttention.Reference], [TileGEMM.Reference], ...).
//
// Mutation runs check that those tests can fail: each mutant replaces one
// exact line of emitted CUDA (a barrier, a wait depth, an index, a fill),
// runs the emulated tests and must make at least one fail; a mutant that
// survives is either a missing test or code that does nothing. A mutant
// that fails only under the poison modes must survive with them off.
//
// Golden files (testdata/golden/*.cu, rewritten by go test -update) are the
// emitted translation units. TestGoldensCompile compiles each for sm_90a
// with nvcc when it is on PATH and fails on errors. Each golden also has a
// PTX census (testdata/golden/*.census.json): its global loads and stores
// by width and the loads issued after the first barrier. TestGoldenCensus
// checks, without nvcc, that every census matches its golden and that
// every kernel issues its loads before its first barrier and stores
// adjacent 16-bit values together, naming the goldens exempt from each
// rule and why; go test -update-census rewrites the censuses with nvcc
// ($CUDACOMPILECHECK_NVCC, run through $CUDACOMPILECHECK_EXEC if set).
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package hopper
