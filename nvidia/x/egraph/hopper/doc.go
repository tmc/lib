// Package hopper describes typed tensor kernels as e-graph terms and lowers
// them to CUDA C++ across Hopper (sm_90a), Blackwell datacenter and workstation
// (sm_100a, sm_103a, sm_120a, sm_121a), and Vera Rubin (sm_107a, sm_107f) with
// PTX ISA 9.4 support.
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
// extracts by [Cost], a stack of layers layer by layer ([Extract]);
// [Compile] lowers the result to a [Plan] of [SkinnyGEMM], [TileGEMM] and
// [RowKernel] launches. A skinny gemm reads a cast of an F32 input itself
// ([SkinnyGEMM.X]) and applies SwiGLU in its epilogue ([SkinnyGEMM.GLU]),
// and [Plan.FuseTails] runs a single-CTA row kernel that follows it, such
// as a decode RMSNorm, in its last block. For wide rows, setting
// [RowKernel.Cluster] emits an 8-CTA distributed shared memory reduction
// schedule with a matching single-block wide schedule for large batches, and
// [SourceTVMFFI] emits TVM-FFI host wrapper headers for load_jit.
//
// [CompileFor] lowers for a named GPU. On targets with tcgen05 ([UMMATarget]:
// sm_100a, sm_103a and sm_110a, as probed in cudadevicetable) BF16 and
// NVFP4 gemms become [UMMAGEMM]s, TMA-fed tcgen05.mma kernels accumulating
// in tensor memory, with an [NVFP4Quant] of the activations for NVFP4.
// These kernels are compiled with ptxas 13.4 and run under cudahostemu's
// descriptor-level emulation; they have not run on a tcgen05 GPU.
//
// Multi-architecture lowering and PTX ISA 9.4 emitters complement the e-graph
// pipeline:
//
//   - [SelectArchDispatchPlan] selects the optimal architecture execution
//     strategy across sm_90a (WGMMA + TMA), sm_100a / sm_103a (tcgen05 TMEM +
//     2-CTA Cluster Launch Control), sm_107a / sm_107f (Vera Rubin PTX ISA 9.4
//     block-scaled UMMA + NVLink 6), and sm_120a / sm_121a (Blackwell RTX 5090
//     and DGX Spark register-file `mma.sync.aligned.block_scale` without TMEM
//     or cluster multicast) with online hardware telemetry adaptation.
//   - [SM120BlockScaleGEMM] emits sm_120a / sm_121a register-file block-scaled
//     NVFP4/MXFP8 kernels within the 99 KB shared memory envelope.
//   - [RubinPTX94BlockScaleGEMM] and [PTX94VectorizedEpilogue] emit Vera Rubin
//     sm_107a / sm_107f PTX ISA 9.4 `tcgen05.mma.cta_group::2.kind::mxf4nvf4`,
//     128-bit vectorized `.v4.bf16x2` / `.v4.f16x2` SIMD epilogues, and
//     hardware `cvt.rn.satfinite.e2m1x2.f32` quantization intrinsics.
//   - [DynamicNVFP4Quantize] and [DynamicMXFP8Quantize] emit single-pass warp
//     amax reduction and hardware FP4/MXFP8 dynamic activation quantization.
//
// Kernels that terms do not yet describe are emitted directly:
// [PagedAttention] and [MTPPagedAttention] (paged FP8 split-KV and multi-token
// tree-verify attention), [RowTopK] (biased top-k routing), [ExpertAlign]
// (expert-sorted MoE routes for a grouped [TileGEMM]), and [ExpertSum] (weighted
// MoE reduction).
//
// Higher-level model compilation, qualification, TileIR lowering, and
// speculative decoding live in subpackages built on top of hopper:
//
//   - [github.com/tmc/lib/nvidia/x/egraph/model] parses HuggingFace
//     architectures, lowers transformer decoder layers to e-graph [Plan]s and
//     multi-bucket dispatchers, repacks safetensors weights, and packages
//     multi-architecture fatbinaries.
//   - [github.com/tmc/lib/nvidia/x/egraph/qualify] provides PTX/SASS static
//     qualification, autotuning, and profile-guided schedule refinement.
//   - [github.com/tmc/lib/nvidia/x/egraph/tilebridge] lowers schedules to CUDA
//     TileIR modules and cuTile JSON descriptors.
//   - [github.com/tmc/lib/nvidia/x/egraph/specdecode] implements Philox-4x32-10
//     Gumbel-max sampling, speculative decoding loops, tree verification, and
//     adaptive early-exit routing.
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
//   - cp.async.bulk writes 0xff the same way, and its data lands only when
//     a thread waits on the mbarrier phase the copy completes; a satisfied
//     mbarrier wait returns only once no other thread of the block can run,
//     so a missing or wrong-parity wait, or a producer that refills a stage
//     its consumers have not released, reads NaN or deadlocks loudly;
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
