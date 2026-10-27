// Package tcgen05 implements Blackwell (sm_100a, sm_103a, sm_100f, sm_103f) and
// Vera Rubin (sm_107a, sm_107f) Tensor Memory (TMEM) Fifth- and Sixth-Generation
// Tensor Core (tcgen05 / PTX ISA 9.4), FlashAttention-4 paged and absorbed
// Multi-head Latent Attention (MLA), 2:4 structured sparsity and spcompress,
// and Cluster Launch Control (CLC) kernel emitters, hardware descriptors, and
// GB200/GB300/VR200 residency planners.
//
// Kernels emitted by this package route every PTX instruction through a
// self-contained device prelude so the same CUDA C++ source compiles for
// sm_100a, sm_103a, and sm_107a with nvcc and executes on the host CPU via
// cudahostemu:
//
//   - [TCGen05GEMM] and [TCGen05SparseGEMM] emit dense and 2:4 structured
//     sparse `.cta_group::2` block-scaled NVFP4/MXFP4/MXFP8 UMMA pipelines
//     with TMEM lifecycle tracking (`tcgen05.alloc` through `tcgen05.dealloc`).
//   - [TCGen05PagedAttention] and [TCGen05MLAPagedAttention] emit
//     FlashAttention-4 and DeepSeek MLA absorbed-KV 5-role warp-specialized
//     ping-pong TMEM pipelines with `scale_vec::4X` and 2x SFU exponential
//     emulation on sm_103a and sm_107a.
//   - [RubinBlockScaledUMMA] emits Vera Rubin sm_107a / sm_107f PTX ISA 9.4
//     dual-CTA `.cta_group::2` block-scaled NVFP4/MXFP4/MXFP6/MXFP8 kernels
//     with TMA L2 multicast (`cp.async.bulk.tensor.2d.multicast::cluster`),
//     hardware `ld.global.red` / `st.global.spcompress` activation compression,
//     and 256 KB per-SM TMEM budgeting.
//
// Each tcgen05 kernel also produces a formal TraceEvent sequence verified by
// cudatmem.VerifyTrace.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package tcgen05
