// Package rankgroup runs the ranks of a tensor-parallel group from one
// process, either all on one GPU or one per GPU, and sums buffers across
// them through an [AllReducer].
//
// [OnDevice] places N ranks on one device, so multi-rank code paths can be
// checked end to end without a multi-GPU node. Its [Mode] picks how the
// ranks share the device:
//
//   - [Serial]: one stream; rank r's work runs after rank r-1's.
//   - [Streams]: one stream per rank; ranks overlap, ordered only by
//     [Group.Barrier] and the all-reduce.
//   - [Green]: one green context ([github.com/tmc/lib/nvidia/x/cudagreen])
//     and stream per rank, each on its own SMs, as each rank would own a
//     GPU.
//
// [OnDevices] places one rank on each of several devices.
//
// Two [AllReducer]s sum float32 buffers: [DeviceAllReduce] launches, on
// each rank's stream, a kernel that reads every rank's input through its
// device address and sums in rank order, so every rank gets the same bits.
// It needs every input addressable from every rank: one device, or peers
// with P2P access. [NCCLAllReduce] calls ncclAllReduce on communicators made
// by ncclCommInitAll, which needs one distinct GPU per rank.
//
// The native tests run under CUDA_RANKGROUP_NATIVE=1 and an inherited GPU
// lock. On one GB10 (DGX Spark, 48 SMs, driver 13.0, NCCL 2.31) on
// 2026-10-28, DeviceAllReduce summed 4099 floats exactly over 1, 2 and 4
// ranks in every mode (green partitions of 32, 16+16 and 4×8 SMs: the
// driver makes 4 groups of 8), and NCCLAllReduce ran with one rank; NCCL
// with 2 or 4 ranks skipped, needing that many GPUs. No run has crossed
// NVLink or a second GPU. examples/nvidia/llama-infer -tp runs a
// tensor-parallel LLaMA on a Group.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch
// releases.
package rankgroup
