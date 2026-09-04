// Package moonep is a Go port of the host-side layer of MoonEP
// (github.com/MoonshotAI/MoonEP), an expert-parallel MoE communication
// library. It builds on package nvlshm for the cross-process NVLink shared
// buffers and implements the buffer layout, sizing math, and plan allocation
// expected by the supported exported MoonEP device kernels.
//
// Notation follows MoonEP: S tokens per rank, K experts per token (top-k),
// E total experts, R ranks, B prefetch slots, H hidden size, NvS dispatched
// token slots per rank.
//
// # What is and is not here
//
// This package provides the [Layout] computation, the [Buffer] owning the
// NVLink shared hidden/meta buffers plus local scratch, [CommPlan]
// allocation, and thin kernel loading/launch helpers ([LoadModule],
// [LaunchCooperative]).
//
// The device kernels themselves are not included. MoonEP's kernels are
// CUTLASS CuTe-DSL JIT programs (planning, dispatch, dispatch epilogue,
// combine prologue, combine, prefetch, grad reduce); they must be
// pre-compiled to cubins with the Python toolchain and loaded at runtime —
// see the kernels section of the README for the export procedure. The
// runtime synchronization protocol is entirely device-side (shared-memory
// flag barriers inside meta_buf), so the host layer here has no
// per-iteration role beyond launching kernels.
//
// # The meta buffer
//
// Each rank's meta chunk is an int32 array with the following regions, all
// offsets in int32 elements and every region 4-element (16-byte) aligned so
// vectorized device access is safe (see [Layout]):
//
//	[WeightsOff, +NvS)     per-slot fp32 route weights (bit-cast to int32)
//	[TpeOff, +R*E)         tokens-per-expert gather (rank 0 is the target)
//	[PlanOff, +planElems)  planning outputs; the leading 3*E*R block is
//	                       broadcast to all ranks via the multicast mapping
//	[Topk0Off, +N4)        topk offload scratch
//	[OrderOff, +N4)        C1 order
//	[Order0Off, +N4)       rank 1's replay of rank 0's C1
//	[BarrierOff, +3)       cross-rank phase barrier (2 signals + 1 counter)
//	[SrcInfoOff, +NvS)     slot provenance: src_rank*NvS + offv, -1 = empty
//
// The barrier region must start all-zero on every rank; [NewBuffer] zeroes
// it collectively before returning. Nothing else needs initialization.
package moonep
