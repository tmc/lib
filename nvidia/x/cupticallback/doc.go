// Package cupticallback provides managed lifecycle ownership, thread safety,
// reentrancy control, and typed event dispatch for NVIDIA CUPTI callbacks.
//
// Callbacks observe runtime/driver API invocations, resource lifecycles,
// stream synchronizations, and NVTX markers synchronously on the calling thread.
//
// Collection is qualified across CUPTI versions (24, 26, 28, 130001, 130101,
// 130200, 130202, 130301, 130400).
// A later release of the same CUDA major uses the newest of those layouts at
// or below it; a new major or an older release is refused.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cupticallback
