// Package cupticallback provides managed lifecycle ownership, thread safety,
// reentrancy control, and typed event dispatch for NVIDIA CUPTI callbacks.
//
// Callbacks observe runtime/driver API invocations, resource lifecycles,
// stream synchronizations, and NVTX markers synchronously on the calling thread.
//
// Collection is qualified across CUPTI versions (24, 26, 28, 130001, 130101,
// 130200, 130202, 130301, 130400).
package cupticallback
