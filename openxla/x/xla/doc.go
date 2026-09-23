// Package xla provides a high-level, idiomatic Go API for OpenXLA PJRT execution.
//
// The package manages hardware acceleration plugins (such as TPU, GPU, or CPU),
// compilation of StableHLO/MLIR computational graphs, device buffer lifecycle,
// and context-aware program execution.
//
// Close buffers and executables before closing their engine. Calls that close
// a resource must be externally synchronized with calls that use it. Close
// invalidates the resource even if native destruction reports an error. Execution
// waits for native completion even if its context is cancelled; cancellation
// does not interrupt PJRT work.
package xla
