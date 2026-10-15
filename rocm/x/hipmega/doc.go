// Package hipmega sketches single-device HIP persistent-grid admission using
// the existing occupancy and cooperative-launch bindings. It checks device and
// function geometry, shared memory and whole-grid residency before submission.
// Missing co-resident progress, cluster primitives and TMEM are refused.
//
// Launch queries limits and occupancy for each request. HIPOperations supplies
// the real binding adapter; injected operations are host contract tests, not HIP
// execution evidence. This package does not emit a worker, implement a task
// executor, inspect kernel instructions or prove numerical equivalence.
// No shared-memory opt-in or function configuration is performed; the queried
// limits must already permit the requested launch.
//
// The caller owns the initialized runtime, current context, matching function
// and stream, kernel geometry, arguments and all buffers through actual stream
// completion. Operations for a current context must run on the appropriate
// locked OS thread. Submission or cancellation is not completion. Kernel waits,
// memory ordering, bounded termination and buffer epochs remain kernel/caller
// responsibilities; multi-device progress is unsupported.
//
// The existing driver package attempts library loading during import init.
// CODEGEN_SKIP_BINDINGS=1 prevents loading in host tests. Constructing the new
// adapter makes no HIP calls, but importing this package is not a globally lazy
// runtime initialization contract. No ROCm device execution is qualified here.
package hipmega
