// Package cudaworkload tracks and correlates Go application operations with host submissions and device activities.
//
// High-level Go applications enqueue asynchronous work to NVIDIA GPUs through
// CUDA driver and runtime APIs, CUPTI correlation channels, or CUDA Graph execution.
// Because the Go runtime multiplexes goroutines across OS threads and GPU execution
// is asynchronous and pipelined, associating application operations with GPU work
// cannot rely on temporal proximity or naive timestamp joins without introducing
// false exact joins.
//
// # Identity Model and Lifetime
//
// Workload correlation is built on explicit, verifiable identity:
//   - [OpID]: Monotonically increasing 64-bit identifier allocated per operation scope.
//     An OpID is unique within a process generation. Zero represents an invalid or unassigned ID.
//   - [ProcessGenID]: Pairs an OS PID with a monotonic generation counter (or boot-time
//     instance ID). When an OS recycles a PID after process exit, the generation counter
//     changes, preventing stale submissions or device activities from falsely joining across
//     process lifetimes.
//   - [WorkloadIdentity]: Preserves process-generation identity, workload names, run IDs,
//     and operational labels.
//   - [ContextStreamID]: Identifies the CUDA context handle and CUDA stream handle.
//   - [GraphReplayGeneration]: Identifies a specific replay instance of an instantiated
//     CUDA graph. Capturing a CUDA graph assigns node identities once, while replaying the
//     graph executes those nodes repeatedly. Tracking the replay generation counter
//     ensures activities in iteration N are not conflated with iteration N+1.
//
// # Scoped Correlation and Thread Pinning
//
// CUPTI external-correlation push and pop APIs are thread-local: they operate on the
// calling OS thread's correlation stack. Because the Go runtime can preempt and migrate
// goroutines between OS threads at arbitrary preemption points, pushing an external
// correlation ID without thread locking risks popping on a different OS thread or attributing
// activities to unrelated goroutines.
//
// [Scope.NativeCall] solves this by locking the calling goroutine to its current OS thread
// via [runtime.LockOSThread] strictly for the duration of the native submission call, pushing
// the correlation ID, invoking the native call, and in a defer popping the correlation ID and
// unlocking the thread with [runtime.UnlockOSThread]. Thread pinning is restricted to the
// minimal submission window rather than the lifetime of the application operation.
//
// Invariant: Scoped cleanup is balanced under all exit paths, including errors, panics,
// and context cancellation.
//
// # Join Semantics
//
// The [Joiner] correlates [HostSubmission] records with [DeviceActivity] records.
// Outcomes are explicitly classified into four categories:
//   - [OutcomeExact]: Explicit 1-to-1 correlation independently established via matching
//     process generation, native correlation ID, graph replay generation, or CUPTI external ID.
//   - [OutcomeAssociated]: Circumstantial association based on context, stream, or thread
//     provenance where explicit correlation was not recorded.
//   - [OutcomeAmbiguous]: Multiple conflicting operations or external IDs claim the activity.
//   - [OutcomeMissing]: Activity or submission has no counterpart due to late attach, buffer
//     FIFO eviction, or unannotated background work.
//
// Timestamp proximity alone is never promoted to an exact join.
package cudaworkload
