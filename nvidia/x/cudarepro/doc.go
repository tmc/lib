// Package cudarepro reduces recorded GPU failures without editing kernel source.
//
// Reduction considers a closed set of dimensions, in order: input values and
// lengths; each tensor shape extent; grid, block, and dynamic shared memory;
// and subsets of the campaign's recorded injection decisions. Unknown record
// fields are errors. Kernel source reduction is outside this package's scope.
// Recorded schedules replay against pristine source through cudaschedule.Replay;
// every measured run must report the digest of that exact replayed artifact.
// Empty decision subsets execute pristine source. The caller supplies the
// existing campaign runner and its explicit workload input contract.
//
// A candidate must preserve the recorded failure identity: the campaign verdict
// and its kind-specific signature. A mismatch identifies the logical output
// element and numerical comparison policy. An instrumented access violation
// identifies its violation kind and access site. Missing identity evidence is
// an error; another failure does not substitute for the recorded one.
//
// Reproduction measurements require an explicit positive replay count R. A
// candidate uses the same R and must reproduce at least as often as the fresh
// baseline. This conservative empirical rule has no statistical confidence
// claim. Each measurement makes at most R calls to the existing runner; it
// never retries invalid execution evidence. One reproduction in R attempts is
// a one-off, which callers must explicitly opt into reducing.
//
// ExportPackage emits a standalone, runnable Go test package into an isolated
// directory. The exported package preserves pristine and injected kernel source,
// reduced inputs, shape, launch geometry, recorded failure identity, and the
// required oracle (such as lane 52's memory bounds patch). For nondeterministic
// findings, DetectionAttempts derives a bounded attempt count from the measured
// reproduction rate for a target detection probability. Emitted tests verify
// artifact integrity and reproduce the defect, failing if the defect reproduces
// and passing on a fixed kernel. Without a device or native gate, tests report
// the shared pending sentinel rather than passing.
//
// Capture and Replay isolate a single CUDA launch for debugging. Capture
// copies entire backing allocations, preserving aliases and pointer offsets,
// and retains a native kernel image. It synchronizes the supplied stream;
// callers must exclude concurrent mutation and keep allocations alive. Graph
// capture, mutable module globals, embedded device pointers, external
// resources, and race reproduction are outside this single-launch model.
// Captures contain application data and are written only by explicit request.
// Native images and scalar layouts require a compatible GPU and host ABI.
package cudarepro
