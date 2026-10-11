// Package cudaregress turns admitted CUDA performance and diagnostic incidents
// into bounded, reproducible regression trials.
//
// When a performance regression or execution bottleneck incident is admitted
// via package [github.com/tmc/lib/nvidia/x/cudadiag] or package
// [github.com/tmc/lib/nvidia/x/cudaprof/tracecontract], ad-hoc re-executions
// risk non-reproducible outcomes due to environment drift, unverified binary
// closures, and uncontrolled background noise.
//
// cudaregress formulates a strict [ExperimentPlan] referencing immutable source,
// binary, library, and input artifacts by SHA-256 digests. Unavailable or corrupt
// original artifacts are treated as reconstruction blockers rather than guessed.
// Preflight validation detects environment drift (device, driver, toolkit, clock,
// or binary changes) and either produces a visible refusal or creates an explicitly
// identified fork experiment. Arbitrary capture-supplied shell commands are never
// executed.
//
// Execution proceeds through interleaved ABBA rounds via package
// [github.com/tmc/lib/nvidia/x/cudatrial] with baseline-versus-baseline null controls
// to establish empirical noise floors. Latency is decomposed into non-additive
// channels ([cudatrial.TimingWallElapsed], [cudatrial.TimingSummedKernel],
// [cudatrial.TimingGPUBusy], and [cudatrial.TimingAPICall]), preserving missing
// architectural families versus measured zero durations, and hardware/resource
// changes versus performance changes.
//
// Receipts are separated into [PlanReceipt], [ExecutionReceipt], and
// [ComparisonReceipt], and can be recorded into immutable profile history via
// package [github.com/tmc/lib/nvidia/x/cudaprofilehistory].
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaregress
