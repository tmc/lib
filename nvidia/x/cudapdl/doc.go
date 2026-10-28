// Package cudapdl implements Programmatic Dependent Launch (PDL) region composition,
// eligibility evaluation, and hardware/software fallback policies for NVIDIA GPUs.
//
// Programmatic Dependent Launch allows producer CUDA kernels to trigger dependent
// consumer grids early via the PTX intrinsic griddepcontrol.launch_dependents, enabling
// consumer grid CTAs to overlap execution and execute independent preamble work
// before synchronizing at griddepcontrol.wait.
//
// # Hardware Architecture Support
//
// PDL requires NVIDIA Hopper or newer architectures (compute capability sm_90 or greater,
// such as sm_90, sm_90a, sm_100, sm_100a, sm_121, sm_121a) with probed target resource
// geometry in cudakernel.ProbedTargetSM. Target architectures below sm_90 or with
// unprobed estimated geometry are rejected during eligibility inspection and fall back
// to standard completion edges.
//
// # Phase Effect, Resource, and Vendor Kernel Requirements
//
// To participate in a programmatic dependent launch:
//   - The producer kernel must declare HasLaunchDependents, TriggerAfterSharedWrites,
//     and bounded BeforeLaunchDependents phase effects that write at least one handoff
//     memory resource.
//   - The consumer kernel must declare HasWait, an honest WaitPlacement
//     (WaitAfterIndependentPreamble with producer-independent BeforeWait effects and
//     HasIndependentWork=true, or WaitAtEntryBeforeDependentBody with empty BeforeWait
//     and HasIndependentWork=false), and AfterWait phase effects that read a producer
//     handoff resource without shared-resource hazards.
//   - Both kernels must supply bounded hardware resource requirements
//     (cudakernel.ResourceRequirements) and CTA thread counts that fit within the
//     probed target SM limits and supported execution-group/TMEM capabilities.
//   - Opaque vendor kernels (e.g. cuBLASLt, cuDNN, or unverified vendor blobs) are treated
//     as ineligible until an executed and documented path verifies proper griddepcontrol
//     cooperation.
//
// # Provenance, Overlap, and Graph Update Boundaries
//
// CheckEligibility validates owner-declared planning constraints and static
// target/artifact bounds (recorded as ProvenanceOwnerDeclared). Source strings and raw
// artifact bytes never imply eligibility. Sum-of-resources arithmetic and programmatic
// graph edges do not guarantee runtime CTA overlap or performance gain, and admitted
// pairs must remain semantically correct when zero overlap occurs.
//
// Updates to instantiated graphs containing PDL edges remain governed by cudagraph's
// existing node-update constraints (fixed graph topology and edge policy, owned
// parameter layout, and compatible function signatures); arbitrary hot-swapping or
// edge-policy mutation after instantiation is not supported.
//
// # Graceful Fallback
//
// If either kernel is ineligible, Compose falls back gracefully to a standard CUDA graph
// completion edge, recording the explicit fallback reason while preserving strict numerical
// and semantic correctness.
package cudapdl
