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
// such as sm_90, sm_90a, sm_100, sm_100a, sm_121, sm_121a). Target architectures below
// sm_90 are rejected during eligibility inspection and fall back to standard completion edges.
//
// # Kernel Requirements & Vendor Kernels
//
// To participate in a programmatic dependent launch:
//   - The producer kernel must emit griddepcontrol.launch_dependents in its execution path.
//   - The consumer kernel must emit griddepcontrol.wait in its execution path prior to
//     consuming producer outputs.
//   - Opaque vendor kernels (e.g. cuBLASLt, cuDNN, or unverified vendor blobs) are treated
//     as ineligible until an executed and documented path verifies proper griddepcontrol
//     cooperation.
//
// # Graceful Fallback
//
// If either kernel is ineligible, Compose falls back gracefully to a standard CUDA graph
// completion edge, recording the explicit fallback reason while preserving strict numerical
// and semantic correctness.
package cudapdl
