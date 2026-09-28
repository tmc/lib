// Package nvvmx compiles NVVM IR with owned inputs and copied output.
// Compile returns PTX by default; the native -gen-lto option selects LTOIR
// for linking with nvjitlinkx. Lazy inputs can supply libdevice bitcode.
// The selected NVVM library determines supported IR versions and options.
// Compilation is synchronous and does not require a CUDA device context.
// Do not unload the native library while compilation or cleanup is pending.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// # Address-Space Inference
//
// [InferAddressSpaces] pre-scans function bodies in an NVVM IR module and
// infers whether local stack slots are exclusively written with CTA-local shared
// memory addresses (address space 3). When proven shared, the slot declaration
// in the entry block is changed from the generic default to addrspace(3). This
// enables LLVM/NVVM to lower memory accesses directly to ld.shared and st.shared
// instructions, eliminating cvta.shared conversions and generic load/store overhead.
// Inference is fail-closed: an unrecognized alias, atomic access, or escape leaves
// the slot generic. InferenceReport.Slots keys each entry by function and slot,
// in the form @function/%slot, so local register names cannot collide across functions.
//
// Inference uses a 3-element lattice over slots:
//
//	StateUnconstrained (top: no writes observed yet)
//	       |
//	       v
//	  StateShared (all observed writes are addrspace(3))
//	       |
//	       v
//	StateGenericRequired (bottom: absorbing, at least one write is non-shared)
//
// The lattice cannot oscillate and is guaranteed to terminate. Transitions are
// strictly monotonic downward along the finite partial order (height 2).
// StateGenericRequired is an absorbing state with no outgoing transitions.
// For a function with N slots, fixed-point iteration performs at most 2*N state
// changes and terminates in at most 2*N + 1 passes over the body.
//
// An explicit source declaration always wins over inference: a slot declared
// in addrspace(1) or addrspace(3) is never overridden. Cluster-mapped pointers
// (addrspace 7) are distinct from CTA-local shared memory and are never inferred
// to addrspace(3).
package nvvmx
