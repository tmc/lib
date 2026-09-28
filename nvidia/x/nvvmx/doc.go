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
// The supported forms are simple pointer-valued alloca slots; non-volatile,
// direct loads and stores of the slot's declared pointer type; shared values
// from a global declared in addrspace(3), an addrspace(3) parameter, a scalar
// getelementptr with simple integer indices whose base is known shared, or an
// addrspacecast from a known shared value to generic. A stored value must
// already have a shared pointer type, come from a dependent slot that narrows
// with its destination, or be a supported inline constant addrspacecast that
// can be removed during rewriting. A generic SSA cast result does not prove
// that its value can be stored in a narrowed slot. Parameter attributes do not
// change the address space of the parameter's leading pointer type. A loaded
// pointer may only be used as a load or store pointer operand, or stored into
// another proven-shared slot; dependent slots narrow or remain generic together.
// Typed-pointer slots are limited to simple scalar pointer types. Volatile,
// atomic, metadata-bearing, indirect, malformed, or otherwise unlisted uses
// keep the slot generic. The parser checks complete LLVM identifiers and all
// uses in a function before rewriting; unknown instruction forms fail closed.
// After rewriting, the function is parsed again and each narrowed slot access
// and loaded-value use is checked for the required address-space type. If any
// rewrite does not match its parsed operation, the original function is kept
// unchanged and all inferred slots in it remain generic. This fallback also
// declines spellings that the rewriter does not yet normalize, such as explicit
// addrspace(0) pointer operands or extra whitespace in a load or store.
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
