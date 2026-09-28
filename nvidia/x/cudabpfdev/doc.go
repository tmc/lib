// Package cudabpfdev verifies and translates bounded warp instrumentation programs.
//
// The version-1 device bytecode contract is specific to this package. It is
// not Linux eBPF encoding or a kernel-registered BPF program type. Registers
// hold 32-bit unsigned integers, start at zero, and arithmetic wraps modulo
// 2^32. Branches test nonzero; only warp-uniform values may control branches
// or map updates. Ballot and Leader explicitly aggregate active-lane values.
// MapAdd adds a uniform value to a uniform slot once per active warp.
// Injected PTX samples activemask at the selected site, so after divergent
// control flow the aggregation can cover a converged subgroup rather than the
// whole warp. Policies that require full-warp attribution must select sites
// that execute convergently.
//
// Repeat closes a single-entry bounded loop whose header is Target and whose
// total iteration count is Imm. Loops may nest, but cannot overlap or admit
// jumps across their boundaries. No unbounded backward branch, arbitrary
// memory access, recursion, barrier, allocation or helper call is admitted.
// Exit returns register A. Verification establishes only these host-side
// properties, not compilation, injection safety or GPU behavior.
//
// PTX injection is opt-in through an exact kernel identity allowlist. Stub
// calls may be selected through InjectionPolicy.StubFunctions only when the
// listed device function is defined, has no parameters or returns, and has an
// empty body. Bodiless declarations and functions with other bodies are
// rejected. Prepared PTX is parseable by cudaptx; this does not establish that
// a native assembler accepts it or that it behaves correctly on a GPU.
package cudabpfdev
