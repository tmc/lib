// Package cudalmvet validates numerical accuracy and catches precision drift
// by comparing GPU activations against CPU reference implementations.
//
// In addition to tensor-level tolerance checks, cudalmvet provides a single-scalar
// trace-hash differential oracle and a schedule-perturbation driver.
//
// # Trace Hash Accumulator Contract
//
// The trace hash accumulates intermediate values into a single uint64 scalar
// using the 64-bit FNV-1a algorithm over little-endian bytes:
//   - Offset basis: 14695981039346656037 (0xcbf29ce484222325)
//   - Prime multiplier: 1099511628211 (0x100000001b3)
//   - Update rule per byte b: hash = (hash ^ uint64(b)) * Prime
//
// # Float Policies
//
// The accumulator defines explicit policies for floating-point values:
//   - [PolicyRawBits] (default): Values supplied to the accumulator are folded
//     using their exact IEEE-754 little-endian byte patterns. Under this policy,
//     +0.0 (0x00000000) and -0.0 (0x80000000) produce distinct hashes, and
//     differing NaN payloads produce distinct hashes.
//   - [PolicyCanonicalizeNaN]: Normalizes all NaN representations to the canonical
//     quiet-NaN bit pattern (0x7fc00000 for float32, 0x7ff8000000000000 for float64)
//     prior to hashing. Signed zeros remain distinct.
//   - [PolicyCanonicalizeZero]: Normalizes -0.0 to +0.0 before hashing.
//   - [PolicyCanonicalizeAll]: Normalizes both signed zeros and NaNs.
//
// Device comparison and schedule perturbations currently accept only
// [PolicyRawBits]. Other policies remain available to the CPU accumulator and
// are refused before compilation on the device path, with an Unmeasured result.
// Device comparison also refuses a graph whose CPU reference trace contains a
// NaN. cudaref may canonicalize NaN results or produce a different NaN value
// from the emitted CUDA operation, so RawBits on the accumulator does not make
// such a graph a valid CPU/device bitwise comparison.
//
// The CPU reference preserves and detects noncanonical NaN payloads. Device
// comparison refuses any graph whose reference trace contains a NaN, including
// when triage is disabled. Infinities alone do not cause this refusal.
//
// # Traversal and Fold Order
//
// The fold sequence follows a declared canonical traversal:
//   - For computation graphs ([cudair.Graph]), nodes are traversed in safe
//     sequential topological order ([cudair.Snapshot.Order]).
//   - Tensor elements are folded in canonical row-major flat index order (0, 1, ..., N-1).
//   - Values from every non-input intermediate node are folded, including graph outputs.
//   - Graph inputs, node metadata, and unit/empty tensors are skipped; skipped values fold
//     zero bytes and leave the accumulator unchanged.
//
// Generated device kernels compute each graph node across threads in one CTA.
// Each thread completes its node value before entering the ordered fold; one CTA
// serializes node-major, then row-major fold tickets before updating the scalar
// hash. A node's fold is an ordering point: a thread cannot compute node k+1
// until its node-k ticket is served, and that ticket waits for all earlier
// tickets. Computation of later nodes is therefore staggered in ticket order.
// Generated kernels are elementwise and have no inter-thread data dependence.
// Race observability concerns caller-supplied kernels and their instrumentation.
// The generated harness refuses tensors or inputs above 1,024 elements.
// Caller-supplied kernels do not receive the generated kernel's fold-limit argument
// and are not localized to a graph fold index.
//
// # Schedule Perturbation
//
// The schedule-perturbation driver recompiles kernels under a bounded set of
// scheduling-affecting compiler options:
//   - NVRTC phase: --extra-device-vectorization (enabled / disabled).
//   - Driver JIT phase: CU_JIT_OPT_LEVEL 1 through 4. Level 0 is unsupported
//     because the JIT option type uses zero to mean unset.
//
// Each perturbation is compiled into an independent artifact with its own cache identity.
// A perturbation whose trace hash moves records a difference; it does not identify
// the cause. For a localized float32 mismatch, the report includes both values,
// their ULP distance, and whether the distance is admissible under the bit-exact
// policy (only zero ULP is admissible). Unlocalized mismatches report DiffIndex -1
// and have no numerical interpretation. Floating-point contraction such as FMA
// may therefore be reported as a disagreement when it changes an intermediate
// value, even if the contracted result is mathematically equivalent. The device
// reference evaluates exp and log in float64 and rounds once, while CUDA expf and
// logf are not correctly rounded; bit-exact comparison can report a disagreement
// of a few ULP for those operations.
//
// ULP distance uses IEEE totalOrder positions. It counts +0 and -0 as one ULP
// apart, and -min_denormal to +min_denormal as three. Other tools may use a
// different convention. ULP distance is unknown when either value is an
// infinity or NaN.
package cudalmvet
