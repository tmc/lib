// Package marker encodes and parses explicit application metadata carried by
// existing NVTX ranges. It requires no profiler or GPU to encode messages.
//
// Messages use Prefix followed by a JSON object. Phase is required; token,
// layer, operation, dtype, cache_length, lookahead, materialization, and event
// are optional. Token and layer
// are zero-based nonnegative indices. Cache length is nonnegative; dtype is an
// application-defined nonempty string. Optional fields describe only the named
// range: parsers must not silently inherit missing values from parent ranges.
//
// Event is delivered, eos, or token_limit and requires a token index. Emit it
// as an instantaneous NVTX marker at the observed event, not as a range. A
// delivered event does not establish which GPU work determined its latency.
//
// Version 3 describes components with a ComponentContract and explicit
// invocation, batch and root coordinates. Model token and cache fields are
// not applicable. Component batch events report application lifecycle and
// counters; they do not prove native GPU completion.
//
// Metadata is an observed application label. It is distinct from inferred
// arg_reduce boundaries and does not establish equivalent kernel workloads.
// NVTX ranges describe host execution. Their timestamps use the collector's
// NVTX clock mapping, not a clock encoded in the message. A host range does not
// imply that asynchronous GPU execution ends when the range ends.
//
// Use nested NVTX push/pop ranges on a single OS thread. In Go, lock the thread
// across both calls. Concurrent threads may overlap; ranges on one thread must
// be nested or disjoint. This package validates message contents, not temporal
// nesting or start/end pairing. Missing, crossing, unpaired, or cross-thread
// ranges must remain visible in the collector's pairing receipts. Operations
// submitted by another thread need separate explicit correlation evidence.
//
// NVTX calls are no-ops without an injection provider, so applications remain
// usable with instrumentation disabled. Encoding still allocates and costs CPU
// time; instrumentation overhead has not been measured. Avoid encoding in a
// disabled path when that overhead matters.
//
// This package is outside the compatibility promise of github.com/tmc/lib/nvidia.
package marker
