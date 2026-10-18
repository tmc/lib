// Package cudaspec provides high-throughput speculative decoding algorithms,
// explicit branch state, tree attention masks, proposal lifecycle management,
// prompt-cache n-gram drafting, and batched multi-token verification.
//
// Verification compares draft tokens against caller-supplied target outputs or
// sequential single-token step callbacks on the host. Candidate branches and KV blocks
// follow explicit commit and rollback invariants preserving reference counts.
package cudaspec
