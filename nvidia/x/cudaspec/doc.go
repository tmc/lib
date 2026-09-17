// Package cudaspec provides high-throughput speculative decoding algorithms,
// explicit branch state, tree attention masks, proposal lifecycle management,
// prompt-cache n-gram drafting, and batched multi-token verification.
//
// Speculative decoding drafts candidate tokens cheaply and verifies them
// in a single batched target model forward pass. Candidate branches and KV blocks
// follow explicit commit and rollback invariants preserving reference counts.
package cudaspec
