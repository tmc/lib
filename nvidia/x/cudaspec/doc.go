// Package cudaspec provides high-throughput speculative decoding algorithms,
// prompt-cache n-gram drafting, and batched multi-token GPU verification.
//
// Speculative decoding drafts K candidate tokens cheaply and verifies them
// in a single batched target model forward pass, achieving a 2-3x wall-clock speedup.
package cudaspec
