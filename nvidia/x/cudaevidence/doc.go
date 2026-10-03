// Package cudaevidence classifies CUDA observations and the conclusions drawn
// from them. Values retain producer, artifact, compiler and workload identity.
// The zero Value is Unread, with no numeric value.
//
// New copies its input. Record returns a copy, so Values can be shared between
// goroutines. Decode validates JSON with the same rules as New. CheckClaim
// checks an existing classification and identity; it cannot authenticate a
// producer or prove an observation correct.
package cudaevidence
