// Package cudatensor provides a fluent tensor-expression builder that emits
// fused CUDA source.
//
// The package does not compile, load, or execute the generated source. Compile
// currently supports float32 elementwise arithmetic and activations; it returns
// an error for operations whose CUDA lowering is not implemented.
package cudatensor
