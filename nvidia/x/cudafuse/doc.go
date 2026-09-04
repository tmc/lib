// Package cudafuse generates CUDA C++ source for fused float32 elementwise
// graphs. It rejects operations whose lowering is not implemented, including
// matrix multiplication, RMS normalization, rotary embedding, and reductions.
//
// The package does not compile, load, or execute the generated source. Callers
// are responsible for validating it with a CUDA compiler and execution runtime.
package cudafuse
