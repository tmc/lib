// Package clangcuda provides a small purego wrapper around libclang for
// parsing CUDA translation units.
//
// It accepts unsaved .cu and .cuh files, so callers can analyze sources held
// by an embed.FS without writing them to disk. The package exposes only the
// declarations and diagnostics needed by static CUDA tooling; nvcc or NVRTC
// remains the authority for compilation.
package clangcuda
