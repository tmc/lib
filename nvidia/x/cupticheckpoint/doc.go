// Package cupticheckpoint provides managed ownership and replay lifecycle for
// NVIDIA CUPTI checkpoints.
//
// Checkpoints save and restore device state across CUDA operations to support
// multi-pass metric profiling, reproducible kernel execution, and controlled replay.
//
// Collection is restricted to Linux amd64/arm64 and qualified CUPTI versions
// (24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400).
// A later release of the same CUDA major uses the newest of those layouts at
// or below it; a new major or an older release is refused.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cupticheckpoint
