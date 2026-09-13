// Package cupticheckpoint provides managed ownership and replay lifecycle for
// NVIDIA CUPTI checkpoints.
//
// Checkpoints save and restore device state across CUDA operations to support
// multi-pass metric profiling, reproducible kernel execution, and controlled replay.
//
// Collection is restricted to Linux amd64/arm64 and qualified CUPTI versions
// (24, 26, 28, 130001, 130101, 130200, 130202, 130301, 130400).
package cupticheckpoint
