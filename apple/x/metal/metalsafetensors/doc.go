//go:build darwin

// Package metalsafetensors reads safetensors files and copies selected tensors
// into shared Metal buffers.
//
// Open validates the complete tensor layout before mapping the file. A Weights
// must remain open while its tensor methods are in use. Buffers returned by the
// Metal methods own copies of tensor data and remain valid after Weights.Close;
// callers must free those buffers.
package metalsafetensors
