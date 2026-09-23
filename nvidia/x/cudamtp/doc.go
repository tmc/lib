// Package cudamtp implements Multi-Token Prediction (MTP) and speculative tree
// decoding optimized via fused CUDA C++ megakernels on NVIDIA GPUs.
//
// MTPConfig configures parallel draft heads. MTPReference computes the CPU
// mathematical reference forward pass for multi-head projections.
// GenerateMTPCudaSource emits fused megakernel source for device execution.
package cudamtp
