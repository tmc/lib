// Package mxfp4 provides OCP Microscaling FP4 (E2M1) quantization, dequantization,
// and high-throughput CUDA fused GEMV kernels.
//
// MXFP4 packs two 4-bit floating-point weights per byte with per-group (typically 32 elements)
// floating-point scale factors, reducing weight memory bandwidth by approximately 4x.
package mxfp4
