// Package mxfp4 provides OCP Microscaling FP4 (E2M1) quantization,
// dequantization, and CUDA fused GEMV kernels.
//
// MXFP4 packs two 4-bit floating-point weights per byte with per-group (typically 32 elements)
// floating-point scale factors. Actual storage and bandwidth savings depend on
// scale representation, padding, kernel shape, and device execution.
package mxfp4
