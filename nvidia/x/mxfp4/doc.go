// Package mxfp4 provides OCP Microscaling FP4 (E2M1) quantization,
// dequantization, and CUDA fused GEMV kernels.
//
// MXFP4 packs two 4-bit floating-point weights per byte with per-group (typically 32 elements)
// floating-point scale factors. Actual storage and bandwidth savings depend on
// scale representation, padding, kernel shape, and device execution.
//
// The CUDA engine is experimental and has no production consumer in this
// repository. Its numerical tests and opt-in hardware benchmark do not establish
// model-level accuracy or performance; adoption requires a matched
// model-workload campaign against the production precision path.
package mxfp4
