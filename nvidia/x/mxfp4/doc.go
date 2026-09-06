// Package mxfp4 provides OCP Microscaling FP4 (E2M1) quantization,
// dequantization, and CUDA fused GEMV kernels.
//
// MXFP4 packs two 4-bit floating-point weights per byte with per-group (typically 32 elements)
// floating-point scale factors. Actual storage and bandwidth savings depend on
// scale representation, padding, kernel shape, and device execution.
//
// The element encoding is OCP E2M1, but the block scale is not: the OCP
// Microscaling specification stores one E8M0 byte per block, a power of two,
// while this package stores an arbitrary float32 (the group's maximum
// magnitude divided by 6). Data written here therefore does not interchange
// with a conformant MXFP4 consumer in either direction, and a 32-element group
// costs 20 bytes rather than 17. The two representations are otherwise
// equivalent in accuracy; the choice is a compatibility one, not a numerical
// one, and it has not been made.
//
// The CUDA engine is experimental and has no production consumer in this
// repository. Its numerical tests and opt-in hardware benchmark do not establish
// model-level accuracy or performance; adoption requires a matched
// model-workload campaign against the production precision path.
package mxfp4
