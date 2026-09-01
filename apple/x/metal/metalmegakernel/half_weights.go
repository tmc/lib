//go:build darwin

package metalmegakernel

import (
	_ "embed"
)

// HalfWeightKernelsMSL contains decoder kernel variants that read
// half-precision weights while keeping float32 activations and
// accumulation. Only kernels that touch weights have variants; RoPE,
// SiLU-mul, and attention operate on activations and are shared with
// the float32 path.
//
//go:embed half_weights.metal
var HalfWeightKernelsMSL string
