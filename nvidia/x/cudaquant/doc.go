// SPDX-License-Identifier: Apache-2.0

// Package cudaquant decodes Q4_0 and Q8_0 blocks used by CUDA kernels.
// Each block contains 32 values and an IEEE binary16 scale. Decoders reject
// partial blocks, non-finite scales, and trailing encoded bytes.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaquant
