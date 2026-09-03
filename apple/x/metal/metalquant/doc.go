// SPDX-License-Identifier: Apache-2.0

// Package metalquant encodes and decodes Q4_0 and Q8_0 blocks used by Metal kernels.
// Each block contains 32 values and an IEEE binary16 scale. Functions reject
// partial blocks, non-finite values and scales, and trailing encoded bytes.
package metalquant
