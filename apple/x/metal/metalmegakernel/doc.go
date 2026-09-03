// Package metalmegakernel provides experimental Metal kernels used by the
// llama inference example.
//
// LlamaDecoderKernelsMSL and HalfWeightKernelsMSL contain discrete kernels and
// narrowly fused projection kernels. GenerateFusedFFNMSL emits one fused
// RMSNorm and feed-forward kernel. None is a semantically complete transformer
// megakernel: attention, KV-cache updates, and sampling remain separate.
package metalmegakernel
