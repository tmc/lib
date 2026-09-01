//go:build darwin

package metalmegakernel

import (
	_ "embed"
)

// LlamaDecoderKernelsMSL contains high-throughput 2D tiled GEMV compute shaders
// for fast GPU-resident transformer inference on Apple Silicon.
//
//go:embed llama_kernels.metal
var LlamaDecoderKernelsMSL string
