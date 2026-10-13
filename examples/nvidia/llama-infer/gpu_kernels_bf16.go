package main

import _ "embed"

// forwardBF16CUDA holds the BF16-weight kernels. It is compiled after
// ForwardKernelsCUDA in the same module and uses its device helpers.
//
//go:embed forward_bf16.cu
var forwardBF16CUDA string
