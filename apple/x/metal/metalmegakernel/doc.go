// Package metalmegakernel generates experimental Metal kernels and provides
// reusable kernels used by the llama inference example.
//
// GenerateLayerMegakernelMSL emits an RMSNorm and feed-forward path. It does
// not implement attention or a complete transformer layer, despite the
// generated kernel's historical name.
package metalmegakernel
