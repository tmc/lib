// Package cudaprepack transforms host tensor storage and describes its byte layout.
//
// It provides padded row-major storage, quantized format conversions, separate
// scale buffers, sparsity metadata and binary storage. Q8_0 repacking preserves
// each block's quantization bytes and two-byte scale entry without arithmetic.
//
// Host layout or descriptor validation does not qualify CUDA tensor-map encoding,
// TMA transfers, device alignment or bank-conflict behavior. Device allocation,
// descriptor construction and execution require separate validation by callers.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudaprepack
